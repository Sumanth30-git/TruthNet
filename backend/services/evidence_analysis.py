"""Conservative claim-to-source evidence analysis boundary.

Sentence splitting supplies candidate claims only. The default analyzer deliberately
does not infer relevance or stance from word overlap; a validated analyzer can be
injected later without changing the News pipeline contract.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol
from urllib.parse import urlparse

from backend.schemas import (
    Claim,
    EvidenceAnalysisStatus,
    EvidenceItem,
    EvidenceRelevance,
    EvidenceStance,
    NLIEvidenceSignal,
    SearchResult,
    SignalStatus,
)


class EvidenceAnalyzer(Protocol):
    """Replaceable boundary for a future evaluated stance/relevance model."""

    def analyze(self, claim: str, source_content: str) -> "EvidenceAssessment": ...


class EvidenceAssessment:
    def __init__(
        self,
        *,
        stance: EvidenceStance,
        relevance: EvidenceRelevance,
        summary: str,
        reason: str,
        status: EvidenceAnalysisStatus = EvidenceAnalysisStatus.COMPLETE,
        nli_signal: NLIEvidenceSignal | None = None,
    ) -> None:
        self.stance = stance
        self.relevance = relevance
        self.summary = summary
        self.reason = reason
        self.status = status
        self.nli_signal = nli_signal


class ConservativeEvidenceAnalyzer:
    """Safe baseline until a validated NLI/relevance model is integrated."""

    def analyze(self, claim: str, source_content: str) -> EvidenceAssessment:
        if not source_content.strip():
            return EvidenceAssessment(
                stance=EvidenceStance.INSUFFICIENT,
                relevance=EvidenceRelevance.NOT_ASSESSED,
                summary="The source candidate contains no usable text.",
                reason="No source content was available for analysis.",
                status=EvidenceAnalysisStatus.INSUFFICIENT,
            )
        return EvidenceAssessment(
            stance=EvidenceStance.INSUFFICIENT,
            relevance=EvidenceRelevance.NOT_ASSESSED,
            summary="Source content was retrieved, but its relationship to the claim was not established.",
            reason="No validated claim relevance or stance analyzer is configured.",
            status=EvidenceAnalysisStatus.INSUFFICIENT,
        )


class NLIAdapter(Protocol):
    def predict(self, premise: object, hypothesis: object) -> NLIEvidenceSignal: ...


class NLIEvidenceAnalyzer:
    """Maps a source-text/claim NLI signal to an evidence stance.

    Relevance remains independently not assessed: an NLI label describes the
    source-text premise's relationship to the claim hypothesis, not source quality.
    """

    def __init__(self, adapter: NLIAdapter | None = None) -> None:
        if adapter is None:
            from backend.models.nli_model import NLIModelAdapter

            adapter = NLIModelAdapter()
        self.adapter = adapter

    def analyze(self, claim: str, source_content: str) -> EvidenceAssessment:
        signal = self.adapter.predict(source_content, claim)
        if signal.status == SignalStatus.NOT_RUN:
            return EvidenceAssessment(
                stance=EvidenceStance.INSUFFICIENT,
                relevance=EvidenceRelevance.NOT_ASSESSED,
                summary="The source candidate was not analyzed because NLI model inference is disabled.",
                reason="Enable model inference to run NLI stance analysis.",
                status=EvidenceAnalysisStatus.INSUFFICIENT,
                nli_signal=signal,
            )
        if signal.status != SignalStatus.COMPLETE:
            return EvidenceAssessment(
                stance=EvidenceStance.INSUFFICIENT,
                relevance=EvidenceRelevance.NOT_ASSESSED,
                summary="The source candidate could not be analyzed by the NLI model.",
                reason="NLI stance analysis was unavailable for this source candidate.",
                status=EvidenceAnalysisStatus.UNAVAILABLE,
                nli_signal=signal,
            )
        mapping = {
            "entailment": EvidenceStance.SUPPORTS,
            "contradiction": EvidenceStance.CONTRADICTS,
            "neutral": EvidenceStance.NEUTRAL,
        }
        stance = mapping.get(signal.prediction)
        if stance is None:
            return EvidenceAssessment(
                stance=EvidenceStance.INSUFFICIENT,
                relevance=EvidenceRelevance.NOT_ASSESSED,
                summary="The NLI model returned an unrecognized relation label.",
                reason="The NLI result could not be mapped to an evidence stance.",
                status=EvidenceAnalysisStatus.UNAVAILABLE,
                nli_signal=signal,
            )
        return EvidenceAssessment(
            stance=stance,
            relevance=EvidenceRelevance.NOT_ASSESSED,
            summary=f"NLI classified the source-text/claim relationship as {signal.prediction}.",
            reason="NLI assesses textual relation only; it does not establish source credibility or factual truth.",
            status=EvidenceAnalysisStatus.COMPLETE,
            nli_signal=signal,
        )


def extract_candidate_claims(text: object) -> list[Claim]:
    """Split input into simple sentence candidates without semantic claim extraction."""
    if not isinstance(text, str) or not text.strip():
        return []
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    return [Claim(claim_id=f"claim-{index}", text=part) for index, part in enumerate(parts, 1)]


class EvidenceAnalysisService:
    def __init__(self, analyzer: EvidenceAnalyzer | None = None) -> None:
        self.analyzer = analyzer or NLIEvidenceAnalyzer()

    def analyze(
        self,
        claims: Sequence[Claim],
        sources: Sequence[SearchResult],
    ) -> tuple[list[EvidenceItem], EvidenceAnalysisStatus]:
        if not sources:
            return [], EvidenceAnalysisStatus.NO_SOURCES
        if not claims:
            return [], EvidenceAnalysisStatus.INSUFFICIENT

        evidence: list[EvidenceItem] = []
        statuses: list[EvidenceAnalysisStatus] = []
        for claim in claims:
            for raw_source in sources:
                try:
                    source = raw_source if isinstance(raw_source, SearchResult) else SearchResult.model_validate(raw_source)
                    parsed_url = urlparse(source.url)
                    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                        raise ValueError("invalid source URL")
                except Exception:
                    evidence.append(EvidenceItem(
                        kind="news_source_candidate",
                        summary="This source candidate is malformed and could not be analyzed.",
                        claim_id=claim.claim_id,
                        claim=claim.text,
                        stance=EvidenceStance.INSUFFICIENT,
                        relevance=EvidenceRelevance.NOT_ASSESSED,
                        analysis_status=EvidenceAnalysisStatus.UNAVAILABLE,
                        reason="The source candidate did not contain valid structured metadata.",
                    ))
                    statuses.append(EvidenceAnalysisStatus.UNAVAILABLE)
                    continue
                content = source.snippet if isinstance(source.snippet, str) else ""
                try:
                    assessment = self.analyzer.analyze(claim.text, content)
                    status = assessment.status
                    item = EvidenceItem(
                        kind="news_source_candidate",
                        summary=assessment.summary,
                        source=source.publisher or source.url,
                        claim_id=claim.claim_id,
                        claim=claim.text,
                        source_title=source.title,
                        source_url=source.url,
                        publisher=source.publisher,
                        published_at=source.published_at,
                        retrieved_at=source.retrieved_at,
                        stance=assessment.stance,
                        relevance=assessment.relevance,
                        analysis_status=status,
                        reason=assessment.reason,
                        nli_signal=assessment.nli_signal,
                    )
                except Exception:
                    # A single analyzer/source failure must not discard other candidates.
                    status = EvidenceAnalysisStatus.UNAVAILABLE
                    item = EvidenceItem(
                        kind="news_source_candidate",
                        summary="This source candidate could not be analyzed.",
                        source=source.publisher or source.url,
                        claim_id=claim.claim_id,
                        claim=claim.text,
                        source_title=source.title,
                        source_url=source.url,
                        publisher=source.publisher,
                        published_at=source.published_at,
                        retrieved_at=source.retrieved_at,
                        stance=EvidenceStance.INSUFFICIENT,
                        relevance=EvidenceRelevance.NOT_ASSESSED,
                        analysis_status=status,
                        reason="Evidence analysis failed for this source candidate.",
                    )
                evidence.append(item)
                statuses.append(status)

        if all(status == EvidenceAnalysisStatus.COMPLETE for status in statuses):
            overall = EvidenceAnalysisStatus.COMPLETE
        elif all(status == EvidenceAnalysisStatus.INSUFFICIENT for status in statuses):
            overall = EvidenceAnalysisStatus.INSUFFICIENT
        else:
            overall = EvidenceAnalysisStatus.PARTIAL
        return evidence, overall


evidence_analysis = EvidenceAnalysisService()
