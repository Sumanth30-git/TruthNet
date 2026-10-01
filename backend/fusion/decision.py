"""Safe response defaults and pure, conservative News verdict decisions."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from backend.schemas import (
    AnalysisResponse,
    Claim,
    EvidenceAggregateState,
    EvidenceAnalysisStatus,
    EvidenceSummary,
    ModelInfo,
    SignalResult,
    Uncertainty,
    Verdict,
)


_STRONG_SUPPORT = frozenset({
    EvidenceAggregateState.STRONG_SUPPORT,
    EvidenceAggregateState.VERY_STRONG_SUPPORT,
})
_STRONG_CONTRADICTION = frozenset({
    EvidenceAggregateState.STRONG_CONTRADICTION,
    EvidenceAggregateState.VERY_STRONG_CONTRADICTION,
})


@dataclass(frozen=True)
class NewsVerdictDecision:
    """Pure policy output for a caller to apply to an existing response."""

    verdict: Verdict
    confidence: float | None
    uncertainty: Uncertainty
    reason: str


def _inconclusive(reason: str, *, unavailable: bool = False) -> NewsVerdictDecision:
    return NewsVerdictDecision(
        verdict=Verdict.INCONCLUSIVE,
        confidence=None,
        uncertainty=Uncertainty.NOT_ASSESSED if unavailable else Uncertainty.HIGH,
        reason=reason,
    )


def decide_news_verdict(
    claims: Sequence[Claim],
    evidence_summaries: Sequence[EvidenceSummary],
    *,
    search_status: str | None = "complete",
    evidence_analysis_status: EvidenceAnalysisStatus = EvidenceAnalysisStatus.COMPLETE,
    aggregation_succeeded: bool = True,
    unavailable_claim_ids: Collection[str] = (),
    signals: Sequence[SignalResult] = (),
) -> NewsVerdictDecision:
    """Decide strictly from claim summaries and processing availability.

    ``signals`` is accepted only so callers can retain model signals alongside
    the decision. It is intentionally not read: the uncalibrated BERT baseline
    must not influence an evidence-based News verdict.
    """
    del signals
    if search_status != "complete":
        return _inconclusive("Search processing was unavailable.", unavailable=True)
    if not aggregation_succeeded:
        return _inconclusive("Evidence aggregation did not complete.", unavailable=True)
    if evidence_analysis_status in {
        EvidenceAnalysisStatus.NOT_RUN,
        EvidenceAnalysisStatus.PARTIAL,
        EvidenceAnalysisStatus.UNAVAILABLE,
    }:
        return _inconclusive("Evidence analysis was unavailable for one or more claims.", unavailable=True)
    if unavailable_claim_ids:
        return _inconclusive("Required NLI evidence was unavailable for one or more claims.", unavailable=True)
    if evidence_analysis_status in {
        EvidenceAnalysisStatus.NO_SOURCES,
        EvidenceAnalysisStatus.INSUFFICIENT,
    }:
        return _inconclusive("No usable evidence analysis was available for the claims.")
    if not claims or not evidence_summaries:
        return _inconclusive("No usable evidence summaries were available for every claim.")

    summaries_by_claim: dict[str, list[EvidenceSummary]] = {}
    for summary in evidence_summaries:
        summaries_by_claim.setdefault(summary.claim_id, []).append(summary)
    claim_ids = {claim.claim_id for claim in claims}
    if set(summaries_by_claim) != claim_ids or any(
        len(summaries_by_claim[claim_id]) != 1 for claim_id in claim_ids
    ):
        return _inconclusive("Every extracted claim requires exactly one usable evidence summary.")

    states = [summaries_by_claim[claim.claim_id][0].overall_strength for claim in claims]
    if all(state in _STRONG_SUPPORT for state in states):
        return NewsVerdictDecision(
            verdict=Verdict.LIKELY_AUTHENTIC,
            confidence=None,
            uncertainty=Uncertainty.MEDIUM,
            reason="Every extracted claim has strong supporting evidence.",
        )
    if all(state in _STRONG_CONTRADICTION for state in states):
        return NewsVerdictDecision(
            verdict=Verdict.LIKELY_FAKE_NEWS,
            confidence=None,
            uncertainty=Uncertainty.MEDIUM,
            reason="Every extracted claim has strong contradicting evidence.",
        )
    return _inconclusive(
        "Claim evidence is weak, insufficient, conflicting, or does not agree across all claims."
    )


def inconclusive_result(
    message: str,
    signals: list[SignalResult] | None = None,
    model_versions: list[ModelInfo] | None = None,
    limitations: list[str] | None = None,
) -> AnalysisResponse:
    """Safe default until a calibrated, validated fusion policy exists."""
    return AnalysisResponse(
        verdict=Verdict.INCONCLUSIVE,
        confidence=None,
        uncertainty=Uncertainty.HIGH,
        signals=signals or [],
        model_versions=model_versions or [],
        limitations=[
            "No calibrated evidence fusion has run.",
            "A final result is intentionally not forced into authentic or fake.",
            *(limitations or []),
        ],
        message=message,
    )
