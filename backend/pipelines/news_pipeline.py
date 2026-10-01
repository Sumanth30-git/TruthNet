import logging
from typing import Any

from backend.config import model_inference_enabled
from backend.fusion.decision import decide_news_verdict, inconclusive_result
from backend.models.base import BaseModelAdapter
from backend.models.registry import ModelRegistry, registry
from backend.schemas import AnalysisResponse, ContentType, EvidenceQuality, SignalResult, SignalStatus
from backend.schemas import EvidenceAnalysisStatus
from backend.services.evidence_analysis import evidence_analysis, extract_candidate_claims
from backend.services.claim_understanding import claim_understanding
from backend.services.evidence_aggregation import evidence_aggregation
from backend.services.evidence_matching import evidence_matching
from backend.services.news_search import news_search
from backend.services.source_quality import source_quality

logger = logging.getLogger(__name__)

NEWS_LIMITATIONS = [
    "This is a single text-classifier signal, not live claim verification.",
    "Claims are simple sentence candidates; NLI assesses source-text/claim relation, not factual truth or source credibility.",
    "Evidence fusion has not run, so the final verdict remains inconclusive.",
    "Raw model scores are not calibrated probabilities or factual certainty.",
]


def _news_signal(
    *,
    adapter,
    prediction: str,
    status: SignalStatus,
    raw_score: float | None = None,
    inference_time_ms: float | None = None,
    details: dict[str, Any] | None = None,
) -> SignalResult:
    return SignalResult(
        model_id=adapter.metadata.model_id,
        model_version=adapter.metadata.model_version,
        prediction=prediction,
        raw_score=raw_score,
        calibrated_score=None,
        content_type=ContentType.TEXT,
        status=status,
        inference_time_ms=inference_time_ms,
        details=details or {},
    )


def _unavailable_claim_ids(response: AnalysisResponse) -> frozenset[str]:
    return frozenset(
        item.claim_id
        for item in response.evidence
        if item.claim_id is not None and item.analysis_status == EvidenceAnalysisStatus.UNAVAILABLE
    )


def _apply_news_verdict(response: AnalysisResponse, *, aggregation_succeeded: bool) -> AnalysisResponse:
    """Apply the existing News decision policy without recalculating evidence."""

    search_status = None if response.search is None else response.search.status
    decision = decide_news_verdict(
        response.claims,
        response.evidence_summaries,
        search_status=search_status,
        evidence_analysis_status=response.evidence_analysis_status,
        aggregation_succeeded=aggregation_succeeded,
        unavailable_claim_ids=_unavailable_claim_ids(response),
        signals=response.signals,
    )
    response.verdict = decision.verdict
    response.confidence = decision.confidence
    response.uncertainty = decision.uncertainty
    response.message = decision.reason
    stale_limitation = "Evidence fusion has not run, so the final verdict remains inconclusive."
    response.limitations = [item for item in response.limitations if item != stale_limitation]
    if decision.reason not in response.limitations:
        response.limitations.append(decision.reason)
    return response


def _attach_evidence_quality(response: AnalysisResponse) -> None:
    """Attach deterministic source and claim matching metadata to evidence items.

    Evidence analysis remains responsible for constructing the candidate items and
    its stance/relevance fields. This step only adds the Phase 3E source-quality
    and matching outputs for the corresponding claim/source pair.
    """
    if response.search is None:
        return

    claims_by_id = {claim.claim_id: claim for claim in response.claims}
    contexts = claim_understanding.understand_claims(response.claims)
    for context in contexts:
        claim = claims_by_id.get(context.claim_id)
        if claim is not None:
            claim.context = context

    items_by_pair: dict[tuple[str | None, str | None], list] = {}
    for item in response.evidence:
        items_by_pair.setdefault((item.claim_id, item.source_url), []).append(item)

    for context in contexts:
        for source in response.search.results:
            items = items_by_pair.get((context.claim_id, source.url), [])
            try:
                profile = source_quality.profile(source, claim_context=context)
                match = evidence_matching.match(context, source)
                freshness = evidence_matching.freshness(context, source)
            except Exception:
                # Matching metadata must never turn a retrieved source into a
                # finding or discard the conservative evidence-analysis result.
                logger.warning("Evidence matching failed for claim/source candidate", exc_info=True)
                continue

            for item in items:
                existing_quality = item.evidence_quality
                item.evidence_quality = EvidenceQuality(
                    source_profile=profile,
                    match=match,
                    numerical_check=(
                        existing_quality.numerical_check if existing_quality is not None else None
                    ),
                    freshness=freshness,
                )


def analyze_news(text: str, model_registry: ModelRegistry = registry) -> AnalysisResponse:
    adapter = model_registry.get("fake_news_baseline")
    if not model_inference_enabled():
        signal = _news_signal(
            adapter=adapter,
            prediction="not_assessed",
            status=SignalStatus.NOT_RUN,
            details={"message": "Model inference is disabled."},
        )
    else:
        try:
            signal = adapter.predict(text)
        except Exception:
            logger.exception("News baseline inference failed")
            signal = _news_signal(
                adapter=adapter,
                prediction="unavailable",
                status=SignalStatus.FAILED,
                details={"message": "Model inference is currently unavailable."},
            )

    response = inconclusive_result(
        "This is an individual text-classifier signal, not live claim verification.",
        [signal],
        model_versions=[adapter.metadata],
        limitations=NEWS_LIMITATIONS,
    )
    # Sentence segmentation creates candidate claims, not semantically verified claims.
    response.claims = extract_candidate_claims(text)
    aggregation_succeeded = False
    try:
        response.search = news_search.search(text)
    except Exception:
        logger.warning("News search failed")
        response.search = None
        response.evidence_analysis_status = EvidenceAnalysisStatus.UNAVAILABLE
        return _apply_news_verdict(response, aggregation_succeeded=False)

    if response.search.status == "complete":
        response.evidence, response.evidence_analysis_status = evidence_analysis.analyze(
            response.claims, response.search.results
        )
        _attach_evidence_quality(response)
        try:
            response.evidence = evidence_aggregation.prepare(response.evidence)
            response.evidence_summaries = evidence_aggregation.aggregate(
                response.claims, response.evidence
            )
            aggregation_succeeded = True
        except Exception:
            # Aggregation must not independently invent support or contradiction.
            logger.warning("Evidence aggregation failed", exc_info=True)
            response.evidence_summaries = []
    elif response.search.status in {"failed", "unavailable"}:
        response.evidence_analysis_status = EvidenceAnalysisStatus.UNAVAILABLE
    return _apply_news_verdict(response, aggregation_succeeded=aggregation_succeeded)
