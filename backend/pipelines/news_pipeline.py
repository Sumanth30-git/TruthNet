import logging
from typing import Any

from backend.config import model_inference_enabled
from backend.fusion.decision import inconclusive_result
from backend.models.base import BaseModelAdapter
from backend.models.registry import ModelRegistry, registry
from backend.schemas import AnalysisResponse, ContentType, SignalResult, SignalStatus

logger = logging.getLogger(__name__)

NEWS_LIMITATIONS = [
    "This is a single text-classifier signal, not live claim verification.",
    "No source retrieval, claim extraction, or evidence verification has run.",
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

    return inconclusive_result(
        "This is an individual text-classifier signal, not live claim verification.",
        [signal],
        model_versions=[adapter.metadata],
        limitations=NEWS_LIMITATIONS,
    )
