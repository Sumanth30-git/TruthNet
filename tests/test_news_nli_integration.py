from datetime import datetime, timezone

import pytest

from backend.schemas import (
    EvidenceAnalysisStatus,
    EvidenceStance,
    NLIEvidenceSignal,
    NLIProbabilities,
    SearchResponse,
    SearchResult,
    SignalStatus,
)
from backend.services.evidence_analysis import EvidenceAnalysisService, NLIEvidenceAnalyzer


NOW = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
CLAIM = "Acme launched Product Y in Bengaluru today."
SOURCE = SearchResult(
    title="Acme launch in Bengaluru",
    url="https://www.reuters.com/world/acme-launch",
    snippet="Acme launched Product Y in Bengaluru today.",
    publisher="Reuters",
    published_at="2026-09-24T08:00:00Z",
    retrieved_at=NOW,
)


class FakeNLIAdapter:
    def __init__(self, signal: NLIEvidenceSignal) -> None:
        self.signal = signal
        self.calls: list[tuple[object, object]] = []

    def predict(self, premise: object, hypothesis: object) -> NLIEvidenceSignal:
        self.calls.append((premise, hypothesis))
        return self.signal


def _complete_signal(label: str) -> NLIEvidenceSignal:
    scores = {name: 0.07 for name in ("contradiction", "entailment", "neutral")}
    scores[label] = 0.86
    probabilities = NLIProbabilities(**scores)
    confidence = getattr(probabilities, label)
    return NLIEvidenceSignal(
        model_id="cross-encoder/nli-deberta-v3-base",
        model_version="main",
        prediction=label,
        status=SignalStatus.COMPLETE,
        probabilities=probabilities,
        selected_confidence=confidence,
        inference_time_ms=12.34,
    )


def _configure_pipeline(monkeypatch: pytest.MonkeyPatch, adapter: FakeNLIAdapter) -> None:
    from backend.pipelines import news_pipeline

    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "false")
    monkeypatch.setattr(
        news_pipeline.news_search,
        "search",
        lambda _query: SearchResponse(status="complete", provider="mock", results=[SOURCE]),
    )
    monkeypatch.setattr(
        news_pipeline,
        "evidence_analysis",
        EvidenceAnalysisService(NLIEvidenceAnalyzer(adapter)),
    )


@pytest.mark.parametrize(
    ("label", "expected_stance"),
    [
        ("entailment", EvidenceStance.SUPPORTS),
        ("contradiction", EvidenceStance.CONTRADICTS),
        ("neutral", EvidenceStance.NEUTRAL),
    ],
)
def test_news_pipeline_preserves_existing_nli_signal_and_step_2a_quality(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    expected_stance: EvidenceStance,
):
    from backend.pipelines import news_pipeline

    signal = _complete_signal(label)
    adapter = FakeNLIAdapter(signal)
    _configure_pipeline(monkeypatch, adapter)

    response = news_pipeline.analyze_news(CLAIM)

    assert adapter.calls == [(SOURCE.snippet, CLAIM)]
    assert response.evidence_analysis_status == EvidenceAnalysisStatus.COMPLETE
    assert len(response.evidence) == 1
    item = response.evidence[0]
    assert item.claim_id == "claim-1"
    assert item.source_url == SOURCE.url
    assert item.stance == expected_stance
    assert item.nli_signal is not None
    assert item.nli_signal.prediction == label
    assert item.nli_signal.probabilities == signal.probabilities
    assert item.nli_signal.selected_confidence == signal.selected_confidence
    assert item.nli_signal.inference_time_ms == signal.inference_time_ms
    assert item.evidence_quality is not None
    assert item.evidence_quality.source_profile.publisher == "Reuters"
    assert item.evidence_quality.match.relevance.value == "high"
    assert item.evidence_quality.freshness.value == "current"


def test_news_pipeline_keeps_failed_nli_conservative_and_retains_step_2a_quality(
    monkeypatch: pytest.MonkeyPatch,
):
    from backend.pipelines import news_pipeline

    signal = NLIEvidenceSignal(
        model_id="cross-encoder/nli-deberta-v3-base",
        model_version="main",
        prediction="unavailable",
        status=SignalStatus.FAILED,
        details={"message": "NLI model inference is currently unavailable."},
    )
    adapter = FakeNLIAdapter(signal)
    _configure_pipeline(monkeypatch, adapter)

    response = news_pipeline.analyze_news(CLAIM)

    assert adapter.calls == [(SOURCE.snippet, CLAIM)]
    assert response.evidence_analysis_status == EvidenceAnalysisStatus.PARTIAL
    item = response.evidence[0]
    assert item.claim_id == "claim-1"
    assert item.source_url == SOURCE.url
    assert item.stance == EvidenceStance.INSUFFICIENT
    assert item.analysis_status == EvidenceAnalysisStatus.UNAVAILABLE
    assert item.nli_signal is not None
    assert item.nli_signal.status == SignalStatus.FAILED
    assert item.nli_signal.probabilities is None
    assert item.nli_signal.selected_confidence is None
    assert item.evidence_quality is not None
    assert item.evidence_quality.match.relevance.value == "high"
    assert item.evidence_quality.source_profile.publisher == "Reuters"
