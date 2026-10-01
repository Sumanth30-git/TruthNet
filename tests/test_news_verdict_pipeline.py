from datetime import datetime, timezone

import pytest

from backend.fusion.decision import NewsVerdictDecision
from backend.models.base import BaseModelAdapter
from backend.models.registry import registry
from backend.schemas import (
    ContentType,
    EvidenceAnalysisStatus,
    EvidenceItem,
    EvidenceStance,
    EvidenceSummary,
    ModelInfo,
    SearchResponse,
    SearchResult,
    SignalResult,
    SignalStatus,
    Uncertainty,
    Verdict,
)


NOW = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
NEWS_MODEL_ID = "jy46604790/Fake-News-Bert-Detect"
SOURCE = SearchResult(
    title="Acme launch report",
    url="https://www.reuters.com/world/acme-launch",
    snippet="Acme launched Product Y in Bengaluru today.",
    publisher="Reuters",
    published_at="2026-09-24T08:00:00Z",
    retrieved_at=NOW,
)
ONE_CLAIM = "Acme launched Product Y in Bengaluru today."


class ScriptedNewsAdapter(BaseModelAdapter):
    metadata = ModelInfo(
        model_id=NEWS_MODEL_ID,
        model_version="main",
        content_type=ContentType.TEXT,
    )

    def __init__(self, signal: SignalResult) -> None:
        self.signal = signal
        self.predict_calls: list[str] = []

    def load(self) -> None:
        return None

    def predict(self, payload: str) -> SignalResult:
        self.predict_calls.append(payload)
        return self.signal


class FakeAggregation:
    def __init__(self, summaries: list[EvidenceSummary] | None = None, *, fail: bool = False) -> None:
        self.summaries = summaries or []
        self.fail = fail
        self.prepare_calls: list[list] = []
        self.aggregate_calls: list[tuple[list, list]] = []

    def prepare(self, evidence):
        self.prepare_calls.append(evidence)
        return evidence

    def aggregate(self, claims, evidence):
        self.aggregate_calls.append((claims, evidence))
        if self.fail:
            raise RuntimeError("aggregation unavailable")
        return self.summaries


def _bert_signal(prediction: str) -> SignalResult:
    return SignalResult(
        model_id=NEWS_MODEL_ID,
        model_version="main",
        prediction=prediction,
        raw_score=0.99,
        calibrated_score=None,
        content_type=ContentType.TEXT,
        status=SignalStatus.COMPLETE,
        inference_time_ms=12.5,
    )


def _summary(claim_id: str) -> EvidenceSummary:
    return EvidenceSummary(claim_id=claim_id, overall_strength="insufficient", reason="Mocked aggregate.")


def _evidence(claim_id: str, claim_text: str) -> EvidenceItem:
    return EvidenceItem(
        kind="news_source_candidate",
        summary="Mocked candidate evidence.",
        source=SOURCE.publisher,
        claim_id=claim_id,
        claim=claim_text,
        source_title=SOURCE.title,
        source_url=SOURCE.url,
        source_snippet=SOURCE.snippet,
        publisher=SOURCE.publisher,
        published_at=SOURCE.published_at,
        retrieved_at=SOURCE.retrieved_at,
        stance=EvidenceStance.SUPPORTS,
        analysis_status=EvidenceAnalysisStatus.COMPLETE,
    )


def _decision(
    verdict: Verdict,
    *,
    confidence: float | None,
    uncertainty: Uncertainty,
) -> NewsVerdictDecision:
    return NewsVerdictDecision(
        verdict=verdict,
        confidence=confidence,
        uncertainty=uncertainty,
        reason="Mocked news verdict decision.",
    )


def _enable_bert(monkeypatch: pytest.MonkeyPatch, prediction: str) -> ScriptedNewsAdapter:
    adapter = ScriptedNewsAdapter(_bert_signal(prediction))
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "true")
    original_get = registry.get

    def get(key: str) -> BaseModelAdapter:
        if key == "fake_news_baseline":
            return adapter
        return original_get(key)

    monkeypatch.setattr(registry, "get", get)
    return adapter


def _configure_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    decision: NewsVerdictDecision,
    *,
    summaries: list[EvidenceSummary] | None = None,
    search: SearchResponse | None = None,
    search_error: Exception | None = None,
    analysis_status: EvidenceAnalysisStatus = EvidenceAnalysisStatus.COMPLETE,
    evidence: list[EvidenceItem] | None = None,
    aggregation_fail: bool = False,
) -> list[dict]:
    from backend.pipelines import news_pipeline

    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "false")
    aggregation = FakeAggregation(summaries, fail=aggregation_fail)
    captured: list[dict] = []

    def fake_search(_query: str) -> SearchResponse:
        if search_error is not None:
            raise search_error
        return search or SearchResponse(status="complete", provider="mock", results=[SOURCE])

    def fake_analyze(claims, _results):
        items = evidence
        if items is None:
            items = [_evidence(claim.claim_id, claim.text) for claim in claims]
        return items, analysis_status

    def fake_decide(claims, evidence_summaries, **kwargs):
        captured.append(
            {
                "claims": claims,
                "evidence_summaries": evidence_summaries,
                **kwargs,
            }
        )
        return decision

    monkeypatch.setattr(news_pipeline.news_search, "search", fake_search)
    monkeypatch.setattr(news_pipeline.evidence_analysis, "analyze", fake_analyze)
    monkeypatch.setattr(news_pipeline, "evidence_aggregation", aggregation)
    monkeypatch.setattr(news_pipeline, "decide_news_verdict", fake_decide)
    return captured


def test_pipeline_applies_likely_authentic(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.pipelines import news_pipeline

    decision = _decision(Verdict.LIKELY_AUTHENTIC, confidence=0.37, uncertainty=Uncertainty.MEDIUM)
    summaries = [_summary("claim-1")]
    captured = _configure_pipeline(monkeypatch, decision, summaries=summaries)

    response = news_pipeline.analyze_news(ONE_CLAIM)

    assert captured[0]["aggregation_succeeded"] is True
    assert captured[0]["claims"] is response.claims
    assert captured[0]["evidence_summaries"] is response.evidence_summaries
    assert response.verdict == Verdict.LIKELY_AUTHENTIC
    assert response.confidence == 0.37
    assert response.uncertainty == Uncertainty.MEDIUM
    assert response.evidence_summaries == summaries
    assert len(response.evidence) == 1
    assert response.evidence[0].kind == "news_source_candidate"
    assert response.signals


def test_pipeline_applies_likely_fake_news(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.pipelines import news_pipeline

    decision = _decision(Verdict.LIKELY_FAKE_NEWS, confidence=0.41, uncertainty=Uncertainty.MEDIUM)
    summaries = [_summary("claim-1")]
    captured = _configure_pipeline(monkeypatch, decision, summaries=summaries)

    response = news_pipeline.analyze_news(ONE_CLAIM)

    assert captured[0]["aggregation_succeeded"] is True
    assert response.verdict == Verdict.LIKELY_FAKE_NEWS
    assert response.confidence == 0.41
    assert response.uncertainty == Uncertainty.MEDIUM
    assert response.evidence_summaries == summaries
    assert len(response.evidence) == 1


def test_pipeline_applies_inconclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.pipelines import news_pipeline

    decision = _decision(Verdict.INCONCLUSIVE, confidence=None, uncertainty=Uncertainty.HIGH)
    summaries = [_summary("claim-1")]
    captured = _configure_pipeline(monkeypatch, decision, summaries=summaries)

    response = news_pipeline.analyze_news(ONE_CLAIM)

    assert captured[0]["aggregation_succeeded"] is True
    assert response.verdict == Verdict.INCONCLUSIVE
    assert response.confidence is None
    assert response.uncertainty == Uncertainty.HIGH
    assert response.evidence_summaries == summaries
    assert len(response.evidence) == 1


def test_pipeline_aggregation_failure_remains_inconclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.pipelines import news_pipeline

    decision = _decision(Verdict.INCONCLUSIVE, confidence=None, uncertainty=Uncertainty.NOT_ASSESSED)
    captured = _configure_pipeline(monkeypatch, decision, aggregation_fail=True)

    response = news_pipeline.analyze_news(ONE_CLAIM)

    assert captured[0]["aggregation_succeeded"] is False
    assert captured[0]["evidence_summaries"] == []
    assert response.verdict == Verdict.INCONCLUSIVE
    assert response.confidence is None
    assert response.uncertainty == Uncertainty.NOT_ASSESSED
    assert response.evidence_summaries == []
    assert len(response.evidence) == 1
    assert response.evidence[0].claim_id == "claim-1"


def test_pipeline_search_failure_remains_inconclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.pipelines import news_pipeline

    decision = _decision(Verdict.INCONCLUSIVE, confidence=None, uncertainty=Uncertainty.NOT_ASSESSED)
    captured = _configure_pipeline(
        monkeypatch,
        decision,
        search=SearchResponse(status="failed", provider="mock", results=[], message="Search failed."),
    )

    response = news_pipeline.analyze_news(ONE_CLAIM)

    assert captured[0]["search_status"] == "failed"
    assert captured[0]["aggregation_succeeded"] is False
    assert response.verdict == Verdict.INCONCLUSIVE
    assert response.confidence is None
    assert response.uncertainty == Uncertainty.NOT_ASSESSED
    assert response.evidence_summaries == []
    assert response.evidence == []
    assert [claim.claim_id for claim in response.claims] == ["claim-1"]


def test_pipeline_preserves_summaries_evidence_and_bert_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.pipelines import news_pipeline

    decision = _decision(Verdict.LIKELY_AUTHENTIC, confidence=0.19, uncertainty=Uncertainty.MEDIUM)
    summaries = [_summary("claim-1")]
    captured = _configure_pipeline(monkeypatch, decision, summaries=summaries)
    adapter = _enable_bert(monkeypatch, "fake_news")

    response = news_pipeline.analyze_news(ONE_CLAIM)

    assert adapter.predict_calls == [ONE_CLAIM]
    assert captured[0]["signals"] is response.signals
    assert captured[0]["signals"][0].prediction == "fake_news"
    assert response.verdict == Verdict.LIKELY_AUTHENTIC
    assert response.confidence == 0.19
    assert response.uncertainty == Uncertainty.MEDIUM
    assert response.evidence_summaries == summaries
    assert [item.claim_id for item in response.evidence] == ["claim-1"]
    assert response.evidence[0].kind == "news_source_candidate"
    assert response.signals[0].prediction == "fake_news"
    assert response.signals[0].status == SignalStatus.COMPLETE
    assert response.signals[0].raw_score == 0.99
