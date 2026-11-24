from datetime import datetime, timezone

import pytest

from backend.fusion.decision import decide_news_verdict
from backend.schemas import (
    EvidenceAnalysisStatus,
    EvidenceStance,
    NLIEvidenceSignal,
    NLIProbabilities,
    SearchResponse,
    SearchResult,
    SignalStatus,
    Verdict,
)
from backend.services.evidence_aggregation import EvidenceAggregationService
from backend.services.evidence_analysis import EvidenceAnalysisService, NLIEvidenceAnalyzer


NOW = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
SOURCES = [
    SearchResult(
        title="Acme launch report",
        url="https://www.reuters.com/world/acme-launch",
        snippet="Acme launched Product Y in Bengaluru today.",
        publisher="Reuters",
        published_at="2026-09-24T08:00:00Z",
        retrieved_at=NOW,
        claim_ids=["claim-1", "claim-2"],
    ),
    SearchResult(
        title="Conflicting Acme launch report",
        url="https://example.org/acme-launch",
        snippet="Contradiction: Acme launched Product Y in Bengaluru today.",
        publisher="Example",
        published_at="2026-09-24T09:00:00Z",
        retrieved_at=NOW,
        claim_ids=["claim-1", "claim-2"],
    ),
]


class FakeNLIAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def predict(self, premise: object, hypothesis: object) -> NLIEvidenceSignal:
        self.calls.append((premise, hypothesis))
        label = "contradiction" if isinstance(premise, str) and premise.startswith("Contradiction:") else "entailment"
        scores = {name: 0.05 for name in ("contradiction", "entailment", "neutral")}
        scores[label] = 0.9
        probabilities = NLIProbabilities(**scores)
        return NLIEvidenceSignal(
            model_id="test/nli",
            model_version="test",
            prediction=label,
            status=SignalStatus.COMPLETE,
            probabilities=probabilities,
            selected_confidence=getattr(probabilities, label),
            inference_time_ms=10,
        )


class RecordingAggregationService:
    def __init__(self) -> None:
        self.service = EvidenceAggregationService()
        self.prepare_calls: list[list] = []
        self.aggregate_calls: list[tuple[list, list]] = []

    def prepare(self, evidence):
        self.prepare_calls.append(evidence)
        return self.service.prepare(evidence)

    def aggregate(self, claims, evidence):
        self.aggregate_calls.append((claims, evidence))
        return self.service.aggregate(claims, evidence)


def _configure_pipeline(monkeypatch: pytest.MonkeyPatch, aggregation) -> FakeNLIAdapter:
    from backend.pipelines import news_pipeline

    nli = FakeNLIAdapter()
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "false")
    monkeypatch.setattr(
        news_pipeline.news_search,
        "search_claims",
        lambda _claims: SearchResponse(status="complete", provider="mock", results=SOURCES),
    )
    monkeypatch.setattr(
        news_pipeline,
        "evidence_analysis",
        EvidenceAnalysisService(NLIEvidenceAnalyzer(nli)),
    )
    monkeypatch.setattr(news_pipeline, "evidence_aggregation", aggregation)
    return nli


def test_news_pipeline_aggregates_enriched_evidence_per_claim(monkeypatch: pytest.MonkeyPatch):
    from backend.pipelines import news_pipeline

    aggregation = RecordingAggregationService()
    nli = _configure_pipeline(monkeypatch, aggregation)

    response = news_pipeline.analyze_news(
        "Acme launched Product Y in Bengaluru today. Acme released Product Z in Bengaluru today."
    )

    assert len(nli.calls) == 4
    assert aggregation.prepare_calls == [response.evidence]
    assert aggregation.aggregate_calls == [(response.claims, response.evidence)]
    assert {(item.claim_id, item.source_url) for item in response.evidence} == {
        ("claim-1", "https://www.reuters.com/world/acme-launch"),
        ("claim-1", "https://example.org/acme-launch"),
        ("claim-2", "https://www.reuters.com/world/acme-launch"),
        ("claim-2", "https://example.org/acme-launch"),
    }
    assert len(response.evidence) == 4
    assert all(item.evidence_quality is not None for item in response.evidence)
    assert all(item.nli_signal is not None for item in response.evidence)
    assert {item.stance for item in response.evidence} == {
        EvidenceStance.SUPPORTS,
        EvidenceStance.CONTRADICTS,
    }
    assert {item.evidence_quality.source_profile.publisher for item in response.evidence} == {"Reuters", "Example"}
    assert all(item.evidence_quality.match.relevance.value == "high" for item in response.evidence)
    assert all(item.evidence_quality.evidence_weight is not None for item in response.evidence)
    assert [summary.claim_id for summary in response.evidence_summaries] == ["claim-1", "claim-2"]
    assert all(summary.support_weight > 0 for summary in response.evidence_summaries)
    assert all(summary.contradiction_weight > 0 for summary in response.evidence_summaries)
    expected = decide_news_verdict(
        response.claims,
        response.evidence_summaries,
        search_status=response.search.status if response.search is not None else None,
        evidence_analysis_status=response.evidence_analysis_status,
        aggregation_succeeded=True,
        signals=response.signals,
    )
    assert response.verdict == expected.verdict
    assert response.confidence is None
    assert response.uncertainty == expected.uncertainty


def test_news_pipeline_handles_aggregation_failure_without_changing_evidence_or_verdict(
    monkeypatch: pytest.MonkeyPatch,
):
    from backend.pipelines import news_pipeline

    class FailingAggregationService:
        def prepare(self, _evidence):
            raise RuntimeError("aggregation unavailable")

    _configure_pipeline(monkeypatch, FailingAggregationService())

    response = news_pipeline.analyze_news("Acme launched Product Y in Bengaluru today.")

    assert response.evidence_analysis_status == EvidenceAnalysisStatus.COMPLETE
    assert response.evidence_summaries == []
    assert len(response.evidence) == 2
    assert all(item.nli_signal is not None for item in response.evidence)
    assert all(item.evidence_quality is not None for item in response.evidence)
    assert response.verdict == Verdict.INCONCLUSIVE


def test_numeric_mismatch_is_recorded_and_does_not_add_support(monkeypatch: pytest.MonkeyPatch):
    from backend.pipelines import news_pipeline

    aggregation = RecordingAggregationService()
    _configure_pipeline(monkeypatch, aggregation)
    monkeypatch.setenv("NEWS_SOURCE_RETRIEVAL_ENABLED", "false")
    source = SearchResult(
        title="Economic report",
        url="https://example.org/economy",
        snippet="GDP grew 8% in 2026.",
        publisher="Example",
        published_at="2026-10-15",
        retrieved_at=NOW,
    )
    monkeypatch.setattr(
        news_pipeline.news_search,
        "search_claims",
        lambda _claims: SearchResponse(status="complete", provider="mock", results=[source]),
    )

    response = news_pipeline.analyze_news("GDP grew 7% in 2026.")

    item = response.evidence[0]
    check = item.evidence_quality.numerical_check
    assert item.stance == EvidenceStance.SUPPORTS
    assert check.status.value == "mismatch"
    assert (check.claim_value, check.source_value) == (7, 8)
    assert response.evidence_summaries[0].support_weight == 0
    assert response.verdict == Verdict.INCONCLUSIVE


def test_one_source_does_not_transfer_support_between_claims(monkeypatch: pytest.MonkeyPatch):
    from backend.pipelines import news_pipeline

    source = SearchResult(
        title="Product Y launch report",
        url="https://www.reuters.com/technology/product-y",
        snippet="Acme launched Product Y in Bengaluru today.",
        publisher="Reuters",
        published_at="2026-09-24T08:00:00Z",
        retrieved_at=NOW,
        claim_ids=["claim-1"],
    )

    class AlwaysEntailNLI:
        def predict(self, premise, hypothesis):
            scores = {name: 0.05 for name in ("contradiction", "entailment", "neutral")}
            scores["entailment"] = 0.9
            probabilities = NLIProbabilities(**scores)
            return NLIEvidenceSignal(
                model_id="test/nli",
                model_version="test",
                prediction="entailment",
                status=SignalStatus.COMPLETE,
                probabilities=probabilities,
                selected_confidence=probabilities.entailment,
            )

    aggregation = RecordingAggregationService()
    nli = AlwaysEntailNLI()
    _configure_pipeline(monkeypatch, aggregation)
    monkeypatch.setenv("NEWS_SOURCE_RETRIEVAL_ENABLED", "false")
    monkeypatch.setattr(
        news_pipeline.news_search,
        "search_claims",
        lambda _claims: SearchResponse(status="complete", provider="mock", results=[source]),
    )
    monkeypatch.setattr(
        news_pipeline,
        "evidence_analysis",
        EvidenceAnalysisService(NLIEvidenceAnalyzer(nli)),
    )

    response = news_pipeline.analyze_news(
        "Acme launched Product Y in Bengaluru today. Acme launched Product Z in Mumbai today."
    )

    assert [(item.claim_id, item.source_url, item.stance) for item in response.evidence] == [
        ("claim-1", source.url, EvidenceStance.SUPPORTS),
    ]
    assert ("claim-2", source.url) not in {(item.claim_id, item.source_url) for item in response.evidence}
    assert response.evidence_summaries[0].support_weight > 0
    assert response.evidence_summaries[1].support_weight == 0
    assert response.verdict == Verdict.INCONCLUSIVE


def test_unrelated_claim_sources_are_not_aggregated_across_claims(monkeypatch: pytest.MonkeyPatch):
    from backend.pipelines import news_pipeline

    who_source = SearchResult(
        title="WHO emergency",
        url="https://who.example/pheic-2020",
        snippet="WHO declared a public health emergency in 2020.",
        publisher="WHO",
        published_at="2020-01-30",
        retrieved_at=NOW,
        claim_ids=["claim-1"],
    )
    spacex_source = SearchResult(
        title="Starship launch",
        url="https://spacex.example/starship-2024",
        snippet="SpaceX launched Starship from Texas in 2024.",
        publisher="SpaceX",
        published_at="2024-06-01",
        retrieved_at=NOW,
        claim_ids=["claim-2"],
    )

    class RecordingNLI:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def predict(self, premise, hypothesis):
            self.calls.append((premise, hypothesis))
            scores = {name: 0.05 for name in ("contradiction", "entailment", "neutral")}
            scores["entailment"] = 0.9
            probabilities = NLIProbabilities(**scores)
            return NLIEvidenceSignal(
                model_id="test/nli",
                model_version="test",
                prediction="entailment",
                status=SignalStatus.COMPLETE,
                probabilities=probabilities,
                selected_confidence=probabilities.entailment,
            )

    aggregation = RecordingAggregationService()
    nli = RecordingNLI()
    _configure_pipeline(monkeypatch, aggregation)
    monkeypatch.setenv("NEWS_SOURCE_RETRIEVAL_ENABLED", "false")
    monkeypatch.setattr(
        news_pipeline.news_search,
        "search_claims",
        lambda _claims: SearchResponse(
            status="complete", provider="mock", results=[who_source, spacex_source]
        ),
    )
    monkeypatch.setattr(
        news_pipeline,
        "evidence_analysis",
        EvidenceAnalysisService(NLIEvidenceAnalyzer(nli)),
    )

    response = news_pipeline.analyze_news(
        "WHO declared a public health emergency in 2020. SpaceX launched Starship from Texas in 2024."
    )

    assert nli.calls == [
        (who_source.snippet, "WHO declared a public health emergency in 2020."),
        (spacex_source.snippet, "SpaceX launched Starship from Texas in 2024."),
    ]
    assert [(item.claim_id, item.source_url, item.stance) for item in response.evidence] == [
        ("claim-1", who_source.url, EvidenceStance.SUPPORTS),
        ("claim-2", spacex_source.url, EvidenceStance.SUPPORTS),
    ]
    evidence_by_claim = {
        summary.claim_id: [
            item.source_url for item in response.evidence if item.claim_id == summary.claim_id
        ]
        for summary in response.evidence_summaries
    }
    assert evidence_by_claim == {
        "claim-1": [who_source.url],
        "claim-2": [spacex_source.url],
    }
    assert who_source.url not in evidence_by_claim["claim-2"]
    assert spacex_source.url not in evidence_by_claim["claim-1"]
    assert response.evidence_summaries[0].support_weight > 0
    assert response.evidence_summaries[1].support_weight > 0
