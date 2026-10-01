from datetime import datetime, timezone

import pytest

from backend.schemas import (
    EvidenceAnalysisStatus,
    EvidenceItem,
    SearchResponse,
    SearchResult,
)


NOW = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)


def _evidence_items(claims, sources):
    return [
        EvidenceItem(
            kind="news_source_candidate",
            summary="Mocked candidate evidence.",
            source=source.publisher or source.url,
            claim_id=claim.claim_id,
            claim=claim.text,
            source_title=source.title,
            source_url=source.url,
            source_snippet=source.snippet,
            publisher=source.publisher,
            published_at=source.published_at,
            retrieved_at=source.retrieved_at,
        )
        for claim in claims
        for source in sources
    ]


def test_news_pipeline_matches_every_claim_source_pair_and_attaches_quality(monkeypatch: pytest.MonkeyPatch):
    from backend.pipelines import news_pipeline

    sources = [
        SearchResult(
            title="Acme launch in Bengaluru",
            url="https://www.reuters.com/world/acme-launch",
            snippet="Acme launched Product Y in Bengaluru today.",
            publisher="Reuters",
            published_at="2026-09-24T08:00:00Z",
            retrieved_at=NOW,
        ),
        SearchResult(
            title="Other report",
            url="https://example.org/other",
            snippet="A separate report with limited details.",
            publisher="Example",
            retrieved_at=NOW,
        ),
    ]
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "false")
    monkeypatch.setattr(
        news_pipeline.news_search,
        "search",
        lambda _query: SearchResponse(status="complete", provider="mock", results=sources),
    )
    monkeypatch.setattr(
        news_pipeline.evidence_analysis,
        "analyze",
        lambda claims, results: (_evidence_items(claims, results), EvidenceAnalysisStatus.INSUFFICIENT),
    )

    calls = []
    original_match = news_pipeline.evidence_matching.match

    def recording_match(context, source):
        calls.append((context.claim_id, source.url))
        return original_match(context, source)

    monkeypatch.setattr(news_pipeline.evidence_matching, "match", recording_match)

    response = news_pipeline.analyze_news(
        "Acme launched Product Y in Bengaluru today. A second claim is unverified."
    )

    assert calls == [
        ("claim-1", source.url)
        for source in sources
    ] + [
        ("claim-2", source.url)
        for source in sources
    ]
    assert len(response.evidence) == 4
    assert {item.claim_id for item in response.evidence} == {"claim-1", "claim-2"}

    matched = response.evidence[0]
    assert matched.evidence_quality is not None
    assert matched.evidence_quality.source_profile.publisher == "Reuters"
    assert matched.evidence_quality.match.entity_match.value == "match"
    assert matched.evidence_quality.match.relevance.value == "high"
    assert matched.evidence_quality.freshness.value == "current"
    assert response.claims[0].context is not None
    assert response.claims[0].context.claim_id == "claim-1"


def test_news_pipeline_preserves_not_assessed_matching_for_underspecified_claim(monkeypatch: pytest.MonkeyPatch):
    from backend.pipelines import news_pipeline

    source = SearchResult(
        title="Brief report",
        url="https://example.org/report",
        snippet="Retrieved text without matching details.",
        retrieved_at=NOW,
    )
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "false")
    monkeypatch.setattr(
        news_pipeline.news_search,
        "search",
        lambda _query: SearchResponse(status="complete", provider="mock", results=[source]),
    )
    monkeypatch.setattr(
        news_pipeline.evidence_analysis,
        "analyze",
        lambda claims, results: (_evidence_items(claims, results), EvidenceAnalysisStatus.INSUFFICIENT),
    )

    item = news_pipeline.analyze_news("Hello there.").evidence[0]

    assert item.claim_id == "claim-1"
    assert item.evidence_quality is not None
    assert item.evidence_quality.match.relevance.value == "not_assessed"
    assert item.evidence_quality.match.entity_match.value == "unknown"
    assert item.evidence_quality.match.event_match.value == "unknown"
    assert item.evidence_quality.match.location_match.value == "not_applicable"
    assert item.evidence_quality.freshness.value == "unknown"
