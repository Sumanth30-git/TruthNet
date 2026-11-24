from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import (
    Claim,
    EvidenceContentOrigin,
    EvidenceAnalysisStatus,
    EvidenceRelevance,
    EvidenceStance,
    SearchResponse,
    SearchResult,
    SourceFetchStatus,
)
from backend.services.evidence_analysis import (
    ConservativeEvidenceAnalyzer,
    EvidenceAnalysisService,
    EvidenceAssessment,
    NLIEvidenceAnalyzer,
    extract_candidate_claims,
)
from backend.services.news_source_retrieval import SourceRetrievalResult, source_url_key

client = TestClient(app)


def _source(
    title: str,
    snippet: str | None = "source text",
    url: str | None = None,
    claim_ids: list[str] | None = None,
) -> SearchResult:
    return SearchResult(
        title=title,
        url=url or f"https://example.org/{title.lower().replace(' ', '-')}",
        snippet=snippet,
        retrieved_at=datetime.now(timezone.utc),
        claim_ids=list(claim_ids or []),
    )


def test_claims_are_conservatively_segmented_into_sentence_candidates():
    claims = extract_candidate_claims("First event happened. Second event followed!")
    assert [(claim.claim_id, claim.text, claim.extraction_method) for claim in claims] == [
        ("claim-1", "First event happened.", "sentence_segmentation"),
        ("claim-2", "Second event followed!", "sentence_segmentation"),
    ]
    assert extract_candidate_claims(None) == []


@pytest.mark.parametrize(
    ("claim_text", "stance", "relevance"),
    [
        ("supported", EvidenceStance.SUPPORTS, EvidenceRelevance.RELEVANT),
        ("contradicted", EvidenceStance.CONTRADICTS, EvidenceRelevance.RELEVANT),
        ("related", EvidenceStance.NEUTRAL, EvidenceRelevance.RELEVANT),
        ("unrelated", EvidenceStance.INSUFFICIENT, EvidenceRelevance.IRRELEVANT),
    ],
)
def test_analysis_adapter_output_preserves_stance_separate_from_relevance(claim_text, stance, relevance):
    class MockAnalyzer:
        def analyze(self, _claim, _content):
            return EvidenceAssessment(
                stance=stance,
                relevance=relevance,
                summary="Mocked analysis result.",
                reason="Deterministic analyzer-boundary test.",
            )

    claim = Claim(claim_id="claim-1", text=claim_text)
    items, status = EvidenceAnalysisService(MockAnalyzer()).analyze([claim], [_source("candidate")])
    assert status == EvidenceAnalysisStatus.COMPLETE
    assert items[0].stance == stance
    assert items[0].relevance == relevance


def test_default_analyzer_does_not_guess_stance_from_opposing_words():
    claim = Claim(claim_id="claim-1", text="The moon is made of cheese.")
    source = _source("denial", "The moon is not made of cheese.")
    items, status = EvidenceAnalysisService().analyze([claim], [source])
    assert status == EvidenceAnalysisStatus.INSUFFICIENT
    assert items[0].stance == EvidenceStance.INSUFFICIENT
    assert items[0].relevance == EvidenceRelevance.NOT_ASSESSED
    assert items[0].nli_signal is not None
    assert items[0].nli_signal.status.value == "not_run"


def test_missing_content_and_optional_metadata_are_preserved_safely():
    source = _source("No content", "   ")
    claim = Claim(claim_id="claim-1", text="A short claim.")
    items, status = EvidenceAnalysisService(ConservativeEvidenceAnalyzer()).analyze([claim], [source])
    item = items[0]
    assert status == EvidenceAnalysisStatus.INSUFFICIENT
    assert item.stance == EvidenceStance.INSUFFICIENT
    assert item.publisher is None and item.published_at is None
    assert item.source_title == "No content" and item.source_url == source.url
    assert item.retrieved_at == source.retrieved_at


def test_fetched_page_text_is_analyzed_and_source_provenance_is_preserved():
    calls = []

    class RecordingAnalyzer:
        def analyze(self, claim, content):
            calls.append((claim, content))
            return EvidenceAssessment(
                stance=EvidenceStance.NEUTRAL,
                relevance=EvidenceRelevance.NOT_ASSESSED,
                summary="Fetched source text was analyzed.",
                reason="Mocked page analysis.",
            )

    source = _source("Fetched page", "Original search excerpt.", "https://example.org/article")
    retrieved = SourceRetrievalResult(
        source_url=source.url,
        content="Visible article body from the fetched page.",
        status=SourceFetchStatus.FETCHED,
        final_url="https://example.org/canonical-article",
    )
    items, status = EvidenceAnalysisService(RecordingAnalyzer()).analyze(
        [Claim(claim_id="claim-1", text="A test claim.")],
        [source],
        retrieved_sources={source_url_key(source.url): retrieved},
    )

    item = items[0]
    assert status == EvidenceAnalysisStatus.COMPLETE
    assert calls == [("A test claim.", "Visible article body from the fetched page.")]
    assert item.source_url == source.url
    assert item.source_title == source.title
    assert item.source_snippet == "Visible article body from the fetched page."
    assert item.source_content_origin == EvidenceContentOrigin.FETCHED_PAGE
    assert item.source_fetch_status == SourceFetchStatus.FETCHED
    assert item.source_final_url == "https://example.org/canonical-article"
    assert item.publisher == source.publisher
    assert item.published_at == source.published_at
    assert item.retrieved_at == source.retrieved_at


def test_failed_page_fetch_falls_back_to_search_snippet():
    calls = []

    class RecordingAnalyzer:
        def analyze(self, claim, content):
            calls.append((claim, content))
            return EvidenceAssessment(
                stance=EvidenceStance.NEUTRAL,
                relevance=EvidenceRelevance.NOT_ASSESSED,
                summary="Snippet fallback analyzed.",
                reason="Mocked fallback.",
            )

    source = _source("Fallback", "Original search excerpt.", "https://example.org/fallback")
    retrieval = SourceRetrievalResult(source.url, None, SourceFetchStatus.TIMEOUT)
    items, status = EvidenceAnalysisService(RecordingAnalyzer()).analyze(
        [Claim(claim_id="claim-1", text="A test claim.")],
        [source],
        retrieved_sources={source_url_key(source.url): retrieval},
    )

    assert status == EvidenceAnalysisStatus.COMPLETE
    assert calls == [("A test claim.", "Original search excerpt.")]
    assert items[0].source_content_origin == EvidenceContentOrigin.SEARCH_SNIPPET
    assert items[0].source_fetch_status == SourceFetchStatus.TIMEOUT
    assert items[0].source_url == source.url


def test_no_page_text_or_snippet_keeps_existing_insufficient_behavior():
    source = _source("Empty", None, "https://example.org/empty")
    retrieval = SourceRetrievalResult(source.url, None, SourceFetchStatus.EMPTY_CONTENT)
    items, status = EvidenceAnalysisService(ConservativeEvidenceAnalyzer()).analyze(
        [Claim(claim_id="claim-1", text="A test claim.")],
        [source],
        retrieved_sources={source_url_key(source.url): retrieval},
    )

    assert status == EvidenceAnalysisStatus.INSUFFICIENT
    assert items[0].analysis_status == EvidenceAnalysisStatus.INSUFFICIENT
    assert items[0].source_content_origin == EvidenceContentOrigin.UNAVAILABLE
    assert items[0].source_fetch_status == SourceFetchStatus.EMPTY_CONTENT


def test_multiple_candidates_are_analyzed_independently():
    calls = []

    class RecordingAnalyzer:
        def analyze(self, claim, content):
            calls.append((claim, content))
            if content == "fail this one":
                raise RuntimeError("private diagnostic")
            return EvidenceAssessment(
                stance=EvidenceStance.NEUTRAL,
                relevance=EvidenceRelevance.RELEVANT,
                summary="Related context.",
                reason="Does not establish the claim.",
            )

    claims = [Claim(claim_id="claim-1", text="First."), Claim(claim_id="claim-2", text="Second.")]
    sources = [_source("bad", "fail this one"), _source("good", "related context")]
    items, status = EvidenceAnalysisService(RecordingAnalyzer()).analyze(claims, sources)
    assert [source.claim_ids for source in sources] == [[], []]
    assert len(items) == 4 and len(calls) == 4
    assert items[0].analysis_status == EvidenceAnalysisStatus.UNAVAILABLE
    assert items[1].stance == EvidenceStance.NEUTRAL
    assert status == EvidenceAnalysisStatus.PARTIAL
    assert "private diagnostic" not in str(items)


def test_origin_tagged_sources_are_analyzed_only_against_originating_claims():
    calls = []

    class RecordingAnalyzer:
        def analyze(self, claim, content):
            calls.append((claim, content))
            return EvidenceAssessment(
                stance=EvidenceStance.SUPPORTS,
                relevance=EvidenceRelevance.NOT_ASSESSED,
                summary="Origin-tagged analysis.",
                reason="Deterministic analyzer-boundary test.",
            )

    claims = [
        Claim(claim_id="claim-1", text="WHO declared a public health emergency in 2020."),
        Claim(claim_id="claim-2", text="SpaceX launched Starship from Texas in 2024."),
    ]
    who_source = _source(
        "WHO emergency",
        "WHO declared a public health emergency in 2020.",
        "https://who.example/pheic-2020",
        claim_ids=["claim-1"],
    )
    spacex_source = _source(
        "Starship launch",
        "SpaceX launched Starship from Texas in 2024.",
        "https://spacex.example/starship-2024",
        claim_ids=["claim-2"],
    )

    items, status = EvidenceAnalysisService(RecordingAnalyzer()).analyze(
        claims, [who_source, spacex_source]
    )

    assert status == EvidenceAnalysisStatus.COMPLETE
    assert calls == [
        (claims[0].text, who_source.snippet),
        (claims[1].text, spacex_source.snippet),
    ]
    assert [(item.claim_id, item.source_url, item.stance) for item in items] == [
        ("claim-1", who_source.url, EvidenceStance.SUPPORTS),
        ("claim-2", spacex_source.url, EvidenceStance.SUPPORTS),
    ]
    assert (who_source.url, "claim-2") not in {(item.source_url, item.claim_id) for item in items}
    assert (spacex_source.url, "claim-1") not in {(item.source_url, item.claim_id) for item in items}


def test_malformed_candidate_and_invalid_url_do_not_raise():
    claim = Claim(claim_id="claim-1", text="A claim.")
    malformed = {"title": "missing retrieval time", "url": "https://example.org"}
    invalid_url = _source("invalid", url="file:///private/path")
    items, status = EvidenceAnalysisService().analyze([claim], [malformed, invalid_url])
    assert len(items) == 2
    assert status == EvidenceAnalysisStatus.PARTIAL
    assert all(item.analysis_status == EvidenceAnalysisStatus.UNAVAILABLE for item in items)
    assert all("private" not in str(item) for item in items)


def test_news_api_exposes_evidence_but_keeps_inconclusive_verdict(monkeypatch: pytest.MonkeyPatch):
    from backend.pipelines import news_pipeline

    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "false")
    monkeypatch.setenv("NEWS_SOURCE_RETRIEVAL_ENABLED", "false")
    monkeypatch.setattr(
        news_pipeline.news_search,
        "search_claims",
        lambda _claims: SearchResponse(
            status="complete",
            provider="mock",
            results=[_source("Candidate", "Some retrieved source text.")],
        ),
    )
    response = client.post("/api/news/analyze", json={"text": "First claim. Second claim."})
    body = response.json()
    assert response.status_code == 200
    assert body["verdict"] == "inconclusive"
    assert [claim["claim_id"] for claim in body["claims"]] == ["claim-1", "claim-2"]
    assert len(body["evidence"]) == 2
    assert body["evidence"][0]["stance"] == "insufficient"
    assert body["evidence_analysis_status"] == "insufficient"
