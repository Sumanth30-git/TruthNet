from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import (
    Claim,
    EvidenceAnalysisStatus,
    EvidenceRelevance,
    EvidenceStance,
    SearchResponse,
    SearchResult,
)
from backend.services.evidence_analysis import (
    ConservativeEvidenceAnalyzer,
    EvidenceAnalysisService,
    EvidenceAssessment,
    NLIEvidenceAnalyzer,
    extract_candidate_claims,
)

client = TestClient(app)


def _source(title: str, snippet: str | None = "source text", url: str | None = None) -> SearchResult:
    return SearchResult(
        title=title,
        url=url or f"https://example.org/{title.lower().replace(' ', '-')}",
        snippet=snippet,
        retrieved_at=datetime.now(timezone.utc),
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
    assert len(items) == 4 and len(calls) == 4
    assert items[0].analysis_status == EvidenceAnalysisStatus.UNAVAILABLE
    assert items[1].stance == EvidenceStance.NEUTRAL
    assert status == EvidenceAnalysisStatus.PARTIAL
    assert "private diagnostic" not in str(items)


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
    monkeypatch.setattr(
        news_pipeline.news_search,
        "search",
        lambda _query: SearchResponse(
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
