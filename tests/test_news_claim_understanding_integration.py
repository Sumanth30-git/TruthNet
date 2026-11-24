import sys
from types import SimpleNamespace

from backend.pipelines import news_pipeline
from backend.schemas import EntityType, SearchResponse, Verdict
from backend.services.claim_understanding import ClaimUnderstandingService


class FakeEntity:
    def __init__(self, text: str, label: str, source: str) -> None:
        self.label_ = label
        self.start_char = source.index(text)
        self.end_char = self.start_char + len(text)


class FakeNLP:
    meta = {"version": "3.8.0-test"}

    def __init__(self, entities):
        self.ents = entities

    def __call__(self, _text):
        return SimpleNamespace(ents=self.ents)


def test_opt_in_spacy_context_attaches_without_changing_news_claims_or_verdict(monkeypatch):
    text = "Acme launched Product Y in Bengaluru today."
    entities = [
        FakeEntity("Acme", "ORG", text),
        FakeEntity("Bengaluru", "GPE", text),
    ]
    load_calls = []
    monkeypatch.setitem(
        sys.modules,
        "spacy",
        SimpleNamespace(load=lambda model: (load_calls.append(model), FakeNLP(entities))[1]),
    )
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "false")
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "false")
    monkeypatch.setattr(
        news_pipeline.news_search,
        "search_claims",
        lambda _claims: SearchResponse(status="complete", provider="mock", results=[]),
    )
    service = ClaimUnderstandingService()
    monkeypatch.setattr(news_pipeline, "claim_understanding", service)

    deterministic_response = news_pipeline.analyze_news(text)
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "true")
    spacy_response = news_pipeline.analyze_news(text)

    assert deterministic_response.verdict == Verdict.INCONCLUSIVE
    assert spacy_response.verdict == deterministic_response.verdict
    assert [claim.text for claim in spacy_response.claims] == [claim.text for claim in deterministic_response.claims]
    assert spacy_response.claims[0].text == text
    assert [
        (entity.text, entity.entity_type)
        for entity in spacy_response.claims[0].context.entity_mentions
    ] == [
        ("Acme", EntityType.ORGANIZATION),
        ("Bengaluru", EntityType.LOCATION),
    ]
    assert spacy_response.claims[0].context.relations == []
    assert load_calls == ["en_core_web_sm"]


def test_news_extracts_atomic_claims_and_context_before_search_and_evidence(monkeypatch):
    text = (
        "NASA launched a mission in Florida in 2025 and "
        "SpaceX announced 3% growth in California in 2026."
    )
    events = []
    service = ClaimUnderstandingService()
    original_extract = service.extract_claims
    original_understand = service.understand_claims

    def extract(text_value):
        events.append("extract")
        return original_extract(text_value)

    def understand(claims):
        events.append("understand")
        return original_understand(claims)

    def search(query):
        assert query == text
        return SearchResponse(status="complete", provider="mock", results=[])

    def search_claims(claims):
        events.append("search")
        assert len(claims) == 2
        assert all(claim.context is not None for claim in claims)
        return search(text)

    def analyze(claims, sources):
        events.append("evidence")
        assert sources == []
        assert [claim.text for claim in claims] == [
            "NASA launched a mission in Florida in 2025.",
            "SpaceX announced 3% growth in California in 2026.",
        ]
        assert all(claim.context is not None for claim in claims)
        return [], news_pipeline.EvidenceAnalysisStatus.NO_SOURCES

    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "false")
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "false")
    monkeypatch.setattr(service, "extract_claims", extract)
    monkeypatch.setattr(service, "understand_claims", understand)
    monkeypatch.setattr(news_pipeline, "claim_understanding", service)
    monkeypatch.setattr(news_pipeline.news_search, "search_claims", search_claims)
    monkeypatch.setattr(news_pipeline.evidence_analysis, "analyze", analyze)

    response = news_pipeline.analyze_news(text)

    assert events == ["extract", "understand", "search", "evidence"]
    assert [claim.claim_id for claim in response.claims] == ["claim-1", "claim-2"]
    first, second = [claim.context for claim in response.claims]
    assert first.subject == "NASA"
    assert first.location == "Florida"
    assert first.time_reference == "2025"
    assert second.subject == "SpaceX"
    assert second.location == "California"
    assert second.time_reference == "2026"
    assert second.numerical_values[0].value == 3
    assert second.numerical_values[0].unit == "percent"


def test_claim_understanding_failure_fails_closed_without_breaking_news(monkeypatch):
    text = "NASA launched a mission."
    service = ClaimUnderstandingService()
    monkeypatch.setattr(service, "understand_claims", lambda _claims: (_ for _ in ()).throw(RuntimeError("unavailable")))
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "false")
    monkeypatch.setattr(news_pipeline, "claim_understanding", service)
    monkeypatch.setattr(
        news_pipeline.news_search,
        "search_claims",
        lambda _claims: SearchResponse(status="complete", provider="mock", results=[]),
    )
    evidence_claims = []
    monkeypatch.setattr(
        news_pipeline.evidence_analysis,
        "analyze",
        lambda claims, _sources: (evidence_claims.extend(claims) or [], news_pipeline.EvidenceAnalysisStatus.NO_SOURCES),
    )

    response = news_pipeline.analyze_news(text)

    assert response.verdict == Verdict.INCONCLUSIVE
    assert response.claims == []
    assert evidence_claims == []


def test_missing_spacy_model_does_not_escape_through_news_pipeline(monkeypatch):
    text = "Acme launched Product Y in Bengaluru today."

    def missing_model(_model):
        raise OSError("model is not installed")

    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=missing_model))
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "false")
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "true")
    monkeypatch.setattr(
        news_pipeline.news_search,
        "search_claims",
        lambda _claims: SearchResponse(status="complete", provider="mock", results=[]),
    )
    monkeypatch.setattr(news_pipeline, "claim_understanding", ClaimUnderstandingService())

    response = news_pipeline.analyze_news(text)

    context = response.claims[0].context
    assert response.verdict == Verdict.INCONCLUSIVE
    assert context is not None
    assert context.location == "Bengaluru"
    assert context.entity_mentions == []
    assert context.understanding_metadata.status == "unavailable"


def test_news_attaches_corrected_deterministic_context_for_possessive_claims(monkeypatch):
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "false")
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "false")
    monkeypatch.setattr(
        news_pipeline.news_search,
        "search_claims",
        lambda _claims: SearchResponse(status="complete", provider="mock", results=[]),
    )

    cases = [
        ("Moon's surface contains water ice.", "Moon's surface"),
        ("Moon\u2019s surface contains water ice.", "Moon\u2019s surface"),
        ("India's space programme has expanded significantly.", "India's space programme"),
        ("India\u2019s space programme has expanded significantly.", "India\u2019s space programme"),
    ]

    for text, expected_subject in cases:
        response = news_pipeline.analyze_news(text)

        assert len(response.claims) == 1
        claim = response.claims[0]
        assert claim.text == text
        assert claim.context is not None
        assert claim.context.factual_statement == text
        assert claim.context.subject == expected_subject
        assert claim.context.location is None
        assert claim.context.understanding_metadata.method == "deterministic_claim_understanding"
