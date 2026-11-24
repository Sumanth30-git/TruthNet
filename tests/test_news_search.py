from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import httpx

from backend.schemas import Claim, ClaimContext, ClaimNumber, EntityMention, EntityType
from backend.services.news_search import SearchConfig, SearchService, TavilySearchProvider, build_claim_search_query


class FakeProvider:
    name = "fake"

    def __init__(self, response: Any = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[tuple[str, float, int]] = []

    def search(self, query: str, *, timeout_seconds: float, max_results: int) -> Sequence[Mapping[str, Any]]:
        self.calls.append((query, timeout_seconds, max_results))
        if self.error:
            raise self.error
        return self.response


def _service(provider: FakeProvider, maximum: int = 5) -> SearchService:
    return SearchService(SearchConfig(provider="fake", timeout_seconds=2.5, max_results=maximum), provider)


def test_search_parses_candidate_and_missing_optional_metadata():
    service = _service(FakeProvider([{"title": "Report", "url": "https://example.org/story"}]))
    response = service.search("sample claim")
    assert response.status == "complete"
    assert len(response.results) == 1
    assert response.results[0].title == "Report"
    assert response.results[0].publisher is None
    assert response.results[0].published_at is None
    assert response.results[0].retrieved_at.tzinfo is not None


def test_search_returns_multiple_candidates_and_respects_maximum():
    provider = FakeProvider([{"title": str(i), "url": f"https://example.org/{i}"} for i in range(3)])
    response = _service(provider, maximum=2).search("claim")
    assert len(response.results) == 2
    assert provider.calls == [("claim", 2.5, 2)]


def test_empty_search_results_are_complete_empty_results():
    response = _service(FakeProvider([])).search("claim")
    assert response.status == "complete" and response.results == []


def test_timeout_is_returned_as_failed_state():
    response = _service(FakeProvider(error=TimeoutError())).search("claim")
    assert response.status == "failed" and response.results == []
    assert "timed out" in response.message


def test_network_failure_is_returned_as_unavailable_state():
    response = _service(FakeProvider(error=ConnectionError("secret detail"))).search("claim")
    assert response.status == "unavailable" and "secret detail" not in str(response)


def test_malformed_provider_response_is_safely_failed():
    response = _service(FakeProvider({"unexpected": "object"})).search("claim")
    assert response.status == "failed" and response.results == []


def test_invalid_urls_and_malformed_rows_are_skipped():
    provider = FakeProvider([
        {"title": "bad", "url": "javascript:alert(1)"},
        {"title": "good", "url": "https://example.org"},
        None,
    ])
    response = _service(provider).search("claim")
    assert [item.title for item in response.results] == ["good"]


def test_unconfigured_search_does_not_call_provider():
    provider = FakeProvider([])
    response = SearchService(SearchConfig(), provider).search("claim")
    assert response.status == "not_configured"
    assert provider.calls == []


def test_search_configuration_uses_environment_and_safe_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEWS_SEARCH_PROVIDER", "fake")
    monkeypatch.setenv("NEWS_SEARCH_TIMEOUT_SECONDS", "3.25")
    monkeypatch.setenv("NEWS_SEARCH_MAX_RESULTS", "8")
    config = SearchConfig.from_env()
    assert (config.provider, config.timeout_seconds, config.max_results) == ("fake", 3.25, 8)
    monkeypatch.setenv("NEWS_SEARCH_TIMEOUT_SECONDS", "999")
    monkeypatch.setenv("NEWS_SEARCH_MAX_RESULTS", "0")
    config = SearchConfig.from_env()
    assert config.timeout_seconds == 5 and config.max_results == 5


class FakeHTTPResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        return self.payload


def test_tavily_maps_one_and_multiple_results_and_request_parameters(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-secret")
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeHTTPResponse(payload={"results": [
            {"title": "Article A", "url": "https://news.example/a", "content": "Summary A", "published_date": "2026-09-20"},
            {"title": "Article B", "url": "https://other.example/b", "content": "Summary B"},
        ]})

    service = SearchService(SearchConfig(provider="tavily", timeout_seconds=4, max_results=3), TavilySearchProvider(post=post))
    response = service.search("current query")
    assert response.status == "complete" and len(response.results) == 2
    first, second = response.results
    assert (first.title, first.url, first.snippet, first.publisher, first.published_at) == (
        "Article A", "https://news.example/a", "Summary A", "news.example", "2026-09-20"
    )
    assert second.publisher == "other.example" and second.published_at is None
    url, kwargs = calls[0]
    assert url == TavilySearchProvider.endpoint
    assert kwargs["headers"]["Authorization"] == "Bearer test-secret"
    assert kwargs["json"]["query"] == "current query"
    assert kwargs["json"]["max_results"] == 3 and kwargs["timeout"] == 4


def test_tavily_empty_results_are_complete(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-secret")
    provider = TavilySearchProvider(post=lambda *_args, **_kwargs: FakeHTTPResponse(payload={"results": []}))
    response = SearchService(SearchConfig(provider="tavily"), provider).search("query")
    assert response.status == "complete" and response.results == []


def test_tavily_missing_api_key_fails_safely(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider = TavilySearchProvider(post=lambda *_args, **_kwargs: pytest.fail("request must not be sent"))
    response = SearchService(SearchConfig(provider="tavily"), provider).search("query")
    assert response.status == "unavailable"
    assert "key" not in str(response).lower()


def test_tavily_timeout_and_http_failures_are_sanitized(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-secret")
    timed = TavilySearchProvider(post=lambda *_args, **_kwargs: (_ for _ in ()).throw(httpx.ReadTimeout("test-secret")))
    timed_result = SearchService(SearchConfig(provider="tavily"), timed).search("query")
    assert timed_result.status == "failed" and "test-secret" not in str(timed_result)

    limited = TavilySearchProvider(post=lambda *_args, **_kwargs: FakeHTTPResponse(status_code=429))
    limited_result = SearchService(SearchConfig(provider="tavily"), limited).search("query")
    assert limited_result.status == "unavailable" and "rate limit" in limited_result.message

    failed = TavilySearchProvider(post=lambda *_args, **_kwargs: FakeHTTPResponse(status_code=500))
    failed_result = SearchService(SearchConfig(provider="tavily"), failed).search("query")
    assert failed_result.status == "unavailable"


def test_tavily_malformed_payload_and_invalid_url_are_handled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-secret")
    malformed = TavilySearchProvider(post=lambda *_args, **_kwargs: FakeHTTPResponse(payload={"unexpected": []}))
    malformed_result = SearchService(SearchConfig(provider="tavily"), malformed).search("query")
    assert malformed_result.status == "failed" and "test-secret" not in str(malformed_result)

    provider = TavilySearchProvider(post=lambda *_args, **_kwargs: FakeHTTPResponse(payload={"results": [
        {"title": "Bad URL", "url": "file:///private/key"},
        {"title": "Good URL", "url": "https://example.com"},
    ]}))
    response = SearchService(SearchConfig(provider="tavily"), provider).search("query")
    assert [item.title for item in response.results] == ["Good URL"]


def test_tavily_selected_from_environment_without_key_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEWS_SEARCH_PROVIDER", "tavily")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    response = SearchService(SearchConfig.from_env()).search("query")
    assert response.provider == "tavily" and response.status == "unavailable"


def test_claim_query_preserves_structured_subject_entity_location_time_and_numbers():
    claim = Claim(
        claim_id="claim-1",
        text="NASA expects 7.1% growth in Florida in 2025.",
        context=ClaimContext(
            claim_id="claim-1",
            factual_statement="NASA expects 7.1% growth in Florida in 2025.",
            subject="NASA",
            event="growth",
            location="Florida",
            time_reference="2025",
            numerical_values=[ClaimNumber(raw_text="7.1%", value=7.1, unit="percent")],
            entity_mentions=[EntityMention(text="NASA", entity_type=EntityType.ORGANIZATION, start=0, end=4)],
        ),
    )

    query = build_claim_search_query(claim)

    assert query is not None
    assert all(term in query for term in ("NASA", "growth", "Florida", "2025", "7.1%", "expects"))


class RoutingProvider:
    name = "fake"

    def __init__(self, routes: dict[str, list[dict[str, str]]]):
        self.routes = routes
        self.calls: list[tuple[str, float, int]] = []

    def search(self, query: str, *, timeout_seconds: float, max_results: int) -> Sequence[Mapping[str, Any]]:
        self.calls.append((query, timeout_seconds, max_results))
        for marker, rows in self.routes.items():
            if marker in query:
                return rows
        return []


def test_search_claims_queries_multiple_claims_and_preserves_result_provenance():
    provider = FakeProvider([{
        "title": "Report",
        "url": "https://example.org/story",
        "snippet": "Source excerpt",
        "publisher": "Example News",
        "published_at": "2026-10-01",
    }])
    service = _service(provider, maximum=5)
    claims = [
        Claim(claim_id="claim-1", text="NASA launched the mission in Florida in 2025.", context=ClaimContext(
            claim_id="claim-1", factual_statement="NASA launched the mission in Florida in 2025.",
            subject="NASA", location="Florida", time_reference="2025",
        )),
        Claim(claim_id="claim-2", text="SpaceX announced a new flight from California in 2026.", context=ClaimContext(
            claim_id="claim-2", factual_statement="SpaceX announced a new flight from California in 2026.",
            subject="SpaceX", location="California", time_reference="2026",
        )),
    ]

    response = service.search_claims(claims)

    assert response.status == "complete"
    assert len(provider.calls) == 2
    assert "NASA" in provider.calls[0][0] and "Florida" in provider.calls[0][0] and "2025" in provider.calls[0][0]
    assert "SpaceX" in provider.calls[1][0] and "California" in provider.calls[1][0] and "2026" in provider.calls[1][0]
    assert len(response.results) == 1
    result = response.results[0]
    assert (result.title, result.url, result.snippet, result.publisher, result.published_at) == (
        "Report", "https://example.org/story", "Source excerpt", "Example News", "2026-10-01"
    )
    assert result.claim_ids == ["claim-1", "claim-2"]


def test_search_claims_retains_originating_claim_id_for_distinct_sources():
    provider = RoutingProvider({
        "NASA": [{
            "title": "NASA report",
            "url": "https://nasa.example/florida-2025",
            "snippet": "NASA launched the mission in Florida in 2025.",
            "publisher": "NASA News",
        }],
        "SpaceX": [{
            "title": "SpaceX report",
            "url": "https://spacex.example/california-2026",
            "snippet": "SpaceX announced a new flight from California in 2026.",
            "publisher": "SpaceX News",
        }],
    })
    claims = [
        Claim(claim_id="claim-1", text="NASA launched the mission in Florida in 2025.", context=ClaimContext(
            claim_id="claim-1", factual_statement="NASA launched the mission in Florida in 2025.",
            subject="NASA", location="Florida", time_reference="2025",
        )),
        Claim(claim_id="claim-2", text="SpaceX announced a new flight from California in 2026.", context=ClaimContext(
            claim_id="claim-2", factual_statement="SpaceX announced a new flight from California in 2026.",
            subject="SpaceX", location="California", time_reference="2026",
        )),
    ]

    response = _service(provider).search_claims(claims)

    assert response.status == "complete"
    assert len(provider.calls) == 2
    assert [(result.url, result.claim_ids) for result in response.results] == [
        ("https://nasa.example/florida-2025", ["claim-1"]),
        ("https://spacex.example/california-2026", ["claim-2"]),
    ]
    by_claim = {claim_id: result.url for result in response.results for claim_id in result.claim_ids}
    assert by_claim == {
        "claim-1": "https://nasa.example/florida-2025",
        "claim-2": "https://spacex.example/california-2026",
    }


def test_search_claims_handles_no_usable_claims_without_provider_call():
    provider = FakeProvider([])
    service = _service(provider)

    response = service.search_claims([
        {"claim_id": "missing-text"},
        Claim(claim_id="empty", text="  "),
    ])

    assert response.status == "not_run"
    assert "No usable claims" in response.message
    assert provider.calls == []
    assert service.search_claims(None).status == "not_run"


def test_search_claims_preserves_disabled_and_provider_failure_states(caplog):
    disabled_provider = FakeProvider([])
    disabled = SearchService(SearchConfig(), disabled_provider).search_claims([
        Claim(claim_id="claim-1", text="NASA launched a mission.")
    ])
    assert disabled.status == "not_configured"
    assert disabled_provider.calls == []

    failing_provider = FakeProvider(error=ConnectionError("provider-secret"))
    failed = _service(failing_provider).search_claims([
        Claim(claim_id="claim-1", text="NASA launched a mission.")
    ])
    assert failed.status == "unavailable"
    assert "provider-secret" not in str(failed)
    assert "provider-secret" not in caplog.text


def test_search_claims_bounds_query_count_and_total_results():
    provider = FakeProvider([{"title": str(index), "url": f"https://example.org/{index}"} for index in range(4)])
    service = _service(provider, maximum=2)
    claims = [Claim(claim_id=f"claim-{index}", text=f"Organization {index} announced a project.") for index in range(4)]

    response = service.search_claims(claims)

    assert response.status == "partial"
    assert len(provider.calls) == 2
    assert len(response.results) == 2
    assert all(call[2] == 2 for call in provider.calls)
