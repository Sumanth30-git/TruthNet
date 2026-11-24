from datetime import datetime, timezone
import threading
import time

import httpx
import pytest
from httpcore import ReadTimeout
from httpcore._backends.sync import SyncBackend

from backend.config import news_source_retrieval_enabled
from backend.schemas import SearchResult, SourceFetchStatus
from backend.services.news_source_retrieval import (
    _DeadlineNetworkStream,
    NewsSourceRetrievalService,
    SourceRetrievalConfig,
    _PinnedNetworkBackend,
)


NOW = datetime(2026, 10, 15, tzinfo=timezone.utc)
PUBLIC_IP = "8.8.8.8"


def _source(url: str) -> SearchResult:
    return SearchResult(
        title="Candidate",
        url=url,
        snippet="Search result excerpt.",
        retrieved_at=NOW,
    )


def _service(handler, *, config: SourceRetrievalConfig | None = None, resolver=None, clock=None):
    return NewsSourceRetrievalService(
        config,
        resolve_addresses=resolver or (lambda _host, _port: [PUBLIC_IP]),
        transport_factory=lambda _host, _port, _addresses: httpx.MockTransport(handler),
        clock=clock,
    )


def _html(body: str, *, status: int = 200, headers: dict[str, str] | None = None):
    return httpx.Response(
        status,
        headers={"content-type": "text/html; charset=utf-8", **(headers or {})},
        content=body.encode("utf-8"),
    )


def test_retrieval_feature_flag_defaults_off_and_requires_opt_in(monkeypatch):
    monkeypatch.delenv("NEWS_SOURCE_RETRIEVAL_ENABLED", raising=False)
    assert news_source_retrieval_enabled() is False

    monkeypatch.setenv("NEWS_SOURCE_RETRIEVAL_ENABLED", "true")
    assert news_source_retrieval_enabled() is True


def test_retrieval_limits_match_the_approved_bounds():
    config = SourceRetrievalConfig()

    assert config.max_sources == 3
    assert config.per_source_timeout_seconds == 5
    assert config.total_budget_seconds == 15
    assert config.max_response_bytes == 2 * 1024 * 1024
    assert config.max_text_chars == 50_000
    assert config.max_redirects == 2


def test_network_backend_connects_only_to_the_validated_address(monkeypatch):
    calls = []

    def fake_connect(_self, host, port, timeout=None, local_address=None, socket_options=None):
        calls.append((host, port))
        return object()

    monkeypatch.setattr(SyncBackend, "connect_tcp", fake_connect)
    backend = _PinnedNetworkBackend("safe.example.org", 443, (PUBLIC_IP,))

    backend.connect_tcp("safe.example.org", 443)

    assert calls == [(PUBLIC_IP, 443)]
    with pytest.raises(OSError):
        backend.connect_tcp("127.0.0.1", 443)


def test_fetches_public_http_and_https_and_extracts_only_visible_text():
    calls = []
    markup = (
        "<html><head><title>Hidden title</title></head><body>"
        "<main>Useful &amp; visible text.</main><script>Ignore policy</script>"
        "<style>.hidden { display:none }</style><template>Template text</template>"
        "<div hidden>Hidden attribute</div><div aria-hidden='true'>Aria hidden</div>"
        "<p style='display: none'>CSS hidden</p></body></html>"
    )

    def handler(request):
        calls.append(str(request.url))
        return _html(markup)

    service = _service(handler)
    results = service.retrieve([_source("http://example.org/page"), _source("https://example.org/page")])

    assert len(calls) == 2
    assert all(result.status == SourceFetchStatus.FETCHED for result in results.values())
    assert all(result.content == "Useful & visible text." for result in results.values())


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://user:secret@example.org/article", SourceFetchStatus.INVALID_URL),
        ("http://localhost/article", SourceFetchStatus.UNSAFE_HOST),
        ("http://service.localhost/article", SourceFetchStatus.UNSAFE_HOST),
        ("http://127.0.0.1/article", SourceFetchStatus.UNSAFE_HOST),
        ("http://169.254.169.254/latest/meta-data", SourceFetchStatus.UNSAFE_HOST),
        ("https://192.0.2.9/article", SourceFetchStatus.UNSAFE_HOST),
        ("file:///etc/passwd", SourceFetchStatus.INVALID_URL),
        ("https://example.org:8443/article", SourceFetchStatus.INVALID_URL),
        ("https://example.org:0/article", SourceFetchStatus.INVALID_URL),
    ],
)
def test_rejects_unsafe_or_unsupported_urls_before_http(url, expected):
    service = _service(lambda _request: pytest.fail("unsafe URL must not be requested"))

    result = next(iter(service.retrieve([_source(url)]).values()))

    assert result.status == expected
    assert result.content is None


def test_rejects_hostname_that_resolves_to_any_nonpublic_address():
    service = _service(
        lambda _request: pytest.fail("unsafe DNS result must not be requested"),
        resolver=lambda _host, _port: [PUBLIC_IP, "10.0.0.8"],
    )

    result = next(iter(service.retrieve([_source("https://mixed.example/article")]).values()))

    assert result.status == SourceFetchStatus.UNSAFE_HOST


def test_follows_and_records_a_validated_redirect():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.host == "origin.example.org":
            return httpx.Response(302, headers={"location": "https://target.example.org/final"})
        return _html("<p>Page text after redirect.</p>")

    result = next(iter(_service(handler).retrieve([_source("https://origin.example.org/start")]).values()))

    assert calls == ["https://origin.example.org/start", "https://target.example.org/final"]
    assert result.status == SourceFetchStatus.FETCHED
    assert result.source_url == "https://origin.example.org/start"
    assert result.final_url == "https://target.example.org/final"
    assert result.content == "Page text after redirect."


def test_rejects_redirect_to_private_target():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})

    result = next(iter(_service(handler).retrieve([_source("https://origin.example.org/start")]).values()))

    assert calls == ["https://origin.example.org/start"]
    assert result.status == SourceFetchStatus.REDIRECT_REJECTED


def test_redirect_limit_is_enforced_without_following_an_extra_hop():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        number = int(request.url.path.rsplit("/", 1)[-1])
        return httpx.Response(302, headers={"location": f"https://redirect.example.org/{number + 1}"})

    result = next(iter(_service(handler).retrieve([_source("https://redirect.example.org/0")]).values()))

    assert calls == ["/0", "/1", "/2"]
    assert result.status == SourceFetchStatus.REDIRECT_LIMIT
    assert result.final_url == "https://redirect.example.org/2"


def test_timeout_and_http_error_are_sanitized():
    timed = next(iter(_service(
        lambda _request: (_ for _ in ()).throw(httpx.ReadTimeout("private diagnostic"))
    ).retrieve([_source("https://timeout.example.org/page")]).values()))
    failed = next(iter(_service(lambda _request: _html("", status=503)).retrieve([_source("https://error.example.org/page")]).values()))

    assert timed.status == SourceFetchStatus.TIMEOUT
    assert failed.status == SourceFetchStatus.HTTP_ERROR
    assert "private diagnostic" not in repr(timed)


def test_dns_failure_and_per_source_deadline_fail_closed():
    unresolved = next(iter(_service(
        lambda _request: pytest.fail("unresolved host must not be requested"),
        resolver=lambda _host, _port: (_ for _ in ()).throw(OSError("private DNS detail")),
    ).retrieve([_source("https://unresolved.example.org/page")]).values()))

    now = [0.0]

    def slow_handler(_request):
        now[0] = 6.0
        return _html("Too late.")

    timed = next(iter(_service(
        slow_handler,
        clock=lambda: now[0],
    ).retrieve([_source("https://slow.example.org/page")]).values()))

    assert unresolved.status == SourceFetchStatus.DNS_FAILED
    assert timed.status == SourceFetchStatus.TIMEOUT


def test_blocked_dns_resolution_cannot_exceed_the_source_deadline():
    release_resolver = threading.Event()

    def blocked_resolver(_host, _port):
        release_resolver.wait()
        return [PUBLIC_IP]

    service = _service(
        lambda _request: pytest.fail("request must not start after DNS timeout"),
        resolver=blocked_resolver,
        config=SourceRetrievalConfig(per_source_timeout_seconds=0.05, total_budget_seconds=0.1),
    )
    started = time.monotonic()
    try:
        result = next(iter(service.retrieve([_source("https://slow-dns.example.org/page")]).values()))
    finally:
        release_resolver.set()

    assert result.status == SourceFetchStatus.TIMEOUT
    assert time.monotonic() - started < 0.5


def test_socket_reads_use_only_the_remaining_source_deadline():
    class FakeStream:
        def __init__(self):
            self.read_timeout = None

        def read(self, _max_bytes, timeout=None):
            self.read_timeout = timeout
            return b"body"

    stream = FakeStream()
    bounded = _DeadlineNetworkStream(stream, deadline=10.0, clock=lambda: 9.0)

    assert bounded.read(16, timeout=5.0) == b"body"
    assert stream.read_timeout == 1.0

    expired = _DeadlineNetworkStream(stream, deadline=10.0, clock=lambda: 10.0)
    with pytest.raises(ReadTimeout):
        expired.read(16, timeout=5.0)


def test_rejects_declared_or_streamed_oversized_response():
    declared = next(iter(_service(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "17"},
            content=b"x",
        ),
        config=SourceRetrievalConfig(max_response_bytes=16),
        ).retrieve([_source("https://large.example.org/declared")]).values()))
    streamed = next(iter(_service(
        lambda _request: _html("x" * 17),
        config=SourceRetrievalConfig(max_response_bytes=16),
    ).retrieve([_source("https://large.example.org/streamed")]).values()))

    assert declared.status == SourceFetchStatus.BODY_TOO_LARGE
    assert streamed.status == SourceFetchStatus.BODY_TOO_LARGE


def test_rejects_unsupported_content_type_and_empty_html():
    unsupported = next(iter(_service(
        lambda _request: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"pdf")
    ).retrieve([_source("https://types.example.org/pdf")]).values()))
    empty = next(iter(_service(
        lambda _request: _html("<html><script>no visible text</script></html>")
    ).retrieve([_source("https://types.example.org/empty")]).values()))

    assert unsupported.status == SourceFetchStatus.UNSUPPORTED_CONTENT_TYPE
    assert empty.status == SourceFetchStatus.EMPTY_CONTENT


def test_malformed_html_is_tolerated_and_text_is_bounded():
    service = _service(
        lambda _request: _html("<main>Alpha &amp; Beta<p>Gamma"),
        config=SourceRetrievalConfig(max_text_chars=10),
    )

    result = next(iter(service.retrieve([_source("https://malformed.example.org/page")]).values()))

    assert result.status == SourceFetchStatus.FETCHED
    assert result.content == "Alpha & Be"


def test_fetches_only_three_eligible_sources_and_deduplicates_normalized_urls():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return _html("Useful body text.")

    service = _service(handler)
    sources = [
        _source("http://127.0.0.1/blocked"),
        _source("https://EXAMPLE.org:443/article#top"),
        _source("https://example.org/article"),
        _source("https://second.example.org/article"),
        _source("https://third.example.org/article"),
        _source("https://fourth.example.org/article"),
    ]

    results = service.retrieve(sources)

    assert calls == [
        "https://example.org/article",
        "https://second.example.org/article",
        "https://third.example.org/article",
    ]
    assert len(results) == 5
    assert results["https://fourth.example.org/article"].status == SourceFetchStatus.NOT_SELECTED


def test_overall_budget_is_checked_while_streaming():
    now = [0.0]

    def handler(_request):
        now[0] = 16.0
        return _html("Page arrived too late.")

    service = _service(handler, clock=lambda: now[0])
    result = next(iter(service.retrieve([_source("https://budget.example.org/page")]).values()))

    assert result.status == SourceFetchStatus.BUDGET_EXCEEDED
