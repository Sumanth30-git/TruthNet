"""Bounded, fail-closed retrieval of public News source pages."""

from __future__ import annotations

import codecs
import ipaddress
import re
import socket
import threading
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from httpcore import ConnectTimeout, ReadTimeout, WriteTimeout
from httpcore._backends.sync import SyncBackend

from backend.config import (
    NEWS_SOURCE_RETRIEVAL_BUDGET_SECONDS,
    NEWS_SOURCE_RETRIEVAL_MAX_REDIRECTS,
    NEWS_SOURCE_RETRIEVAL_MAX_RESPONSE_BYTES,
    NEWS_SOURCE_RETRIEVAL_MAX_SOURCES,
    NEWS_SOURCE_RETRIEVAL_MAX_TEXT_CHARS,
    NEWS_SOURCE_RETRIEVAL_TIMEOUT_SECONDS,
)
from backend.schemas import SearchResult, SourceFetchStatus


@dataclass(frozen=True)
class SourceRetrievalConfig:
    max_sources: int = NEWS_SOURCE_RETRIEVAL_MAX_SOURCES
    per_source_timeout_seconds: float = NEWS_SOURCE_RETRIEVAL_TIMEOUT_SECONDS
    total_budget_seconds: float = NEWS_SOURCE_RETRIEVAL_BUDGET_SECONDS
    max_response_bytes: int = NEWS_SOURCE_RETRIEVAL_MAX_RESPONSE_BYTES
    max_text_chars: int = NEWS_SOURCE_RETRIEVAL_MAX_TEXT_CHARS
    max_redirects: int = NEWS_SOURCE_RETRIEVAL_MAX_REDIRECTS


@dataclass(frozen=True)
class SourceRetrievalResult:
    source_url: str
    content: str | None
    status: SourceFetchStatus
    final_url: str | None = None


@dataclass(frozen=True)
class _ValidatedURL:
    source_url: str
    url: str
    normalized_url: str
    host: str
    port: int
    addresses: tuple[str, ...]


class _VisibleHTMLText(HTMLParser):
    """Extract visible text without executing or interpreting page markup."""

    _IGNORED_TAGS = frozenset({"head", "script", "style", "template", "noscript", "svg", "canvas"})
    _BLOCK_TAGS = frozenset({
        "address", "article", "blockquote", "br", "dd", "div", "dl", "dt", "footer",
        "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "ol",
        "p", "section", "table", "td", "th", "tr", "ul",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._suppressed: list[str] = []

    @staticmethod
    def _is_hidden(attrs: list[tuple[str, str | None]]) -> bool:
        values = {name.casefold(): (value or "").strip().casefold() for name, value in attrs}
        if "hidden" in values or values.get("aria-hidden") == "true":
            return True
        style = re.sub(r"\s+", "", values.get("style", ""))
        return any(marker in style for marker in ("display:none", "visibility:hidden", "content-visibility:hidden"))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if self._suppressed:
            return
        if tag in self._IGNORED_TAGS or self._is_hidden(attrs):
            self._suppressed.append(tag)
        elif tag in self._BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if self._suppressed:
            if tag in self._suppressed:
                del self._suppressed[self._suppressed.index(tag):]
            return
        if tag in self._BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._suppressed:
            self.parts.append(data)

    def visible_text(self) -> str:
        return unicodedata.normalize("NFC", " ".join("".join(self.parts).split()))


def _source_url_key(value: str | None) -> str | None:
    """Normalize only URL components that cannot identify different content."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
        scheme = parsed.scheme.casefold()
        host = parsed.hostname
        if scheme not in {"http", "https"} or not host or parsed.username is not None or parsed.password is not None:
            return value.strip().casefold()
        host = host.rstrip(".").encode("idna").decode("ascii").casefold()
        port = parsed.port
        default_port = 80 if scheme == "http" else 443
        if port == default_port:
            port = None
        rendered_host = f"[{host}]" if ":" in host else host
        netloc = rendered_host if port is None else f"{rendered_host}:{port}"
        return urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))
    except (UnicodeError, ValueError):
        return value.strip().casefold()


def source_url_key(value: str | None) -> str | None:
    """Expose the stable key used to join retrieval results to evidence sources."""
    return _source_url_key(value)


def _is_public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return bool(
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_reserved
        and not address.is_unspecified
    )


def _system_resolve(host: str, port: int) -> Sequence[str]:
    return tuple(
        item[4][0]
        for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    )


class _PinnedNetworkBackend(SyncBackend):
    """Connect to the exact public address checked for this URL and retain TLS SNI."""

    def __init__(
        self,
        host: str,
        port: int,
        addresses: tuple[str, ...],
        deadline: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._expected_host = host.casefold()
        self._expected_port = port
        self._addresses = addresses
        self._deadline = deadline
        self._clock = clock

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        if str(host).casefold() != self._expected_host or port != self._expected_port:
            raise OSError("Connection target did not match the validated public host.")
        # httpcore performs TLS using the original URL hostname after this
        # socket is returned, so certificate and SNI checks remain. Try only
        # one address; retrieval adds no retry behavior.
        stream = super().connect_tcp(self._addresses[0], port, timeout, local_address, socket_options)
        if self._deadline is None:
            return stream
        return _DeadlineNetworkStream(stream, self._deadline, self._clock)


class _DeadlineNetworkStream:
    """Cap each socket operation by the source's remaining wall-clock budget."""

    def __init__(self, stream, deadline: float, clock: Callable[[], float]) -> None:
        self._stream = stream
        self._deadline = deadline
        self._clock = clock

    def _remaining(self, timeout: float | None, timeout_error):
        remaining = self._deadline - self._clock()
        if remaining <= 0:
            raise timeout_error("Source retrieval deadline exceeded.")
        return remaining if timeout is None else min(timeout, remaining)

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._stream.read(max_bytes, self._remaining(timeout, ReadTimeout))

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._stream.write(buffer, self._remaining(timeout, WriteTimeout))

    def start_tls(self, ssl_context, server_hostname=None, timeout: float | None = None):
        stream = self._stream.start_tls(
            ssl_context,
            server_hostname=server_hostname,
            timeout=self._remaining(timeout, ConnectTimeout),
        )
        return _DeadlineNetworkStream(stream, self._deadline, self._clock)

    def close(self) -> None:
        self._stream.close()

    def get_extra_info(self, info: str):
        return self._stream.get_extra_info(info)


class _PinnedHTTPTransport(httpx.HTTPTransport):
    """httpx transport whose TCP connection uses the already-validated DNS answer."""

    def __init__(
        self,
        host: str,
        port: int,
        addresses: tuple[str, ...],
        deadline: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(trust_env=False, retries=0, limits=httpx.Limits(max_connections=1, max_keepalive_connections=0))
        # HTTPTransport does not expose httpcore's network backend in its public
        # constructor. The project pins httpx, and this narrowly scoped transport
        # prevents a second DNS lookup from bypassing public-address validation.
        self._pool._network_backend = _PinnedNetworkBackend(host, port, addresses, deadline, clock)


TransportFactory = Callable[[str, int, tuple[str, ...]], httpx.BaseTransport]
AddressResolver = Callable[[str, int], Sequence[str]]


class NewsSourceRetrievalService:
    """Fetch bounded visible page text from a small number of public URLs."""

    _REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
    _SUPPORTED_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain"})
    _LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".test", ".invalid", ".example")

    def __init__(
        self,
        config: SourceRetrievalConfig | None = None,
        *,
        resolve_addresses: AddressResolver | None = None,
        transport_factory: TransportFactory | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or SourceRetrievalConfig()
        self._resolve_addresses = resolve_addresses or _system_resolve
        self._transport_factory = transport_factory
        self._clock = clock or time.monotonic
        # getaddrinfo has no portable timeout. Keep at most one resolver worker
        # outstanding for this service; a stalled system resolver therefore
        # cannot exceed the retrieval deadline or create an unbounded thread pile.
        self._resolver_lock = threading.Lock()

    def _resolve_with_deadline(
        self,
        host: str,
        port: int,
        deadline: float,
    ) -> tuple[Sequence[str] | None, SourceFetchStatus | None]:
        remaining = deadline - self._clock()
        if remaining <= 0 or not self._resolver_lock.acquire(blocking=False):
            return None, SourceFetchStatus.TIMEOUT

        completed = threading.Event()
        outcome: dict[str, object] = {}

        def resolve() -> None:
            try:
                outcome["addresses"] = tuple(self._resolve_addresses(host, port))
            except Exception as exc:
                outcome["error"] = exc
            finally:
                completed.set()
                self._resolver_lock.release()

        try:
            threading.Thread(target=resolve, name="truthnet-source-dns", daemon=True).start()
        except Exception:
            self._resolver_lock.release()
            return None, SourceFetchStatus.DNS_FAILED
        if not completed.wait(max(0.0, deadline - self._clock())):
            return None, SourceFetchStatus.TIMEOUT
        if "error" in outcome:
            return None, SourceFetchStatus.DNS_FAILED
        addresses = outcome.get("addresses")
        if not isinstance(addresses, tuple):
            return None, SourceFetchStatus.DNS_FAILED
        return addresses, None

    @staticmethod
    def failed_results(sources: Sequence[SearchResult], status: SourceFetchStatus = SourceFetchStatus.FAILED):
        results: dict[str, SourceRetrievalResult] = {}
        for source in sources:
            key = _source_url_key(getattr(source, "url", None))
            if key is not None:
                results.setdefault(key, SourceRetrievalResult(source.url, None, status))
        return results

    def retrieve(self, sources: Sequence[SearchResult]) -> dict[str, SourceRetrievalResult]:
        started = self._clock()
        overall_deadline = started + self.config.total_budget_seconds
        results: dict[str, SourceRetrievalResult] = {}
        fetched_count = 0

        for source in sources:
            raw_url = getattr(source, "url", None)
            key = _source_url_key(raw_url)
            if key is None:
                continue
            if key in results:
                continue
            if self._clock() >= overall_deadline:
                results[key] = SourceRetrievalResult(raw_url, None, SourceFetchStatus.BUDGET_EXCEEDED)
                continue
            if fetched_count >= self.config.max_sources:
                results[key] = SourceRetrievalResult(raw_url, None, SourceFetchStatus.NOT_SELECTED)
                continue

            source_deadline = min(
                overall_deadline,
                self._clock() + self.config.per_source_timeout_seconds,
            )
            validated, error_status = self._validate_url(raw_url, source_deadline)
            if validated is None:
                results[key] = SourceRetrievalResult(raw_url, None, error_status)
                continue
            if self._clock() >= overall_deadline:
                results[key] = SourceRetrievalResult(raw_url, None, SourceFetchStatus.BUDGET_EXCEEDED)
                continue

            fetched_count += 1
            try:
                results[key] = self._fetch_validated(validated, source_deadline, overall_deadline)
            except Exception:
                # Never expose request, DNS, or parser diagnostics in API output.
                results[key] = SourceRetrievalResult(raw_url, None, SourceFetchStatus.FAILED)

        return results

    def _validate_url(
        self,
        value: object,
        deadline: float,
    ) -> tuple[_ValidatedURL | None, SourceFetchStatus]:
        if not isinstance(value, str) or not value.strip():
            return None, SourceFetchStatus.INVALID_URL
        try:
            parsed = urlsplit(value.strip())
            scheme = parsed.scheme.casefold()
            host = parsed.hostname
            if scheme not in {"http", "https"} or not host:
                return None, SourceFetchStatus.INVALID_URL
            if parsed.username is not None or parsed.password is not None:
                return None, SourceFetchStatus.INVALID_URL
            host = host.rstrip(".").encode("idna").decode("ascii").casefold()
            if not host or "%" in host or host == "localhost" or host.endswith(self._LOCAL_SUFFIXES):
                return None, SourceFetchStatus.UNSAFE_HOST
            try:
                specified_port = parsed.port
            except ValueError:
                return None, SourceFetchStatus.INVALID_URL
            port = specified_port if specified_port is not None else (80 if scheme == "http" else 443)
            expected_port = 80 if scheme == "http" else 443
            if port != expected_port:
                return None, SourceFetchStatus.INVALID_URL
            try:
                literal = ipaddress.ip_address(host)
            except ValueError:
                literal = None
            if literal is not None:
                addresses = (str(literal),)
            else:
                if self._clock() >= deadline:
                    return None, SourceFetchStatus.TIMEOUT
                resolved, resolve_status = self._resolve_with_deadline(host, port, deadline)
                if resolve_status is not None:
                    return None, resolve_status
                addresses = tuple(dict.fromkeys(resolved or ()))
            if not addresses:
                return None, SourceFetchStatus.DNS_FAILED
            if any(not _is_public_address(address) for address in addresses):
                return None, SourceFetchStatus.UNSAFE_HOST
            if self._clock() >= deadline:
                return None, SourceFetchStatus.TIMEOUT

            default_port = 80 if scheme == "http" else 443
            rendered_host = f"[{host}]" if ":" in host else host
            netloc = rendered_host if port == default_port else f"{rendered_host}:{port}"
            normalized = urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))
            return _ValidatedURL(value.strip(), normalized, normalized, host, port, addresses), SourceFetchStatus.FETCHED
        except (UnicodeError, ValueError):
            return None, SourceFetchStatus.INVALID_URL

    def _fetch_validated(
        self,
        original: _ValidatedURL,
        source_deadline: float,
        overall_deadline: float,
    ) -> SourceRetrievalResult:
        current_url = original.url
        current = original
        redirected = False
        visited = {original.normalized_url}

        def result(status: SourceFetchStatus, content: str | None = None) -> SourceRetrievalResult:
            final_url = current_url if redirected and current.normalized_url != original.normalized_url else None
            return SourceRetrievalResult(original.source_url, content, status, final_url)

        for redirect_count in range(self.config.max_redirects + 1):
            remaining = min(source_deadline, overall_deadline) - self._clock()
            if remaining <= 0:
                status = SourceFetchStatus.BUDGET_EXCEEDED if self._clock() >= overall_deadline else SourceFetchStatus.TIMEOUT
                return result(status)
            if self._transport_factory is None:
                transport = _PinnedHTTPTransport(
                    current.host,
                    current.port,
                    current.addresses,
                    min(source_deadline, overall_deadline),
                    self._clock,
                )
            else:
                transport = self._transport_factory(current.host, current.port, current.addresses)
            timeout = httpx.Timeout(max(0.01, min(self.config.per_source_timeout_seconds, remaining)))
            try:
                with httpx.Client(
                    transport=transport,
                    timeout=timeout,
                    follow_redirects=False,
                    trust_env=False,
                ) as client:
                    with client.stream(
                        "GET",
                        current_url,
                        headers={
                            "Accept": "text/html, application/xhtml+xml, text/plain;q=0.8",
                            "Accept-Encoding": "identity",
                            "User-Agent": "TruthNet-NewsSourceRetrieval/1.0",
                        },
                    ) as response:
                        if response.status_code in self._REDIRECT_CODES:
                            location = response.headers.get("location")
                            if not location:
                                return result(SourceFetchStatus.REDIRECT_REJECTED)
                            if redirect_count >= self.config.max_redirects:
                                return result(SourceFetchStatus.REDIRECT_LIMIT)
                            redirect_url = urljoin(current_url, location)
                            redirect_key = _source_url_key(redirect_url)
                            if redirect_key is None or redirect_key in visited:
                                return result(SourceFetchStatus.REDIRECT_REJECTED)
                            next_url, error_status = self._validate_url(redirect_url, min(source_deadline, overall_deadline))
                            if next_url is None:
                                return result(SourceFetchStatus.REDIRECT_REJECTED)
                            visited.add(next_url.normalized_url)
                            current_url = next_url.url
                            current = next_url
                            redirected = True
                            continue

                        if response.status_code != 200:
                            return result(SourceFetchStatus.HTTP_ERROR)
                        media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
                        if media_type not in self._SUPPORTED_CONTENT_TYPES:
                            return result(SourceFetchStatus.UNSUPPORTED_CONTENT_TYPE)
                        content_length = response.headers.get("content-length")
                        if content_length is not None:
                            try:
                                if int(content_length) < 0:
                                    return result(SourceFetchStatus.HTTP_ERROR)
                                if int(content_length) > self.config.max_response_bytes:
                                    return result(SourceFetchStatus.BODY_TOO_LARGE)
                            except ValueError:
                                return result(SourceFetchStatus.HTTP_ERROR)

                        body = bytearray()
                        for chunk in response.iter_bytes(chunk_size=16 * 1024):
                            if self._clock() >= overall_deadline:
                                return result(SourceFetchStatus.BUDGET_EXCEEDED)
                            if self._clock() >= source_deadline:
                                return result(SourceFetchStatus.TIMEOUT)
                            body.extend(chunk)
                            if len(body) > self.config.max_response_bytes:
                                return result(SourceFetchStatus.BODY_TOO_LARGE)
                        if self._clock() >= overall_deadline:
                            return result(SourceFetchStatus.BUDGET_EXCEEDED)
                        if self._clock() >= source_deadline:
                            return result(SourceFetchStatus.TIMEOUT)
                        try:
                            text = self._extract_text(bytes(body), media_type, response.headers.get("content-type", ""))
                        except Exception:
                            return result(SourceFetchStatus.EXTRACTION_FAILED)
                        if not text:
                            return result(SourceFetchStatus.EMPTY_CONTENT)
                        if len(text) > self.config.max_text_chars:
                            text = text[: self.config.max_text_chars]
                        return result(SourceFetchStatus.FETCHED, text)
            except httpx.TimeoutException:
                status = SourceFetchStatus.BUDGET_EXCEEDED if self._clock() >= overall_deadline else SourceFetchStatus.TIMEOUT
                return result(status)
            except httpx.HTTPError:
                return result(SourceFetchStatus.HTTP_ERROR)
            except Exception:
                return result(SourceFetchStatus.FAILED)

        return result(SourceFetchStatus.REDIRECT_LIMIT)

    @staticmethod
    def _extract_text(body: bytes, media_type: str, content_type_header: str) -> str:
        charset_match = re.search(r"(?:^|;)\s*charset\s*=\s*['\"]?([^;\s'\"]+)", content_type_header, re.IGNORECASE)
        charset = charset_match.group(1) if charset_match else "utf-8"
        codecs.lookup(charset)
        decoded = body.decode(charset, errors="replace")
        if media_type == "text/plain":
            return unicodedata.normalize("NFC", " ".join(decoded.split()))
        parser = _VisibleHTMLText()
        parser.feed(decoded)
        parser.close()
        return parser.visible_text()


news_source_retrieval = NewsSourceRetrievalService()
