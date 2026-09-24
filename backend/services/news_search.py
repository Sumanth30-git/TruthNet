"""Provider-independent search boundary for current-news source candidates."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from backend.schemas import SearchResponse, SearchResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchConfig:
    provider: str = "none"
    timeout_seconds: float = 5.0
    max_results: int = 5

    @classmethod
    def from_env(cls) -> "SearchConfig":
        try:
            timeout = float(os.getenv("NEWS_SEARCH_TIMEOUT_SECONDS", "5"))
            if timeout <= 0 or timeout > 60:
                raise ValueError
        except ValueError:
            timeout = 5.0
        try:
            maximum = int(os.getenv("NEWS_SEARCH_MAX_RESULTS", "5"))
            if maximum < 1 or maximum > 20:
                raise ValueError
        except ValueError:
            maximum = 5
        return cls(
            provider=os.getenv("NEWS_SEARCH_PROVIDER", "none").strip().lower() or "none",
            timeout_seconds=timeout,
            max_results=maximum,
        )


class SearchProvider(Protocol):
    """Adapter contract. Providers return mappings from their documented API."""

    name: str

    def search(self, query: str, *, timeout_seconds: float, max_results: int) -> Sequence[Mapping[str, Any]]: ...


class SearchTimeoutError(Exception):
    pass


class SearchUnavailableError(Exception):
    pass


class SearchRateLimitError(Exception):
    pass


class TavilySearchProvider:
    """Tavily Search API adapter using its documented HTTP endpoint."""

    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, *, post=httpx.post):
        self._api_key = api_key if api_key is not None else os.getenv("TAVILY_API_KEY", "").strip()
        self._post = post

    def search(self, query: str, *, timeout_seconds: float, max_results: int) -> Sequence[Mapping[str, Any]]:
        if not self._api_key:
            raise SearchUnavailableError("Tavily API key is not configured")
        try:
            response = self._post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"query": query, "search_depth": "basic", "max_results": max_results, "topic": "general"},
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise SearchTimeoutError from exc
        except httpx.HTTPError as exc:
            raise SearchUnavailableError from exc
        if response.status_code == 429:
            raise SearchRateLimitError
        if response.status_code < 200 or response.status_code >= 300:
            raise SearchUnavailableError
        try:
            payload = response.json()
        except (ValueError, TypeError) as exc:
            raise ValueError("malformed Tavily response") from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("results"), list):
            raise ValueError("malformed Tavily response")
        mapped = []
        for item in payload["results"]:
            if not isinstance(item, Mapping):
                mapped.append(item)
                continue
            url = item.get("url")
            domain = urlparse(url).hostname if isinstance(url, str) else None
            mapped.append({
                "title": item.get("title"),
                "url": url,
                "snippet": item.get("content"),
                "publisher": domain,
                "published_at": item.get("published_date"),
            })
        return mapped


class SearchService:
    def __init__(self, config: SearchConfig | None = None, provider: SearchProvider | None = None):
        self.config = config or SearchConfig.from_env()
        self.provider = provider

    def search(self, query: str) -> SearchResponse:
        if self.config.provider == "none":
            return SearchResponse(status="not_configured", provider="none", message="No search provider is configured.")
        provider = self.provider
        if provider is None and self.config.provider == "tavily":
            provider = TavilySearchProvider()
        if not provider or provider.name != self.config.provider:
            return SearchResponse(status="unavailable", provider=self.config.provider, message="The configured search provider is unavailable.")
        try:
            raw = provider.search(query, timeout_seconds=self.config.timeout_seconds, max_results=self.config.max_results)
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError("provider response is not a result sequence")
            results = []
            for item in raw[: self.config.max_results]:
                candidate = self._parse_result(item)
                if candidate is not None:
                    results.append(candidate)
            return SearchResponse(status="complete", provider=provider.name, results=results)
        except (TimeoutError, SearchTimeoutError):
            return SearchResponse(status="failed", provider=provider.name, message="Search timed out.")
        except SearchRateLimitError:
            return SearchResponse(status="unavailable", provider=provider.name, message="Search provider rate limit reached.")
        except (ConnectionError, OSError, SearchUnavailableError):
            return SearchResponse(status="unavailable", provider=provider.name, message="Search provider is unavailable or not configured.")
        except Exception:
            # Provider exception text can contain request details; never include it in logs or responses.
            logger.warning("News search provider failed")
            return SearchResponse(status="failed", provider=provider.name, message="Search provider returned an invalid response or failed.")

    @staticmethod
    def _parse_result(item: Any) -> SearchResult | None:
        if not isinstance(item, Mapping):
            return None
        title = item.get("title")
        url = item.get("url")
        if not isinstance(title, str) or not title.strip() or not isinstance(url, str):
            return None
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        optional = lambda key: item.get(key) if isinstance(item.get(key), str) else None
        return SearchResult(
            title=title.strip(), url=url.strip(), snippet=optional("snippet"),
            publisher=optional("publisher"), published_at=optional("published_at"),
            retrieved_at=datetime.now(timezone.utc),
        )


news_search = SearchService()
