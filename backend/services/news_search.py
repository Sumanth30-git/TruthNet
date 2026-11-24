"""Provider-independent search boundary for current-news source candidates."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from backend.schemas import Claim, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

_QUERY_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "the", "this", "that", "to", "with", "will", "was", "were",
})
_QUERY_TOKEN_RE = re.compile(r"[^\W_]+(?:[.'’/-][^\W_]+)*%?", re.UNICODE)


def _claim_id(raw_claim: object) -> str | None:
    try:
        claim = raw_claim if isinstance(raw_claim, Claim) else Claim.model_validate(raw_claim)
    except Exception:
        return None
    if isinstance(claim.claim_id, str) and claim.claim_id:
        return claim.claim_id
    return None


def _union_claim_ids(existing: Sequence[str], incoming: Sequence[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for claim_id in (*existing, *incoming):
        if not isinstance(claim_id, str) or not claim_id or claim_id in seen:
            continue
        merged.append(claim_id)
        seen.add(claim_id)
    return merged


def build_claim_search_query(raw_claim: object) -> str | None:
    """Build a stable search query from one claim and only its extracted context."""
    try:
        claim = raw_claim if isinstance(raw_claim, Claim) else Claim.model_validate(raw_claim)
    except Exception:
        return None
    text = " ".join(claim.text.split()) if isinstance(claim.text, str) else ""
    if not text:
        return None

    context = claim.context
    structured_terms: list[str] = []
    if context is not None:
        structured_terms.extend(
            value for value in (
                context.subject,
                *(entity.text for entity in context.entity_mentions),
                context.event,
                context.location,
                context.time_reference,
                *(number.raw_text for number in context.numerical_values),
            )
            if isinstance(value, str) and value.strip()
        )

    parts: list[str] = []
    seen_phrases: set[str] = set()
    covered_tokens: set[str] = set()
    for term in structured_terms:
        phrase = " ".join(term.split())
        key = phrase.casefold()
        if key in seen_phrases:
            continue
        seen_phrases.add(key)
        parts.append(phrase)
        covered_tokens.update(token.casefold() for token in _QUERY_TOKEN_RE.findall(phrase))

    # Preserve claim-specific wording that is not already represented by the
    # structured fields, while omitting common connective words.
    for token in _QUERY_TOKEN_RE.findall(text):
        folded = token.casefold()
        if folded in covered_tokens or folded in _QUERY_STOP_WORDS:
            continue
        if folded not in seen_phrases:
            parts.append(token)
            seen_phrases.add(folded)

    return " ".join(parts) or text


@dataclass(frozen=True)
class SearchConfig:
    provider: str = "none"
    timeout_seconds: float = 5.0
    max_results: int = 5
    max_claim_queries: int = 3

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

    def search_claims(self, claims: Sequence[object]) -> SearchResponse:
        """Search a bounded set of deterministic queries derived from claims."""
        if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
            claims = ()
        query_entries: list[tuple[str, list[str]]] = []
        seen_queries: dict[str, int] = {}
        for raw_claim in claims:
            query = build_claim_search_query(raw_claim)
            if not query:
                continue
            key = query.casefold()
            claim_id = _claim_id(raw_claim)
            if key in seen_queries:
                if claim_id:
                    origin_ids = query_entries[seen_queries[key]][1]
                    if claim_id not in origin_ids:
                        origin_ids.append(claim_id)
                continue
            seen_queries[key] = len(query_entries)
            query_entries.append((query, [claim_id] if claim_id else []))

        if not query_entries:
            return SearchResponse(
                status="not_run",
                provider=self.config.provider,
                message="No usable claims were available to search.",
            )
        if self.config.provider == "none":
            return SearchResponse(
                status="not_configured",
                provider="none",
                message="No search provider is configured.",
            )

        query_limit = max(1, min(self.config.max_claim_queries, self.config.max_results))
        selected_queries = query_entries[:query_limit]
        query_set_was_limited = len(selected_queries) < len(query_entries)
        responses = [self.search(query) for query, _origin_ids in selected_queries]
        successful = [response for response in responses if response.status == "complete"]

        results: list[SearchResult] = []
        index_by_url: dict[str, int] = {}
        for response, (_query, origin_ids) in zip(responses, selected_queries):
            if response.status != "complete":
                continue
            for result in response.results:
                url_key = result.url.casefold().rstrip("/")
                if url_key in index_by_url:
                    existing = results[index_by_url[url_key]]
                    existing.claim_ids = _union_claim_ids(existing.claim_ids, origin_ids)
                    continue
                if len(results) >= self.config.max_results:
                    continue
                tagged = result.model_copy(update={"claim_ids": list(origin_ids)})
                index_by_url[url_key] = len(results)
                results.append(tagged)

        if len(successful) == len(responses) and not query_set_was_limited:
            status = "complete"
            message = None
        elif successful:
            status = "partial"
            message = "Search completed for only part of the available claim queries."
        elif any(response.status == "failed" for response in responses):
            status = "failed"
            message = "Claim search failed."
        elif any(response.status == "unavailable" for response in responses):
            status = "unavailable"
            message = "Claim search is unavailable or not configured."
        else:
            status = "not_configured"
            message = "No search provider is configured."

        return SearchResponse(
            status=status,
            provider=self.config.provider,
            results=results[: self.config.max_results],
            message=message,
        )

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
