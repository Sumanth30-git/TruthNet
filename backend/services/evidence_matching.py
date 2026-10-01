"""Conservative, deterministic claim-to-source matching for Phase 3E.

This module deliberately relies only on text and metadata already returned by
search.  It records uncertainty instead of trying to infer facts from semantic
similarity or a source's domain suffix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from backend.schemas import (
    ClaimContext,
    ClaimSourceRelevance,
    EvidenceMatch,
    Freshness,
    MatchStatus,
    SearchResult,
)


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "the", "this", "that", "to", "with", "will", "was",
    "were", "today", "yesterday", "current", "currently", "latest", "announced",
    "announcement", "company", "government", "organization", "product", "event",
}

_LOCATION_ALIASES = {
    "bengaluru": {"bengaluru", "bangalore"},
    "mumbai": {"mumbai", "bombay"},
    "new delhi": {"new delhi", "delhi"},
    "chennai": {"chennai", "madras"},
    "kolkata": {"kolkata", "calcutta"},
    "karnataka": {"karnataka"},
    "maharashtra": {"maharashtra"},
    "india": {"india"},
    "united states": {"united states", "usa", "u.s.", "us"},
    "united kingdom": {"united kingdom", "uk", "u.k."},
}

_LOCATION_PARENTS = {
    "bengaluru": {"karnataka", "india"},
    "mumbai": {"maharashtra", "india"},
    "new delhi": {"india"},
    "chennai": {"india"},
    "kolkata": {"india"},
    "karnataka": {"india"},
    "maharashtra": {"india"},
}


@dataclass(frozen=True)
class EvidenceMatchingConfig:
    """Visible context-sensitive windows used only for time-sensitive claims."""

    financial_current_hours: float = 36.0
    general_current_hours: float = 48.0
    recent_hours: float = 7 * 24.0

    def __post_init__(self) -> None:
        values = (self.financial_current_hours, self.general_current_hours, self.recent_hours)
        if any(value <= 0 for value in values):
            raise ValueError("Evidence matching windows must be positive.")
        if self.recent_hours < max(self.financial_current_hours, self.general_current_hours):
            raise ValueError("The recent window must be at least as long as current windows.")


def _source_text(source: SearchResult) -> str:
    return " ".join(part for part in (source.title, source.snippet or "") if part).strip()


def _normalise(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _tokens(value: str) -> set[str]:
    return {
        _stem(token)
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 1 and token not in _STOP_WORDS
    }


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _canonical_location(location: str) -> str:
    normalised = _normalise(location)
    for canonical, aliases in _LOCATION_ALIASES.items():
        if normalised in {_normalise(alias) for alias in aliases}:
            return canonical
    return normalised


def _known_locations_in(text: str) -> set[str]:
    lower = text.lower()
    locations: set[str] = set()
    for canonical, aliases in _LOCATION_ALIASES.items():
        if any(re.search(rf"(?<!\w){re.escape(alias.lower())}(?!\w)", lower) for alias in aliases):
            locations.add(canonical)
    return locations


def _entity_match(context: ClaimContext, source_text: str) -> MatchStatus:
    if not context.subject:
        return MatchStatus.UNKNOWN
    subject = _normalise(context.subject)
    if not subject:
        return MatchStatus.UNKNOWN
    content = _normalise(source_text)
    if subject in content:
        return MatchStatus.MATCH

    subject_tokens = _tokens(context.subject)
    content_tokens = _tokens(source_text)
    if subject_tokens and subject_tokens <= content_tokens:
        return MatchStatus.MATCH
    if len(subject_tokens) > 1 and subject_tokens & content_tokens:
        return MatchStatus.PARTIAL_MATCH

    # Detect a small number of explicit "Company X"-style mismatches without
    # treating a mere omitted name as a contradiction.
    entity_pattern = re.compile(r"\b(company|organization|government|ministry|team)\s+([a-z0-9-]+)", re.I)
    claim_entities = {(kind.lower(), value.lower()) for kind, value in entity_pattern.findall(context.factual_statement)}
    source_entities = {(kind.lower(), value.lower()) for kind, value in entity_pattern.findall(source_text)}
    if claim_entities and source_entities and not (claim_entities & source_entities):
        return MatchStatus.MISMATCH
    commodities = {"gold", "silver", "platinum", "bitcoin", "crude"}
    claim_commodities = commodities & _tokens(context.factual_statement)
    source_commodities = commodities & content_tokens
    if claim_commodities and source_commodities and claim_commodities.isdisjoint(source_commodities):
        return MatchStatus.MISMATCH
    return MatchStatus.UNKNOWN


def _event_match(context: ClaimContext, source_text: str) -> MatchStatus:
    if not context.event:
        return MatchStatus.UNKNOWN
    event_tokens = _tokens(context.event)
    if not event_tokens:
        return MatchStatus.UNKNOWN
    source_tokens = _tokens(source_text)
    overlap = event_tokens & source_tokens
    if event_tokens <= source_tokens:
        return MatchStatus.MATCH
    if overlap:
        return MatchStatus.PARTIAL_MATCH
    return MatchStatus.UNKNOWN


def _location_match(context: ClaimContext, source_text: str) -> MatchStatus:
    if not context.location:
        return MatchStatus.NOT_APPLICABLE
    claim_location = _canonical_location(context.location)
    source_locations = _known_locations_in(source_text)
    if not source_locations:
        return MatchStatus.UNKNOWN
    if claim_location in source_locations:
        return MatchStatus.MATCH
    if _LOCATION_PARENTS.get(claim_location, set()) & source_locations:
        return MatchStatus.PARTIAL_MATCH
    return MatchStatus.MISMATCH


def _parse_published_at(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.fromisoformat(f"{raw}T00:00:00")
        except ValueError:
            return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _time_terms(value: str | None) -> set[str]:
    return set(re.findall(r"[a-z]+", (value or "").lower()))


def _time_match(context: ClaimContext, source: SearchResult) -> MatchStatus:
    reference = (context.time_reference or "").strip()
    if not reference:
        return MatchStatus.NOT_APPLICABLE
    published = _parse_published_at(source.published_at)
    if published is None:
        return MatchStatus.UNKNOWN
    retrieved = source.retrieved_at.astimezone(timezone.utc)
    terms = _time_terms(reference)
    if "today" in terms or terms & {"current", "currently", "latest", "now"}:
        return MatchStatus.MATCH if published.date() == retrieved.date() else MatchStatus.MISMATCH
    if "yesterday" in terms:
        return MatchStatus.MATCH if (retrieved.date() - published.date()).days == 1 else MatchStatus.MISMATCH
    if "week" in terms:
        return MatchStatus.MATCH if 0 <= (retrieved - published).total_seconds() <= 7 * 24 * 3600 else MatchStatus.MISMATCH
    iso_date = re.search(r"\b\d{4}-\d{2}-\d{2}\b", reference)
    if iso_date:
        return MatchStatus.MATCH if published.date().isoformat() == iso_date.group(0) else MatchStatus.MISMATCH
    year = re.search(r"\b(19|20)\d{2}\b", reference)
    if year:
        # A later source can accurately report a historical event, so a different
        # publication year is not itself a mismatch.
        return MatchStatus.MATCH if published.year == int(year.group(0)) else MatchStatus.UNKNOWN
    return MatchStatus.UNKNOWN


def _is_time_sensitive(context: ClaimContext) -> bool:
    terms = _time_terms(context.time_reference)
    return bool(terms & {"today", "yesterday", "current", "currently", "latest", "now", "week"}) or (
        context.claim_type.value == "financial_price"
    )


def assess_freshness(
    context: ClaimContext,
    source: SearchResult,
    config: EvidenceMatchingConfig | None = None,
) -> Freshness:
    """Assess age only when the claim context makes age material."""
    config = config or EvidenceMatchingConfig()
    published = _parse_published_at(source.published_at)
    if published is None or not _is_time_sensitive(context):
        return Freshness.UNKNOWN
    age_hours = max(0.0, (source.retrieved_at.astimezone(timezone.utc) - published).total_seconds() / 3600)
    current_hours = (
        config.financial_current_hours
        if context.claim_type.value == "financial_price"
        else config.general_current_hours
    )
    if age_hours <= current_hours:
        return Freshness.CURRENT
    if age_hours <= config.recent_hours:
        return Freshness.RECENT
    return Freshness.STALE


def _relevance(
    entity: MatchStatus,
    event: MatchStatus,
    location: MatchStatus,
    time: MatchStatus,
) -> ClaimSourceRelevance:
    if entity == MatchStatus.MISMATCH or location == MatchStatus.MISMATCH:
        return ClaimSourceRelevance.LOW
    direct_matches = sum(match == MatchStatus.MATCH for match in (entity, event, location, time))
    partial_matches = sum(match == MatchStatus.PARTIAL_MATCH for match in (entity, event, location, time))
    if direct_matches >= 2 or (direct_matches >= 1 and partial_matches >= 1):
        return ClaimSourceRelevance.HIGH
    if direct_matches == 1 or partial_matches >= 2:
        return ClaimSourceRelevance.MEDIUM
    if partial_matches == 1:
        return ClaimSourceRelevance.LOW
    return ClaimSourceRelevance.NOT_ASSESSED


def _reason(
    relevance: ClaimSourceRelevance,
    entity: MatchStatus,
    event: MatchStatus,
    location: MatchStatus,
    time: MatchStatus,
) -> str:
    details = []
    if entity == MatchStatus.MATCH:
        details.append("the claimed subject")
    elif entity == MatchStatus.MISMATCH:
        details.append("a different explicit subject")
    if event == MatchStatus.MATCH:
        details.append("the stated event")
    if location == MatchStatus.MATCH:
        details.append("the requested location")
    elif location == MatchStatus.PARTIAL_MATCH:
        details.append("a broader location")
    elif location == MatchStatus.MISMATCH:
        details.append("a different location")
    if time == MatchStatus.MATCH:
        details.append("the stated time")
    elif time == MatchStatus.MISMATCH:
        details.append("a different time")
    if not details:
        return "Available source text did not provide enough deterministic matching signals."
    qualifier = {
        ClaimSourceRelevance.HIGH: "Source matches",
        ClaimSourceRelevance.MEDIUM: "Source partially matches",
        ClaimSourceRelevance.LOW: "Source has limited relevance because it identifies",
        ClaimSourceRelevance.NOT_ASSESSED: "Source matching was not established from",
    }[relevance]
    return f"{qualifier} {', '.join(details)}."


class EvidenceMatchingService:
    def __init__(self, config: EvidenceMatchingConfig | None = None) -> None:
        self.config = config or EvidenceMatchingConfig()

    def match(self, context: ClaimContext, source: SearchResult) -> EvidenceMatch:
        text = _source_text(source)
        entity = _entity_match(context, text)
        event = _event_match(context, text)
        location = _location_match(context, text)
        time = _time_match(context, source)
        relevance = _relevance(entity, event, location, time)
        return EvidenceMatch(
            relevance=relevance,
            entity_match=entity,
            event_match=event,
            location_match=location,
            time_match=time,
            reason=_reason(relevance, entity, event, location, time),
        )

    def freshness(self, context: ClaimContext, source: SearchResult) -> Freshness:
        return assess_freshness(context, source, self.config)


evidence_matching = EvidenceMatchingService()
