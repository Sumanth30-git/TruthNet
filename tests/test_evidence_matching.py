from datetime import datetime, timedelta, timezone

from backend.schemas import ClaimContext, ClaimSourceRelevance, ClaimType, MatchStatus, SearchResult
from backend.services.evidence_matching import EvidenceMatchingConfig, EvidenceMatchingService


NOW = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)


def _source(text: str, *, published_at: str | None = None) -> SearchResult:
    return SearchResult(
        title="Source report",
        url="https://example.org/report",
        snippet=text,
        published_at=published_at,
        retrieved_at=NOW,
    )


def _context(**overrides) -> ClaimContext:
    values = {
        "claim_id": "claim-1",
        "factual_statement": "Company Alpha launched Product Y in Bengaluru today.",
        "subject": "Company Alpha",
        "claim_type": ClaimType.PRODUCT_ANNOUNCEMENT,
        "event": "launched Product Y",
        "location": "Bengaluru",
        "time_reference": "today",
    }
    values.update(overrides)
    return ClaimContext(**values)


def test_matching_entity_event_location_and_time_is_high_relevance():
    service = EvidenceMatchingService()
    result = service.match(
        _context(),
        _source("Company Alpha launched Product Y in Bengaluru today.", published_at="2026-09-24T08:00:00Z"),
    )
    assert result.relevance == ClaimSourceRelevance.HIGH
    assert result.entity_match == MatchStatus.MATCH
    assert result.event_match == MatchStatus.MATCH
    assert result.location_match == MatchStatus.MATCH
    assert result.time_match == MatchStatus.MATCH


def test_explicitly_different_entity_is_low_relevance():
    result = EvidenceMatchingService().match(
        _context(), _source("Company Beta launched Product Y in Bengaluru today.", published_at="2026-09-24")
    )
    assert result.entity_match == MatchStatus.MISMATCH
    assert result.relevance == ClaimSourceRelevance.LOW


def test_location_matching_handles_exact_broader_and_mismatch():
    service = EvidenceMatchingService()
    exact = service.match(_context(), _source("Company Alpha launched Product Y in Bengaluru."))
    broader = service.match(_context(), _source("Company Alpha launched Product Y in Karnataka."))
    mismatch = service.match(_context(), _source("Company Alpha launched Product Y in Mumbai."))
    assert exact.location_match == MatchStatus.MATCH
    assert broader.location_match == MatchStatus.PARTIAL_MATCH
    assert mismatch.location_match == MatchStatus.MISMATCH


def test_time_sensitive_claims_distinguish_current_stale_and_unknown_sources():
    service = EvidenceMatchingService(EvidenceMatchingConfig(general_current_hours=48, financial_current_hours=24, recent_hours=168))
    context = _context()
    current = service.freshness(context, _source("report", published_at="2026-09-23T13:00:00Z"))
    stale = service.freshness(context, _source("report", published_at="2026-09-10T12:00:00Z"))
    unknown = service.freshness(context, _source("report", published_at=None))
    assert current.value == "current"
    assert stale.value == "stale"
    assert unknown.value == "unknown"


def test_historical_claim_does_not_penalize_an_older_source_by_default():
    context = _context(time_reference="2010", claim_type=ClaimType.EVENT)
    freshness = EvidenceMatchingService().freshness(context, _source("report", published_at="2010-01-01"))
    assert freshness.value == "unknown"
