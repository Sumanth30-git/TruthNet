from datetime import datetime, timezone

from backend.schemas import (
    AuthorityLevel,
    ClaimContext,
    ClaimType,
    Officiality,
    SearchResult,
    SourceRelationship,
    SourceType,
)
from backend.services.source_quality import (
    SourceIdentity,
    SourceQualityCatalog,
    SourceQualityService,
    profile_source,
)


def _claim(
    *,
    subject: str | None = "Acme Labs",
    claim_type: ClaimType = ClaimType.COMPANY_ANNOUNCEMENT,
    statement: str = "Acme Labs announced a product launch.",
) -> ClaimContext:
    return ClaimContext(
        claim_id="claim-1",
        factual_statement=statement,
        subject=subject,
        claim_type=claim_type,
    )


def _catalog() -> SourceQualityCatalog:
    official = SourceIdentity(
        owner="Acme Labs",
        source_type=SourceType.OFFICIAL_PRIMARY,
        authority_level=AuthorityLevel.VERY_HIGH,
        officiality=Officiality.VERIFIED_OFFICIAL,
        default_relationship=SourceRelationship.OFFICIAL_STATEMENT,
        authority_reason="The source is an exact configured Acme Labs identity.",
        aliases=("Acme",),
        primary_when_owner_matches=True,
    )
    specialist = SourceIdentity(
        owner="Example Market Data",
        source_type=SourceType.SPECIALIST_FINANCIAL,
        authority_level=AuthorityLevel.HIGH,
        officiality=Officiality.VERIFIED_OFFICIAL,
        default_relationship=SourceRelationship.OFFICIAL_STATEMENT,
        authority_reason="The source is an exact configured market-data institution identity.",
        direct_data_claim_types=frozenset({ClaimType.FINANCIAL_PRICE.value}),
    )
    official_social = SourceIdentity(
        owner="Acme Labs",
        source_type=SourceType.SOCIAL_OFFICIAL,
        authority_level=AuthorityLevel.HIGH,
        officiality=Officiality.VERIFIED_OFFICIAL,
        default_relationship=SourceRelationship.OFFICIAL_STATEMENT,
        authority_reason="The social account is an exact configured Acme Labs identity.",
        aliases=("Acme",),
        primary_when_owner_matches=True,
    )
    return SourceQualityCatalog(
        domains={
            "acme.example": official,
            "market.example": specialist,
        },
        social_accounts={"x.com/acme": official_social},
    )


def test_exact_configured_official_source_is_primary_only_when_owner_matches_claim():
    service = SourceQualityService(_catalog())
    profile = service.profile(
        {
            "url": "https://news.acme.example/releases/product",
            "publisher": "Acme Labs",
            "title": "Product release",
        },
        _claim(),
    )

    assert profile.domain == "news.acme.example"
    assert profile.source_type == SourceType.OFFICIAL_PRIMARY
    assert profile.authority_level == AuthorityLevel.VERY_HIGH
    assert profile.officiality == Officiality.VERIFIED_OFFICIAL
    assert profile.primary_source is True
    assert profile.source_relationship == SourceRelationship.DIRECT_PRIMARY
    assert "configured Acme Labs identity" in profile.authority_reason

    unrelated = service.profile(
        {"url": "https://acme.example/releases/product", "publisher": "Acme Labs"},
        _claim(subject="Other Organization", statement="Other Organization launched a product."),
    )
    assert unrelated.primary_source is False
    assert unrelated.source_relationship == SourceRelationship.OFFICIAL_STATEMENT


def test_exact_established_news_identity_is_independent_reporting():
    profile = profile_source(
        SearchResult(
            url="https://www.reuters.com/world/example",
            publisher="Reuters",
            title="Report",
            retrieved_at=datetime.now(timezone.utc),
        )
    )

    assert profile.domain == "reuters.com"
    assert profile.source_type == SourceType.ESTABLISHED_NEWS
    assert profile.authority_level == AuthorityLevel.HIGH
    assert profile.primary_source is False
    assert profile.officiality == Officiality.NOT_APPLICABLE
    assert profile.source_relationship == SourceRelationship.INDEPENDENT_REPORTING
    assert "exact configured publisher identity" in profile.authority_reason


def test_specialist_financial_identity_is_direct_data_provider_only_for_financial_claim():
    service = SourceQualityService(_catalog())
    financial = service.profile(
        {"url": "https://market.example/prices/gold", "publisher": "Example Market Data"},
        _claim(subject="Gold", claim_type=ClaimType.FINANCIAL_PRICE, statement="Gold is priced at 100 today."),
    )

    assert financial.source_type == SourceType.SPECIALIST_FINANCIAL
    assert financial.authority_level == AuthorityLevel.HIGH
    assert financial.primary_source is True
    assert financial.source_relationship == SourceRelationship.DIRECT_DATA_PROVIDER
    assert "direct data provider" in financial.authority_reason

    non_financial = service.profile(
        {"url": "https://market.example/notices/example", "publisher": "Example Market Data"},
        _claim(),
    )
    assert non_financial.primary_source is False
    assert non_financial.source_relationship == SourceRelationship.OFFICIAL_STATEMENT


def test_exact_configured_social_account_is_official_but_unlisted_account_is_not():
    service = SourceQualityService(_catalog())
    official = service.profile(
        {"url": "https://x.com/acme/status/123", "publisher": "Acme Labs"},
        _claim(),
    )
    assert official.platform == "x"
    assert official.source_type == SourceType.SOCIAL_OFFICIAL
    assert official.officiality == Officiality.VERIFIED_OFFICIAL
    assert official.primary_source is True
    assert official.source_relationship == SourceRelationship.DIRECT_PRIMARY

    unverified = service.profile({"url": "https://x.com/unlisted_account/status/123", "publisher": "Account"})
    assert unverified.platform == "x"
    assert unverified.source_type == SourceType.SOCIAL_UNVERIFIED
    assert unverified.authority_level == AuthorityLevel.UNKNOWN
    assert unverified.primary_source is None
    assert unverified.officiality == Officiality.UNVERIFIED
    assert unverified.source_relationship == SourceRelationship.UNVERIFIED_REPOST
    assert "not established" in unverified.authority_reason


def test_unlisted_tlds_are_not_authority_signals():
    service = SourceQualityService(SourceQualityCatalog())

    for url in ("https://unlisted.gov/notice", "https://unlisted.org/report", "https://unlisted.com/article"):
        profile = service.profile({"url": url, "publisher": "Unlisted"})
        assert profile.source_type == SourceType.UNKNOWN
        assert profile.authority_level == AuthorityLevel.UNKNOWN
        assert profile.primary_source is None
        assert profile.officiality == Officiality.NOT_ESTABLISHED
        assert profile.source_relationship == SourceRelationship.UNKNOWN


def test_commentary_and_malformed_metadata_fail_safely_without_claiming_authority():
    service = SourceQualityService(SourceQualityCatalog())
    commentary = service.profile(
        {
            "url": "https://unknown.example/article",
            "title": "Opinion: a perspective",
            "snippet": "Commentary about the reported event.",
        }
    )
    assert commentary.source_relationship == SourceRelationship.COMMENTARY
    assert commentary.authority_level == AuthorityLevel.UNKNOWN

    malformed = service.profile({"url": "file:///private/path", "publisher": 42})
    assert malformed.domain is None
    assert malformed.publisher is None
    assert malformed.source_type == SourceType.UNKNOWN
    assert malformed.authority_level == AuthorityLevel.UNKNOWN
    assert "does not establish" in malformed.authority_reason
