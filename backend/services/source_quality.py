"""Conservative source-identity profiling for Phase 3E evidence assessment.

This module describes what can be established from a retrieved source's identity.
It deliberately does not decide whether a claim is true, whether a source is
relevant, or whether an unlisted domain is authoritative. In particular, a
top-level domain such as ``.gov`` or ``.org`` is never used as an authority
signal. Higher-authority outcomes require an exact, locally configured source
identity or account.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from urllib.parse import urlparse

from backend.schemas import (
    AuthorityLevel,
    Officiality,
    SourceProfile,
    SourceRelationship,
    SourceType,
)


_SOCIAL_PLATFORMS = {
    "x.com": "x",
    "twitter.com": "x",
    "facebook.com": "facebook",
    "instagram.com": "instagram",
    "youtube.com": "youtube",
    "tiktok.com": "tiktok",
    "linkedin.com": "linkedin",
    "threads.net": "threads",
    "reddit.com": "reddit",
    "telegram.me": "telegram",
    "t.me": "telegram",
}
_COMMENTARY_MARKERS = ("opinion", "editorial", "op-ed", "commentary")


@dataclass(frozen=True)
class SourceIdentity:
    """A reviewed, exact source identity used by :class:`SourceQualityService`.

    The catalog is intentionally explicit. It is an identity allow-list rather
    than a reputation score or a TLD rule. ``primary_when_owner_matches`` only
    marks a source primary when the configured owner appears exactly in the
    provided claim context. ``direct_data_claim_types`` uses ClaimType values
    (for example, ``"financial_price"``) when a source is a direct data
    provider for that category.
    """

    owner: str
    source_type: SourceType
    authority_level: AuthorityLevel
    officiality: Officiality
    default_relationship: SourceRelationship
    authority_reason: str
    aliases: tuple[str, ...] = ()
    primary_when_owner_matches: bool = False
    direct_data_claim_types: frozenset[str] = frozenset()
    primary_by_default: bool = False


@dataclass(frozen=True)
class SourceQualityCatalog:
    """Exact domain and social-account identities available to the service.

    Keys in ``domains`` are domain names such as ``"example.org"``. Keys in
    ``social_accounts`` identify an account root such as ``"x.com/example"``.
    Values must be :class:`SourceIdentity` instances. Invalid entries are
    ignored so malformed configuration cannot fail a News request.
    """

    domains: Mapping[str, SourceIdentity] = field(default_factory=dict)
    social_accounts: Mapping[str, SourceIdentity] = field(default_factory=dict)


def _identity(
    *,
    owner: str,
    source_type: SourceType,
    authority_level: AuthorityLevel,
    officiality: Officiality,
    default_relationship: SourceRelationship,
    authority_reason: str,
    aliases: tuple[str, ...] = (),
    primary_when_owner_matches: bool = False,
    direct_data_claim_types: frozenset[str] = frozenset(),
) -> SourceIdentity:
    return SourceIdentity(
        owner=owner,
        source_type=source_type,
        authority_level=authority_level,
        officiality=officiality,
        default_relationship=default_relationship,
        authority_reason=authority_reason,
        aliases=aliases,
        primary_when_owner_matches=primary_when_owner_matches,
        direct_data_claim_types=direct_data_claim_types,
    )


# This small catalog contains exact source identities only. It is deliberately
# not a general ranking of domain suffixes, publishers, or social platforms.
DEFAULT_SOURCE_QUALITY_CATALOG = SourceQualityCatalog(
    domains={
        "reuters.com": _identity(
            owner="Reuters",
            source_type=SourceType.ESTABLISHED_NEWS,
            authority_level=AuthorityLevel.HIGH,
            officiality=Officiality.NOT_APPLICABLE,
            default_relationship=SourceRelationship.INDEPENDENT_REPORTING,
            authority_reason=(
                "The source matches an exact configured publisher identity for independent news reporting."
            ),
        ),
        "apnews.com": _identity(
            owner="Associated Press",
            source_type=SourceType.ESTABLISHED_NEWS,
            authority_level=AuthorityLevel.HIGH,
            officiality=Officiality.NOT_APPLICABLE,
            default_relationship=SourceRelationship.INDEPENDENT_REPORTING,
            authority_reason=(
                "The source matches an exact configured publisher identity for independent news reporting."
            ),
            aliases=("AP",),
        ),
        "bbc.com": _identity(
            owner="BBC",
            source_type=SourceType.ESTABLISHED_NEWS,
            authority_level=AuthorityLevel.HIGH,
            officiality=Officiality.NOT_APPLICABLE,
            default_relationship=SourceRelationship.INDEPENDENT_REPORTING,
            authority_reason=(
                "The source matches an exact configured publisher identity for independent news reporting."
            ),
        ),
        "bbc.co.uk": _identity(
            owner="BBC",
            source_type=SourceType.ESTABLISHED_NEWS,
            authority_level=AuthorityLevel.HIGH,
            officiality=Officiality.NOT_APPLICABLE,
            default_relationship=SourceRelationship.INDEPENDENT_REPORTING,
            authority_reason=(
                "The source matches an exact configured publisher identity for independent news reporting."
            ),
        ),
        "sec.gov": _identity(
            owner="Securities and Exchange Commission",
            source_type=SourceType.REGULATORY_OR_LEGAL,
            authority_level=AuthorityLevel.VERY_HIGH,
            officiality=Officiality.VERIFIED_OFFICIAL,
            default_relationship=SourceRelationship.OFFICIAL_STATEMENT,
            authority_reason=(
                "The source matches an exact configured regulatory institution identity."
            ),
            aliases=("SEC",),
            primary_when_owner_matches=True,
        ),
        "rbi.org.in": _identity(
            owner="Reserve Bank of India",
            source_type=SourceType.REGULATORY_OR_LEGAL,
            authority_level=AuthorityLevel.VERY_HIGH,
            officiality=Officiality.VERIFIED_OFFICIAL,
            default_relationship=SourceRelationship.OFFICIAL_STATEMENT,
            authority_reason=(
                "The source matches an exact configured regulatory institution identity."
            ),
            aliases=("RBI",),
            primary_when_owner_matches=True,
        ),
        "sebi.gov.in": _identity(
            owner="Securities and Exchange Board of India",
            source_type=SourceType.REGULATORY_OR_LEGAL,
            authority_level=AuthorityLevel.VERY_HIGH,
            officiality=Officiality.VERIFIED_OFFICIAL,
            default_relationship=SourceRelationship.OFFICIAL_STATEMENT,
            authority_reason=(
                "The source matches an exact configured regulatory institution identity."
            ),
            aliases=("SEBI",),
            primary_when_owner_matches=True,
        ),
        "nseindia.com": _identity(
            owner="National Stock Exchange of India",
            source_type=SourceType.SPECIALIST_FINANCIAL,
            authority_level=AuthorityLevel.HIGH,
            officiality=Officiality.VERIFIED_OFFICIAL,
            default_relationship=SourceRelationship.OFFICIAL_STATEMENT,
            authority_reason=(
                "The source matches an exact configured market-data institution identity."
            ),
            aliases=("NSE",),
            primary_when_owner_matches=True,
            direct_data_claim_types=frozenset({"financial_price"}),
        ),
        "bseindia.com": _identity(
            owner="BSE",
            source_type=SourceType.SPECIALIST_FINANCIAL,
            authority_level=AuthorityLevel.HIGH,
            officiality=Officiality.VERIFIED_OFFICIAL,
            default_relationship=SourceRelationship.OFFICIAL_STATEMENT,
            authority_reason=(
                "The source matches an exact configured market-data institution identity."
            ),
            aliases=("Bombay Stock Exchange",),
            primary_when_owner_matches=True,
            direct_data_claim_types=frozenset({"financial_price"}),
        ),
        "who.int": _identity(
            owner="World Health Organization",
            source_type=SourceType.RESEARCH_INSTITUTION,
            authority_level=AuthorityLevel.HIGH,
            officiality=Officiality.VERIFIED_OFFICIAL,
            default_relationship=SourceRelationship.OFFICIAL_STATEMENT,
            authority_reason=(
                "The source matches an exact configured public-health institution identity."
            ),
            aliases=("WHO",),
            primary_when_owner_matches=True,
        ),
    }
)


class SourceQualityService:
    """Build conservative :class:`~backend.schemas.SourceProfile` results."""

    def __init__(self, catalog: SourceQualityCatalog | None = None) -> None:
        self.catalog = catalog or DEFAULT_SOURCE_QUALITY_CATALOG

    def profile(
        self,
        source: object,
        claim: object | None = None,
        *,
        claim_context: object | None = None,
    ) -> SourceProfile:
        """Profile a source without inferring claim truth or source relevance.

        ``source`` can be a ``SearchResult``, mapping, or object with ``url``,
        ``publisher``, ``title``, and ``snippet`` attributes. ``claim`` may be a
        ``ClaimContext``, a string, mapping, or similarly shaped object. The
        explicit ``claim_context`` keyword is provided for pipeline readability.
        """

        context = claim_context if claim_context is not None else claim
        url = _text_field(source, "url")
        publisher = _text_field(source, "publisher")
        title = _text_field(source, "title")
        snippet = _text_field(source, "snippet")
        domain = _domain_from_url(url)
        platform = _social_platform(domain)

        social_identity = self._social_identity(domain, url) if platform else None
        identity = social_identity or self._domain_identity(domain)
        if identity is not None:
            return self._profile_identity(
                identity,
                publisher=publisher,
                domain=domain,
                platform=platform,
                title=title,
                snippet=snippet,
                claim_context=context,
            )

        if platform:
            return SourceProfile(
                publisher=publisher,
                domain=domain,
                platform=platform,
                source_type=SourceType.SOCIAL_UNVERIFIED,
                authority_level=AuthorityLevel.UNKNOWN,
                primary_source=None,
                officiality=Officiality.UNVERIFIED,
                source_relationship=SourceRelationship.UNVERIFIED_REPOST,
                authority_reason=(
                    "The source is a social-platform account whose official identity is not established by available metadata."
                ),
            )

        if _looks_like_commentary(title, snippet):
            return SourceProfile(
                publisher=publisher,
                domain=domain,
                platform=None,
                source_type=SourceType.UNKNOWN,
                authority_level=AuthorityLevel.UNKNOWN,
                primary_source=None,
                officiality=Officiality.NOT_ESTABLISHED,
                source_relationship=SourceRelationship.COMMENTARY,
                authority_reason=(
                    "The available title or snippet labels this item as commentary; source authority is not established."
                ),
            )

        return SourceProfile(
            publisher=publisher,
            domain=domain,
            platform=None,
            source_type=SourceType.UNKNOWN,
            authority_level=AuthorityLevel.UNKNOWN,
            primary_source=None,
            officiality=Officiality.NOT_ESTABLISHED,
            source_relationship=SourceRelationship.UNKNOWN,
            authority_reason=(
                "The available source metadata does not establish an exact source identity, official status, or direct relationship to the claim."
            ),
        )

    def _profile_identity(
        self,
        identity: SourceIdentity,
        *,
        publisher: str | None,
        domain: str | None,
        platform: str | None,
        title: str | None,
        snippet: str | None,
        claim_context: object | None,
    ) -> SourceProfile:
        relationship = identity.default_relationship
        primary_source = identity.primary_by_default
        reason = identity.authority_reason

        if _claim_type_value(claim_context) in identity.direct_data_claim_types:
            relationship = SourceRelationship.DIRECT_DATA_PROVIDER
            primary_source = True
            reason = f"{reason} It is configured as a direct data provider for this claim category."
        elif identity.primary_when_owner_matches and _owner_matches_claim(identity, claim_context):
            relationship = SourceRelationship.DIRECT_PRIMARY
            primary_source = True
            reason = f"{reason} The configured source owner is named in the claim context."

        if _looks_like_commentary(title, snippet) and relationship in {
            SourceRelationship.INDEPENDENT_REPORTING,
            SourceRelationship.SECONDARY_REPORTING,
        }:
            relationship = SourceRelationship.COMMENTARY
            primary_source = False
            reason = f"{reason} The available title or snippet labels this item as commentary."

        return SourceProfile(
            publisher=publisher,
            domain=domain,
            platform=platform,
            source_type=identity.source_type,
            authority_level=identity.authority_level,
            primary_source=primary_source,
            officiality=identity.officiality,
            source_relationship=relationship,
            authority_reason=reason,
        )

    def _domain_identity(self, domain: str | None) -> SourceIdentity | None:
        if not domain:
            return None
        for configured_domain, identity in self.catalog.domains.items():
            normalized = _normalize_domain(configured_domain)
            if (
                normalized
                and isinstance(identity, SourceIdentity)
                and (domain == normalized or domain.endswith(f".{normalized}"))
            ):
                return identity
        return None

    def _social_identity(self, domain: str | None, url: str | None) -> SourceIdentity | None:
        account_key = _social_account_key(domain, url)
        if not account_key:
            return None
        for configured_account, identity in self.catalog.social_accounts.items():
            if _normalize_account_key(configured_account) == account_key and isinstance(identity, SourceIdentity):
                return identity
        return None


def profile_source(
    source: object,
    claim: object | None = None,
    *,
    claim_context: object | None = None,
    catalog: SourceQualityCatalog | None = None,
) -> SourceProfile:
    """Convenience function for one-off source profiling."""

    return SourceQualityService(catalog).profile(source, claim, claim_context=claim_context)


def _text_field(value: object, name: str) -> str | None:
    try:
        raw = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
    except Exception:
        return None
    if not isinstance(raw, str):
        return None
    normalized = raw.strip()
    return normalized or None


def _domain_from_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"}:
            return None
        return _normalize_domain(parsed.hostname)
    except (TypeError, ValueError):
        return None


def _normalize_domain(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if not candidate:
        return None
    if "://" in candidate:
        try:
            candidate = urlparse(candidate).hostname or ""
        except (TypeError, ValueError):
            return None
    else:
        candidate = candidate.split("/", 1)[0].split(":", 1)[0]
    candidate = candidate.rstrip(".")
    if candidate.startswith("www."):
        candidate = candidate[4:]
    return candidate or None


def _social_platform(domain: str | None) -> str | None:
    if not domain:
        return None
    for platform_domain, platform in _SOCIAL_PLATFORMS.items():
        if domain == platform_domain or domain.endswith(f".{platform_domain}"):
            return platform
    return None


def _social_account_key(domain: str | None, url: str | None) -> str | None:
    if not domain or not url:
        return None
    try:
        path_parts = [part.lower() for part in urlparse(url).path.split("/") if part]
    except (TypeError, ValueError):
        return None
    if not path_parts:
        return None
    if domain == "youtube.com" and path_parts[0] in {"channel", "c", "user"} and len(path_parts) > 1:
        return f"{domain}/{path_parts[0]}/{path_parts[1]}"
    return f"{domain}/{path_parts[0]}"


def _normalize_account_key(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    url = raw if "://" in raw else f"https://{raw}"
    domain = _domain_from_url(url)
    return _social_account_key(domain, url)


def _looks_like_commentary(title: str | None, snippet: str | None) -> bool:
    text = " ".join(part for part in (title, snippet) if part).lower()
    return any(marker in text for marker in _COMMENTARY_MARKERS)


def _claim_type_value(claim_context: object | None) -> str | None:
    if claim_context is None:
        return None
    try:
        raw = (
            claim_context.get("claim_type")
            if isinstance(claim_context, Mapping)
            else getattr(claim_context, "claim_type", None)
        )
    except Exception:
        return None
    value = getattr(raw, "value", raw)
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def _owner_matches_claim(identity: SourceIdentity, claim_context: object | None) -> bool:
    if claim_context is None:
        return False
    fields = [_text_field(claim_context, "subject"), _text_field(claim_context, "factual_statement")]
    if isinstance(claim_context, str):
        fields.append(claim_context)
    else:
        fields.append(_text_field(claim_context, "text"))
    identifiers = (identity.owner, *identity.aliases)
    normalized_fields = [_normalize_match_text(item) for item in fields if item]
    for identifier in identifiers:
        normalized_identifier = _normalize_match_text(identifier)
        if len(normalized_identifier.replace(" ", "")) < 3:
            continue
        needle = f" {normalized_identifier} "
        if any(needle in f" {field} " for field in normalized_fields):
            return True
    return False


def _normalize_match_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


source_quality = SourceQualityService()


__all__ = [
    "DEFAULT_SOURCE_QUALITY_CATALOG",
    "SourceIdentity",
    "SourceQualityCatalog",
    "SourceQualityService",
    "profile_source",
    "source_quality",
]
