"""Transparent claim-level evidence weighting and aggregation for Phase 3E.

The service does not make a truth verdict.  It only preserves the separately
weighted support and contradiction available for each candidate claim.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from math import prod
from typing import Iterable
from urllib.parse import urlparse, urlunparse

from backend.schemas import (
    AuthorityLevel,
    Claim,
    EvidenceAggregateState,
    EvidenceIndependence,
    EvidenceItem,
    EvidenceQuality,
    EvidenceStrength,
    EvidenceStance,
    EvidenceSummary,
    Freshness,
    NumericalStatus,
    SignalStatus,
)


@dataclass(frozen=True)
class EvidenceAggregationConfig:
    """Inspectable bounded defaults for the Phase 3E evidence assessment.

    They encode ordering between already-explained evidence dimensions rather
    than universal trust scores for publishers.  Later evaluation can supply a
    replacement configuration without changing response schemas.
    """

    authority_scores: dict[AuthorityLevel, float] = field(default_factory=lambda: {
        AuthorityLevel.VERY_HIGH: 1.0,
        AuthorityLevel.HIGH: 0.75,
        AuthorityLevel.MEDIUM: 0.50,
        AuthorityLevel.LOW: 0.25,
        AuthorityLevel.VERY_LOW: 0.10,
        AuthorityLevel.UNKNOWN: 0.20,
    })
    relevance_scores: dict[str, float] = field(default_factory=lambda: {
        "high": 1.0,
        "medium": 0.60,
        "low": 0.25,
        "not_assessed": 0.20,
    })
    freshness_scores: dict[Freshness, float] = field(default_factory=lambda: {
        Freshness.CURRENT: 1.0,
        Freshness.RECENT: 0.75,
        Freshness.STALE: 0.40,
        Freshness.UNKNOWN: 0.50,
    })
    independence_scores: dict[EvidenceIndependence, float] = field(default_factory=lambda: {
        EvidenceIndependence.INDEPENDENT: 1.0,
        EvidenceIndependence.SAME_GROUP: 0.50,
        EvidenceIndependence.POSSIBLE_COPY: 0.40,
        EvidenceIndependence.UNKNOWN: 0.25,
    })
    weak_threshold: float = 0.15
    moderate_threshold: float = 0.40
    strong_threshold: float = 0.65
    very_strong_threshold: float = 0.85
    conflict_threshold: float = 0.40

    def __post_init__(self) -> None:
        if not 0 <= self.weak_threshold <= self.moderate_threshold <= self.strong_threshold <= self.very_strong_threshold <= 1:
            raise ValueError("Evidence aggregation thresholds must be ordered values from 0 through 1.")
        if not 0 <= self.conflict_threshold <= 1:
            raise ValueError("The conflict threshold must be between 0 and 1.")
        mappings = (self.authority_scores, self.relevance_scores, self.freshness_scores, self.independence_scores)
        if any(any(score < 0 or score > 1 for score in mapping.values()) for mapping in mappings):
            raise ValueError("Evidence component scores must be bounded from 0 through 1.")


def _normalised_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value.strip())
    if not parsed.scheme or not parsed.netloc:
        return None
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/") or "/"
    return urlunparse(("https", host, path, "", "", ""))


def _domain(value: str | None) -> str | None:
    normalised = _normalised_url(value)
    return urlparse(normalised).netloc if normalised else None


def _normalised_text(value: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _snippet_fingerprint(value: str | None) -> str | None:
    normalised = _normalised_text(value)
    if len(normalised.split()) < 20:
        return None
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def _attribution(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(
        r"\b(?:according to|reported by|via|source:)\s+([A-Za-z][A-Za-z0-9 .'-]{2,50})",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    name = _normalised_text(match.group(1)).strip()
    return name if len(name.split()) <= 6 else None


def _copy_groups(items: list[EvidenceItem]) -> dict[int, str]:
    """Return only cross-source, obvious copy group keys."""
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        fingerprint = _snippet_fingerprint(item.source_snippet)
        if fingerprint:
            groups.setdefault(f"copy:{fingerprint}", []).append(index)
        attribution = _attribution(item.source_snippet)
        if attribution:
            groups.setdefault(f"attribution:{attribution}", []).append(index)
    result: dict[int, str] = {}
    for key, indexes in groups.items():
        domains = {_domain(items[index].source_url) for index in indexes}
        if len(indexes) > 1 and len(domains - {None}) > 1:
            for index in indexes:
                result[index] = key
    return result


def _set_independence(items: list[EvidenceItem]) -> None:
    """Attach conservative groups to evidence for one claim at a time."""
    copies = _copy_groups(items)
    url_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for item in items:
        normalised_url = _normalised_url(item.source_url)
        domain = _domain(item.source_url)
        if normalised_url:
            url_counts[normalised_url] = url_counts.get(normalised_url, 0) + 1
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    for index, item in enumerate(items):
        if item.evidence_quality is None:
            continue
        normalised_url = _normalised_url(item.source_url)
        domain = _domain(item.source_url)
        if index in copies:
            independence = EvidenceIndependence.POSSIBLE_COPY
            group = copies[index]
        elif normalised_url and url_counts.get(normalised_url, 0) > 1:
            independence = EvidenceIndependence.SAME_GROUP
            group = f"url:{normalised_url}"
        elif domain and domain_counts.get(domain, 0) > 1:
            independence = EvidenceIndependence.SAME_GROUP
            group = f"domain:{domain}"
        elif domain:
            independence = EvidenceIndependence.INDEPENDENT
            group = f"domain:{domain}"
        else:
            independence = EvidenceIndependence.UNKNOWN
            group = None
        item.evidence_quality.independence = independence
        item.evidence_quality.independence_group = group


def _nli_confidence(item: EvidenceItem) -> float | None:
    signal = item.nli_signal
    if not signal or signal.status != SignalStatus.COMPLETE:
        return None
    return signal.selected_confidence


def _numeric_reason(item: EvidenceItem) -> str | None:
    check = item.evidence_quality.numerical_check if item.evidence_quality else None
    if check and check.status in {NumericalStatus.MISMATCH, NumericalStatus.UNIT_MISMATCH}:
        return "The source's numerical value differs, so it cannot add support for this numeric claim."
    return None


def _weight(item: EvidenceItem, config: EvidenceAggregationConfig) -> tuple[float | None, dict[str, float | None], list[str]]:
    quality = item.evidence_quality
    if quality is None:
        return None, {}, ["Evidence quality could not be assessed for this source candidate."]
    nli_score = _nli_confidence(item)
    component_scores: dict[str, float | None] = {
        "authority": config.authority_scores[quality.source_profile.authority_level],
        "relevance": config.relevance_scores[quality.match.relevance.value],
        "nli_confidence": nli_score,
        "freshness": config.freshness_scores[quality.freshness],
        "independence": config.independence_scores[quality.independence],
    }
    check = quality.numerical_check
    if check is None:
        component_scores["numerical_consistency"] = None
    elif check.status == NumericalStatus.EXACT_MATCH:
        component_scores["numerical_consistency"] = 1.0
    elif check.status == NumericalStatus.WITHIN_TOLERANCE:
        component_scores["numerical_consistency"] = 0.75
    elif check.status in {NumericalStatus.MISMATCH, NumericalStatus.UNIT_MISMATCH}:
        component_scores["numerical_consistency"] = 0.0
    else:
        component_scores["numerical_consistency"] = None

    if nli_score is None:
        return None, component_scores, [
            "Evidence weight is unavailable because NLI stance analysis did not complete.",
            "Authority and NLI confidence remain separate evidence dimensions.",
        ]
    weight = prod(score for key, score in component_scores.items() if key != "numerical_consistency" and score is not None)
    reasons = ["Authority and NLI confidence are separate components of the evidence weight."]
    numeric_reason = _numeric_reason(item)
    if numeric_reason:
        reasons.append(numeric_reason)
    if quality.independence == EvidenceIndependence.POSSIBLE_COPY:
        reasons.append("Multiple sources appear to use the same underlying wording.")
    elif quality.independence == EvidenceIndependence.SAME_GROUP:
        reasons.append("This source shares an independence group with another source candidate.")
    elif quality.independence == EvidenceIndependence.UNKNOWN:
        reasons.append("Source independence could not be established from available metadata.")
    return round(min(1.0, max(0.0, weight)), 4), component_scores, reasons


def _strength(weight: float, config: EvidenceAggregationConfig) -> EvidenceStrength:
    if weight >= config.very_strong_threshold:
        return EvidenceStrength.VERY_STRONG
    if weight >= config.strong_threshold:
        return EvidenceStrength.STRONG
    if weight >= config.moderate_threshold:
        return EvidenceStrength.MODERATE
    if weight >= config.weak_threshold:
        return EvidenceStrength.WEAK
    return EvidenceStrength.INSUFFICIENT


def _state(
    support: float,
    contradiction: float,
    config: EvidenceAggregationConfig,
) -> EvidenceAggregateState:
    if support >= config.conflict_threshold and contradiction >= config.conflict_threshold:
        return EvidenceAggregateState.BALANCED_OR_CONFLICTING
    strength = _strength(max(support, contradiction), config)
    if strength == EvidenceStrength.INSUFFICIENT:
        return EvidenceAggregateState.INSUFFICIENT
    direction = "support" if support >= contradiction else "contradiction"
    return EvidenceAggregateState(f"{strength.value}_{direction}")


def _group_weight(items: Iterable[EvidenceItem], *, support: bool) -> tuple[float, int]:
    groups: dict[str, float] = {}
    for index, item in enumerate(items):
        quality = item.evidence_quality
        if not quality or quality.evidence_weight is None:
            continue
        if support and item.stance != EvidenceStance.SUPPORTS:
            continue
        if not support and item.stance != EvidenceStance.CONTRADICTS:
            continue
        check = quality.numerical_check
        if support and check and check.status in {NumericalStatus.MISMATCH, NumericalStatus.UNIT_MISMATCH}:
            continue
        group = quality.independence_group
        if quality.independence == EvidenceIndependence.UNKNOWN or group is None:
            # An unknown relationship must not be counted as an independent
            # confirmation. Its small score remains visible on the item.
            continue
        groups[group] = max(groups.get(group, 0.0), quality.evidence_weight)
    combined = 1 - prod(1 - weight for weight in groups.values()) if groups else 0.0
    return round(min(1.0, max(0.0, combined)), 4), len(groups)


class EvidenceAggregationService:
    def __init__(self, config: EvidenceAggregationConfig | None = None) -> None:
        self.config = config or EvidenceAggregationConfig()

    def prepare(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        """Assign conservative independence groups and per-item weights."""
        by_claim: dict[str | None, list[EvidenceItem]] = {}
        for item in evidence:
            by_claim.setdefault(item.claim_id, []).append(item)
        for items in by_claim.values():
            _set_independence(items)
            for item in items:
                if item.evidence_quality is None:
                    continue
                weight, component_scores, reasons = _weight(item, self.config)
                item.evidence_quality.evidence_weight = weight
                item.evidence_quality.component_scores = component_scores
                item.evidence_quality.reasons = reasons
        return evidence

    def aggregate(self, claims: list[Claim], evidence: list[EvidenceItem]) -> list[EvidenceSummary]:
        """Aggregate already-prepared evidence per claim without changing verdicts."""
        by_claim: dict[str, list[EvidenceItem]] = {}
        for item in evidence:
            if item.claim_id:
                by_claim.setdefault(item.claim_id, []).append(item)
        summaries: list[EvidenceSummary] = []
        for claim in claims:
            items = by_claim.get(claim.claim_id, [])
            support, support_groups = _group_weight(items, support=True)
            contradiction, contradiction_groups = _group_weight(items, support=False)
            state = _state(support, contradiction, self.config)
            if state == EvidenceAggregateState.BALANCED_OR_CONFLICTING:
                reason = "Supporting and contradicting evidence both meet the configured conflict threshold."
            elif state == EvidenceAggregateState.INSUFFICIENT:
                reason = "No completed, sufficiently matched NLI evidence was available for this claim."
            elif support >= contradiction:
                reason = f"Supporting evidence was aggregated from {support_groups} source independence group(s)."
            else:
                reason = f"Contradicting evidence was aggregated from {contradiction_groups} source independence group(s)."
            summaries.append(EvidenceSummary(
                claim_id=claim.claim_id,
                support_strength=_strength(support, self.config),
                contradiction_strength=_strength(contradiction, self.config),
                overall_strength=state,
                support_weight=support,
                contradiction_weight=contradiction,
                independent_support_groups=support_groups,
                independent_contradiction_groups=contradiction_groups,
                reason=reason,
            ))
        return summaries


evidence_aggregation = EvidenceAggregationService()
