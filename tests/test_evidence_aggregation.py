from datetime import datetime, timezone

from backend.schemas import (
    AuthorityLevel,
    Claim,
    ClaimSourceRelevance,
    EvidenceAggregateState,
    EvidenceItem,
    EvidenceMatch,
    EvidenceQuality,
    EvidenceStance,
    Freshness,
    NLIEvidenceSignal,
    NLIProbabilities,
    NumericalCheck,
    NumericalStatus,
    SignalStatus,
    SourceProfile,
)
from backend.services.evidence_aggregation import EvidenceAggregationService


def _item(
    *,
    claim_id: str = "claim-1",
    stance: EvidenceStance = EvidenceStance.SUPPORTS,
    authority: AuthorityLevel = AuthorityLevel.HIGH,
    relevance: ClaimSourceRelevance = ClaimSourceRelevance.HIGH,
    confidence: float | None = 0.9,
    url: str = "https://example.org/article",
    snippet: str = "A detailed independently written source report supports this claim.",
    numerical: NumericalCheck | None = None,
) -> EvidenceItem:
    signal = None
    if confidence is not None:
        signal = NLIEvidenceSignal(
            model_id="test/nli",
            model_version="test",
            prediction="entailment" if stance == EvidenceStance.SUPPORTS else "contradiction",
            status=SignalStatus.COMPLETE,
            probabilities=NLIProbabilities(contradiction=0.05, entailment=0.9, neutral=0.05),
            selected_confidence=confidence,
        )
    return EvidenceItem(
        kind="news_source_candidate",
        summary="test evidence",
        claim_id=claim_id,
        source_url=url,
        source_snippet=snippet,
        stance=stance,
        nli_signal=signal,
        evidence_quality=EvidenceQuality(
            source_profile=SourceProfile(
                authority_level=authority,
                authority_reason="Test profile with a transparent authority level.",
            ),
            match=EvidenceMatch(relevance=relevance, reason="Test source match."),
            freshness=Freshness.CURRENT,
            numerical_check=numerical,
        ),
    )


def _summaries(items: list[EvidenceItem], claims: list[Claim] | None = None):
    service = EvidenceAggregationService()
    service.prepare(items)
    return service.aggregate(claims or [Claim(claim_id="claim-1", text="Claim")], items)


def test_one_high_authority_primary_source_can_be_very_strong_support():
    item = _item(authority=AuthorityLevel.VERY_HIGH, confidence=0.96)
    summary = _summaries([item])[0]
    assert item.evidence_quality.evidence_weight == 0.96
    assert summary.overall_strength == EvidenceAggregateState.VERY_STRONG_SUPPORT
    assert summary.independent_support_groups == 1


def test_many_weak_copied_sources_do_not_become_many_confirmations():
    text = (
        "This copied report contains exactly the same detailed wording in every location "
        "and repeats the original investigation with no independent reporting added here."
    )
    items = [
        _item(authority=AuthorityLevel.VERY_LOW, relevance=ClaimSourceRelevance.LOW, url="https://one.example/a", snippet=text),
        _item(authority=AuthorityLevel.VERY_LOW, relevance=ClaimSourceRelevance.LOW, url="https://two.example/b", snippet=text),
        _item(authority=AuthorityLevel.VERY_LOW, relevance=ClaimSourceRelevance.LOW, url="https://three.example/c", snippet=text),
    ]
    summary = _summaries(items)[0]
    assert {item.evidence_quality.independence.value for item in items} == {"possible_copy"}
    assert summary.independent_support_groups == 1
    assert summary.overall_strength == EvidenceAggregateState.INSUFFICIENT


def test_independent_supporting_sources_are_aggregated_by_group():
    items = [
        _item(url="https://one.example/a", snippet="Independent source one gives a detailed supporting report for this claim."),
        _item(url="https://two.example/b", snippet="Independent source two provides different supporting reporting for this claim."),
    ]
    summary = _summaries(items)[0]
    assert summary.independent_support_groups == 2
    assert summary.support_weight > items[0].evidence_quality.evidence_weight
    assert summary.overall_strength in {EvidenceAggregateState.VERY_STRONG_SUPPORT, EvidenceAggregateState.STRONG_SUPPORT}


def test_strong_contradiction_is_kept_separate_from_support():
    item = _item(stance=EvidenceStance.CONTRADICTS, authority=AuthorityLevel.VERY_HIGH, confidence=0.95)
    summary = _summaries([item])[0]
    assert summary.support_weight == 0
    assert summary.contradiction_weight == 0.95
    assert summary.overall_strength == EvidenceAggregateState.VERY_STRONG_CONTRADICTION


def test_short_generic_snippets_from_different_sources_are_not_possible_copy():
    items = [
        _item(url="https://one.example/a"),
        _item(url="https://two.example/b"),
    ]
    EvidenceAggregationService().prepare(items)
    assert {item.evidence_quality.independence.value for item in items} == {"independent"}


def test_balanced_conflict_keeps_both_strong_directions_visible():
    support = _item(url="https://support.example/a", confidence=0.9)
    contradiction = _item(
        stance=EvidenceStance.CONTRADICTS,
        url="https://contradiction.example/a",
        confidence=0.9,
    )
    summary = _summaries([support, contradiction])[0]
    assert summary.support_weight > 0.4 and summary.contradiction_weight > 0.4
    assert summary.overall_strength == EvidenceAggregateState.BALANCED_OR_CONFLICTING


def test_missing_completed_nli_is_insufficient_and_weight_is_not_fabricated():
    item = _item(confidence=None)
    summary = _summaries([item])[0]
    assert item.evidence_quality.evidence_weight is None
    assert summary.overall_strength == EvidenceAggregateState.INSUFFICIENT


def test_numeric_mismatch_does_not_add_nli_support_for_a_numeric_claim():
    mismatch = NumericalCheck(
        status=NumericalStatus.MISMATCH,
        claim_value=100,
        source_value=120,
        absolute_difference=20,
        relative_difference=0.2,
        tolerance=0.01,
        reason="Source reports a different numerical value.",
    )
    item = _item(numerical=mismatch)
    summary = _summaries([item])[0]
    assert item.evidence_quality.evidence_weight is not None
    assert summary.support_weight == 0
    assert summary.overall_strength == EvidenceAggregateState.INSUFFICIENT


def test_multiple_claims_are_aggregated_independently():
    items = [
        _item(claim_id="claim-1", url="https://one.example/a"),
        _item(claim_id="claim-2", stance=EvidenceStance.CONTRADICTS, url="https://two.example/a"),
    ]
    summaries = _summaries(items, [Claim(claim_id="claim-1", text="First"), Claim(claim_id="claim-2", text="Second")])
    assert [summary.claim_id for summary in summaries] == ["claim-1", "claim-2"]
    assert summaries[0].support_weight > 0 and summaries[0].contradiction_weight == 0
    assert summaries[1].support_weight == 0 and summaries[1].contradiction_weight > 0
