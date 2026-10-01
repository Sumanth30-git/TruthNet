import pytest

from backend.fusion.decision import NewsVerdictDecision, decide_news_verdict
from backend.schemas import (
    Claim,
    EvidenceAggregateState,
    EvidenceAnalysisStatus,
    EvidenceSummary,
    SignalResult,
    SignalStatus,
    Uncertainty,
    Verdict,
    ContentType,
)


def _claim(claim_id: str = "claim-1") -> Claim:
    return Claim(claim_id=claim_id, text=f"{claim_id} text")


def _summary(claim_id: str, state: EvidenceAggregateState) -> EvidenceSummary:
    return EvidenceSummary(claim_id=claim_id, overall_strength=state, reason="Test summary.")


def _decision(
    *states: EvidenceAggregateState,
    **availability,
) -> NewsVerdictDecision:
    claims = [_claim(f"claim-{index}") for index in range(1, len(states) + 1)]
    summaries = [_summary(claim.claim_id, state) for claim, state in zip(claims, states)]
    return decide_news_verdict(claims, summaries, **availability)


@pytest.mark.parametrize(
    ("state", "expected_verdict", "expected_uncertainty"),
    [
        (EvidenceAggregateState.STRONG_SUPPORT, Verdict.LIKELY_AUTHENTIC, Uncertainty.MEDIUM),
        (EvidenceAggregateState.VERY_STRONG_SUPPORT, Verdict.LIKELY_AUTHENTIC, Uncertainty.MEDIUM),
        (EvidenceAggregateState.STRONG_CONTRADICTION, Verdict.LIKELY_FAKE_NEWS, Uncertainty.MEDIUM),
        (EvidenceAggregateState.VERY_STRONG_CONTRADICTION, Verdict.LIKELY_FAKE_NEWS, Uncertainty.MEDIUM),
        (EvidenceAggregateState.WEAK_SUPPORT, Verdict.INCONCLUSIVE, Uncertainty.HIGH),
        (EvidenceAggregateState.MODERATE_SUPPORT, Verdict.INCONCLUSIVE, Uncertainty.HIGH),
        (EvidenceAggregateState.WEAK_CONTRADICTION, Verdict.INCONCLUSIVE, Uncertainty.HIGH),
        (EvidenceAggregateState.MODERATE_CONTRADICTION, Verdict.INCONCLUSIVE, Uncertainty.HIGH),
        (EvidenceAggregateState.BALANCED_OR_CONFLICTING, Verdict.INCONCLUSIVE, Uncertainty.HIGH),
        (EvidenceAggregateState.INSUFFICIENT, Verdict.INCONCLUSIVE, Uncertainty.HIGH),
    ],
)
def test_single_claim_policy_uses_only_named_aggregation_states(
    state: EvidenceAggregateState,
    expected_verdict: Verdict,
    expected_uncertainty: Uncertainty,
):
    decision = _decision(state)

    assert decision.verdict == expected_verdict
    assert decision.confidence is None
    assert decision.uncertainty == expected_uncertainty
    assert decision.verdict != Verdict.LIKELY_MANIPULATED


def test_empty_summaries_are_inconclusive():
    decision = decide_news_verdict([_claim()], [])

    assert decision.verdict == Verdict.INCONCLUSIVE
    assert decision.confidence is None
    assert decision.uncertainty == Uncertainty.HIGH


def test_multiple_claims_must_all_have_same_strong_direction():
    support = _decision(
        EvidenceAggregateState.STRONG_SUPPORT,
        EvidenceAggregateState.VERY_STRONG_SUPPORT,
    )
    contradiction = _decision(
        EvidenceAggregateState.STRONG_CONTRADICTION,
        EvidenceAggregateState.VERY_STRONG_CONTRADICTION,
    )
    mixed = _decision(
        EvidenceAggregateState.STRONG_SUPPORT,
        EvidenceAggregateState.STRONG_CONTRADICTION,
    )
    supported_and_insufficient = _decision(
        EvidenceAggregateState.STRONG_SUPPORT,
        EvidenceAggregateState.INSUFFICIENT,
    )
    contradicted_and_insufficient = _decision(
        EvidenceAggregateState.STRONG_CONTRADICTION,
        EvidenceAggregateState.INSUFFICIENT,
    )

    assert support.verdict == Verdict.LIKELY_AUTHENTIC
    assert contradiction.verdict == Verdict.LIKELY_FAKE_NEWS
    assert all(
        decision.verdict == Verdict.INCONCLUSIVE
        for decision in (mixed, supported_and_insufficient, contradicted_and_insufficient)
    )
    assert all(decision.confidence is None for decision in (support, contradiction, mixed))


def test_missing_or_duplicate_claim_summary_is_inconclusive():
    claims = [_claim("claim-1"), _claim("claim-2")]
    missing = decide_news_verdict(
        claims,
        [_summary("claim-1", EvidenceAggregateState.STRONG_SUPPORT)],
    )
    duplicate = decide_news_verdict(
        [_claim()],
        [
            _summary("claim-1", EvidenceAggregateState.STRONG_SUPPORT),
            _summary("claim-1", EvidenceAggregateState.STRONG_SUPPORT),
        ],
    )

    assert missing.verdict == duplicate.verdict == Verdict.INCONCLUSIVE
    assert missing.uncertainty == duplicate.uncertainty == Uncertainty.HIGH


def _bert_signal(prediction: str) -> SignalResult:
    return SignalResult(
        model_id="jy46604790/Fake-News-Bert-Detect",
        model_version="main",
        prediction=prediction,
        raw_score=0.99,
        content_type=ContentType.TEXT,
        status=SignalStatus.COMPLETE,
    )


def test_bert_signal_is_ignored_for_final_news_verdict():
    support = _decision(
        EvidenceAggregateState.STRONG_SUPPORT,
        signals=[_bert_signal("fake_news")],
    )
    contradiction = _decision(
        EvidenceAggregateState.STRONG_CONTRADICTION,
        signals=[_bert_signal("real_news")],
    )
    bert_only = decide_news_verdict([], [], signals=[_bert_signal("fake_news")])

    assert support.verdict == Verdict.LIKELY_AUTHENTIC
    assert contradiction.verdict == Verdict.LIKELY_FAKE_NEWS
    assert bert_only.verdict == Verdict.INCONCLUSIVE
    assert all(decision.confidence is None for decision in (support, contradiction, bert_only))


@pytest.mark.parametrize(
    ("availability", "expected_uncertainty"),
    [
        ({"search_status": "failed"}, Uncertainty.NOT_ASSESSED),
        ({"evidence_analysis_status": EvidenceAnalysisStatus.PARTIAL}, Uncertainty.NOT_ASSESSED),
        ({"unavailable_claim_ids": {"claim-1"}}, Uncertainty.NOT_ASSESSED),
        ({"aggregation_succeeded": False}, Uncertainty.NOT_ASSESSED),
        ({"evidence_analysis_status": EvidenceAnalysisStatus.NO_SOURCES}, Uncertainty.HIGH),
    ],
)
def test_unavailable_processing_never_produces_a_news_verdict(availability, expected_uncertainty):
    decision = _decision(EvidenceAggregateState.STRONG_SUPPORT, **availability)

    assert decision.verdict == Verdict.INCONCLUSIVE
    assert decision.confidence is None
    assert decision.uncertainty == expected_uncertainty
