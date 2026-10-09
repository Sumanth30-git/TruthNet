from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.dataset1_score import aggregate_scores, score_dataset


TEXT = "Acme launched Product Y in Bengaluru today."


def _span(fragment: str) -> dict[str, object]:
    start = TEXT.index(fragment)
    return {"start": start, "end": start + len(fragment), "text": fragment}


def _gold() -> dict:
    whole = _span(TEXT)
    return {
        "document_id": "doc-1", "normalized_text": TEXT,
        "abstentions": [],
        "claims": [{
            "claim_id": "claim-1", "source_span": whole, "original_text": TEXT,
            "claim_type": "product_announcement", "is_atomic": True, "is_verifiable_as_textual_claim": True,
            "entities": [
                {"entity_type": "organization", "source_span": _span("Acme")},
                {"entity_type": "product", "source_span": _span("Product Y")},
            ],
            "relations": [{
                "relation_type": "launch", "predicate": "launched", "subject_span": _span("Acme"),
                "object_span": _span("Product Y"), "support_spans": [whole], "negated": False,
                "modality": None, "qualifiers": [],
            }],
            "temporal_annotations": [{"source_span": _span("today"), "value": "relative_today", "normalization": "relative"}],
            "quantity_annotations": [],
            "location_annotations": [{"source_span": _span("Bengaluru"), "text": "Bengaluru", "normalized_name": "Bengaluru"}],
            "modifiers": [], "negation": None, "modality": None, "non_claim_spans": [],
        }],
    }


def test_exact_supported_structures_score_perfectly():
    gold = _gold()
    result = score_dataset(gold, deepcopy(gold))

    assert result["errors"] == []
    assert result["structure_precision"] == result["structure_recall"] == result["structure_f1"] == 1.0
    assert result["unsupported_structure_rate"] == 0.0
    assert result["relation"]["tp"] == 1


def test_unsupported_and_partial_structure_are_counted_not_rewarded():
    gold = _gold()
    predicted = deepcopy(gold)
    predicted["claims"][0]["entities"][0]["entity_type"] = "person"

    result = score_dataset(gold, predicted)

    assert result["entity"]["fp"] == 1
    assert result["entity"]["fn"] == 1
    assert result["entity"]["partial"] == 1
    assert result["unsupported_predicted_structures"] >= 1
    assert result["unsupported_structure_rate"] > 0


def test_abstention_and_zero_predictions_do_not_receive_usr_zero():
    result = score_dataset(_gold(), {
        "document_id": "doc-1", "normalized_text": TEXT, "claims": [],
        "abstentions": [{"scope": "document", "reason": "insufficient_text", "claim_id": None}],
    })

    assert result["asserted_predicted_structures"] == 0
    assert result["unsupported_structure_rate"] is None
    assert result["structure_precision"] == result["structure_recall"] == result["structure_f1"] == 0.0
    assert result["abstention_rate"] == 1.0
    assert result["scorer_abstained"] is True


def test_malformed_prediction_is_reported_and_scored_conservatively():
    malformed = deepcopy(_gold())
    malformed["claims"][0]["source_span"]["end"] = len(TEXT) + 1

    result = score_dataset(_gold(), malformed)

    assert result["errors"]
    assert result["malformed_prediction_count"] == 1
    assert result["abstention_rate"] == 0.0


def test_duplicate_predictions_increase_assertions_and_usr():
    gold = _gold()
    predicted = deepcopy(gold)
    predicted["claims"].append(deepcopy(predicted["claims"][0]))
    predicted["claims"][1]["claim_id"] = "claim-duplicate"

    result = score_dataset(gold, predicted)

    assert result["asserted_predicted_structures"] > result["claim_segmentation"]["tp"]
    assert result["unsupported_predicted_structures"] > 0
    assert result["unsupported_structure_rate"] > 0


@pytest.mark.parametrize("field,value", [
    ("entities", ["bad"]),
    ("relations", [{}]),
    ("temporal_annotations", [{"source_span": _span("today"), "value": "today"}]),
    ("quantity_annotations", [{"source_span": _span("today"), "value": "many", "unit": None, "normalized_unit": None}]),
    ("location_annotations", [{"source_span": _span("Bengaluru"), "text": "Bengaluru"}]),
    ("modifiers", [{"source_span": _span("today"), "modifier_type": "invalid", "value": "very"}]),
    ("negation", {"source_span": _span("today"), "value": "false"}),
    ("modality", {"source_span": _span("today"), "value": "invalid"}),
    ("non_claim_spans", [{"source_span": _span("today"), "reason": "invalid"}]),
])
def test_every_nested_malformed_prediction_is_conservative_and_never_throws(field, value):
    predicted = deepcopy(_gold())
    predicted["claims"][0][field] = value

    result = score_dataset(_gold(), predicted)

    assert result["malformed_prediction_count"] == 1
    assert result["errors"]


def test_explicit_outcomes_and_corpus_abstention_are_reported_separately():
    exact = score_dataset(_gold(), deepcopy(_gold()))
    partial_prediction = deepcopy(_gold())
    partial_prediction["claims"][0]["entities"][0]["entity_type"] = "person"
    partial = score_dataset(_gold(), partial_prediction)
    abstained = score_dataset(_gold(), {
        "document_id": "doc-1", "normalized_text": TEXT, "claims": [],
        "abstentions": [{"scope": "document", "reason": "insufficient_text", "claim_id": None}],
    })

    assert exact["outcomes"]["EXACT_MATCH"] > 0
    assert partial["outcomes"]["PARTIAL_BOUNDARY_MATCH"] > 0
    assert partial["outcomes"]["UNSUPPORTED"] == 0
    corpus = aggregate_scores([exact, partial, abstained])
    assert corpus["corpus_abstention_rate"] == pytest.approx(1 / 3)
    assert corpus["boundary_tolerant"]["partial_boundary_matches"] > 0
