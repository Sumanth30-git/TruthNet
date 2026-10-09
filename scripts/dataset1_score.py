"""Dataset 1-only Unsupported Structure Rate scorer.

Predictions are validated without throwing. Partial/boundary matches are
counted but never treated as supported. This module is not imported by the
TruthNet production runtime.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:  # Supports both ``python -m`` and direct script execution.
    from scripts.dataset1_common import (
        ABSTENTION_REASONS, ABSTENTION_SCOPES, CLAIM_TYPES, ENTITY_TYPES,
        MODALITIES, MODIFIER_TYPES, NON_CLAIM_REASONS, SCORER_VERSION,
        TEMPORAL_NORMALIZATIONS, ValidationIssue, _abstention_issues,
        _span_issues,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI use
    from dataset1_common import (
        ABSTENTION_REASONS, ABSTENTION_SCOPES, CLAIM_TYPES, ENTITY_TYPES,
        MODALITIES, MODIFIER_TYPES, NON_CLAIM_REASONS, SCORER_VERSION,
        TEMPORAL_NORMALIZATIONS, ValidationIssue, _abstention_issues,
        _span_issues,
    )

STRUCTURE_KINDS = (
    "claim_segmentation",
    "claim_type",
    "atomicity",
    "verifiability",
    "entity",
    "relation",
    "temporal",
    "quantity",
    "location",
    "modifier",
    "negation",
    "modality",
    "non_claim",
)


@dataclass(frozen=True)
class _Structure:
    kind: str
    span: tuple[int, int, str] | None
    key: tuple[Any, ...]


def _issue_dict(issue: ValidationIssue) -> dict[str, str]:
    return {"code": issue.code, "message": issue.message, "path": issue.path}


def _span_key(span: object) -> tuple[int, int, str] | None:
    if not isinstance(span, Mapping):
        return None
    start, end, text = span.get("start"), span.get("end"), span.get("text")
    if not isinstance(start, int) or not isinstance(end, int) or not isinstance(text, str):
        return None
    return (start, end, text)


def _overlaps(left: tuple[int, int, str] | None, right: tuple[int, int, str] | None) -> bool:
    return bool(left and right and left[0] < right[1] and right[0] < left[1])


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _covering_span(*spans: object) -> tuple[int, int, str] | None:
    keys = [item for item in (_span_key(span) for span in spans) if item is not None]
    if not keys:
        return None
    start = min(item[0] for item in keys)
    end = max(item[1] for item in keys)
    return (start, end, "")


def _prediction_issues(predicted: Mapping[str, Any], text: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if "claims" not in predicted or "abstentions" not in predicted:
        return [ValidationIssue("missing_field", "predictions require claims and canonical abstentions", "")]
    claims = predicted["claims"]
    if not isinstance(claims, list):
        return [ValidationIssue("invalid_claims", "claims must be a list", "claims")]
    claim_ids: set[str] = set()
    for index, claim in enumerate(claims):
        path = f"claims[{index}]"
        if not isinstance(claim, Mapping):
            issues.append(ValidationIssue("invalid_claim", "claim must be an object", path))
            continue
        allowed_claim = {
            "claim_id", "source_span", "original_source_span", "original_text", "claim_type", "is_atomic",
            "is_verifiable_as_textual_claim", "entities", "relations",
            "temporal_annotations", "quantity_annotations", "location_annotations",
            "modifiers", "negation", "modality", "non_claim_spans",
        }
        extras = set(claim) - allowed_claim
        if extras:
            issues.append(ValidationIssue("unexpected_field", "claim has unsupported fields", path))
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id or claim_id in claim_ids:
            issues.append(ValidationIssue("invalid_claim", "claim_id must be unique and non-empty", f"{path}.claim_id"))
        else:
            claim_ids.add(claim_id)
        issues.extend(_span_issues(claim.get("source_span"), text, f"{path}.source_span"))
        if claim.get("claim_type") not in CLAIM_TYPES:
            issues.append(ValidationIssue("invalid_enum", "invalid ClaimType", f"{path}.claim_type"))
        for key in ("is_atomic", "is_verifiable_as_textual_claim"):
            if not isinstance(claim.get(key), (bool, type(None))):
                issues.append(ValidationIssue("invalid_claim", f"{key} must be boolean or null", f"{path}.{key}"))
        entities = claim.get("entities", [])
        if isinstance(entities, list):
            for entity_index, entity in enumerate(entities):
                item = f"{path}.entities[{entity_index}]"
                if isinstance(entity, Mapping):
                    if set(entity) != {"entity_type", "source_span"} or entity.get("entity_type") not in ENTITY_TYPES:
                        issues.append(ValidationIssue("invalid_entity", "entity requires only a valid entity_type and source_span", item))
                    issues.extend(_span_issues(entity.get("source_span"), text, f"{item}.source_span"))
                else:
                    issues.append(ValidationIssue("invalid_structure", "entity must be an object", item))
        elif entities is not None:
            issues.append(ValidationIssue("invalid_structure", "entities must be a list", f"{path}.entities"))
        relations = claim.get("relations", [])
        if isinstance(relations, list):
            for relation_index, relation in enumerate(relations):
                item = f"{path}.relations[{relation_index}]"
                if not isinstance(relation, Mapping):
                    issues.append(ValidationIssue("invalid_structure", "relation must be an object", item))
                    continue
                allowed_relation = {"relation_type", "predicate", "subject_span", "object_span", "support_spans", "negated", "modality", "qualifiers"}
                if set(relation) - allowed_relation or not isinstance(relation.get("relation_type"), str) or not relation.get("relation_type") or not isinstance(relation.get("predicate"), str) or not relation.get("predicate"):
                    issues.append(ValidationIssue("invalid_relation", "relation has invalid or missing required fields", item))
                issues.extend(_span_issues(relation.get("subject_span"), text, f"{item}.subject_span"))
                issues.extend(_span_issues(relation.get("object_span"), text, f"{item}.object_span"))
                supports = relation.get("support_spans", [])
                if isinstance(supports, list) and supports:
                    for support_index, span in enumerate(supports):
                        issues.extend(_span_issues(span, text, f"{item}.support_spans[{support_index}]"))
                else:
                    issues.append(ValidationIssue("missing_support", "relation requires support_spans", item))
                if not isinstance(relation.get("negated", False), (bool, type(None))) or relation.get("modality") not in (None, *MODALITIES) or not isinstance(relation.get("qualifiers", []), list) or not all(isinstance(value, str) for value in relation.get("qualifiers", [])):
                    issues.append(ValidationIssue("invalid_relation", "relation negation, modality, or qualifiers are invalid", item))
        elif relations is not None:
            issues.append(ValidationIssue("invalid_structure", "relations must be a list", f"{path}.relations"))
        controlled = {
            "temporal_annotations": ({"source_span", "value", "normalization"}, lambda value: isinstance(value.get("value"), str) and bool(value.get("value")) and value.get("normalization") in TEMPORAL_NORMALIZATIONS),
            "quantity_annotations": ({"source_span", "value", "unit", "normalized_unit"}, lambda value: isinstance(value.get("value"), (int, float)) and not isinstance(value.get("value"), bool) and isinstance(value.get("unit"), (str, type(None))) and isinstance(value.get("normalized_unit"), (str, type(None)))),
            "location_annotations": ({"source_span", "text", "normalized_name"}, lambda value: isinstance(value.get("text"), str) and bool(value.get("text")) and isinstance(value.get("normalized_name"), (str, type(None)))),
            "modifiers": ({"source_span", "modifier_type", "value"}, lambda value: value.get("modifier_type") in MODIFIER_TYPES and isinstance(value.get("value"), str) and bool(value.get("value"))),
            "non_claim_spans": ({"source_span", "reason"}, lambda value: value.get("reason") in NON_CLAIM_REASONS),
        }
        for key, (required, valid) in controlled.items():
            values = claim.get(key, [])
            if isinstance(values, list):
                for value_index, value in enumerate(values):
                    item = f"{path}.{key}[{value_index}]"
                    if isinstance(value, Mapping):
                        if set(value) != required or not valid(value):
                            issues.append(ValidationIssue("invalid_structure", f"{key} has invalid controlled fields", item))
                        issues.extend(_span_issues(value.get("source_span"), text, f"{item}.source_span"))
                    else:
                        issues.append(ValidationIssue("invalid_structure", "structure must be an object", item))
            elif values is not None:
                issues.append(ValidationIssue("invalid_structure", f"{key} must be a list", f"{path}.{key}"))
        for key in ("negation", "modality"):
            value = claim.get(key)
            if value is not None and not isinstance(value, Mapping):
                issues.append(ValidationIssue("invalid_structure", f"{key} must be an object or null", f"{path}.{key}"))
            elif isinstance(value, Mapping):
                valid = (set(value) == {"source_span", "value"} and isinstance(value.get("value"), bool)) if key == "negation" else (set(value) == {"source_span", "value"} and value.get("value") in MODALITIES)
                if not valid:
                    issues.append(ValidationIssue("invalid_structure", f"{key} has invalid controlled fields", f"{path}.{key}"))
                issues.extend(_span_issues(value.get("source_span"), text, f"{path}.{key}.source_span"))
    for index, item in enumerate(predicted["abstentions"] if isinstance(predicted["abstentions"], list) else []):
        if not isinstance(item, Mapping) or set(item) - {"scope", "reason", "claim_id"}:
            issues.append(ValidationIssue("invalid_abstention", "abstention has unsupported fields", f"abstentions[{index}]"))
    issues.extend(_abstention_issues(predicted.get("abstentions"), claim_ids))
    return issues


def _extract(record: Mapping[str, Any]) -> list[_Structure]:
    structures: list[_Structure] = []
    claims = record.get("claims")
    if not isinstance(claims, list):
        return structures
    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        span = _span_key(claim.get("source_span"))
        structures.append(_Structure("claim_segmentation", span, ("claim_segmentation", span)))
        if claim.get("claim_type") is not None:
            structures.append(_Structure("claim_type", span, ("claim_type", span, claim.get("claim_type"))))
        if claim.get("is_atomic") is not None:
            structures.append(_Structure("atomicity", span, ("atomicity", span, claim.get("is_atomic"))))
        if claim.get("is_verifiable_as_textual_claim") is not None:
            structures.append(_Structure("verifiability", span, ("verifiability", span, claim.get("is_verifiable_as_textual_claim"))))
        for entity in claim.get("entities") or []:
            if not isinstance(entity, Mapping):
                continue
            entity_span = _span_key(entity.get("source_span"))
            structures.append(_Structure("entity", entity_span, ("entity", entity.get("entity_type"), entity_span)))
        for relation in claim.get("relations") or []:
            if not isinstance(relation, Mapping):
                continue
            supports = relation.get("support_spans") or []
            support_keys = tuple(sorted(key for key in (_span_key(span) for span in supports) if key is not None))
            covering = _covering_span(relation.get("subject_span"), relation.get("object_span"), *supports)
            structures.append(_Structure("relation", covering, (
                "relation",
                relation.get("relation_type"),
                relation.get("predicate"),
                _span_key(relation.get("subject_span")),
                _span_key(relation.get("object_span")),
                support_keys,
                relation.get("negated", False),
                relation.get("modality"),
                tuple(sorted(str(item) for item in relation.get("qualifiers") or [])),
            )))
        kind_fields = (
            ("temporal_annotations", "temporal"),
            ("quantity_annotations", "quantity"),
            ("location_annotations", "location"),
            ("modifiers", "modifier"),
            ("non_claim_spans", "non_claim"),
        )
        for field, kind in kind_fields:
            for value in claim.get(field) or []:
                if not isinstance(value, Mapping):
                    continue
                item_span = _span_key(value.get("source_span"))
                payload = tuple(sorted((key, _freeze(item)) for key, item in value.items() if key != "source_span"))
                structures.append(_Structure(kind, item_span, (kind, item_span, payload)))
        for field, kind in (("negation", "negation"), ("modality", "modality")):
            value = claim.get(field)
            if not isinstance(value, Mapping):
                continue
            item_span = _span_key(value.get("source_span"))
            payload = tuple(sorted((key, _freeze(item)) for key, item in value.items() if key != "source_span"))
            structures.append(_Structure(kind, item_span, (kind, item_span, payload)))
    return structures


def _match(gold: list[_Structure], predicted: list[_Structure]) -> dict[str, int]:
    gold_used = [False] * len(gold)
    predicted_used = [False] * len(predicted)
    true_positives = 0
    for predicted_index, predicted_item in enumerate(predicted):
        for gold_index, gold_item in enumerate(gold):
            if gold_used[gold_index] or predicted_item.key != gold_item.key:
                continue
            gold_used[gold_index] = True
            predicted_used[predicted_index] = True
            true_positives += 1
            break
    partial = 0
    for predicted_index, predicted_item in enumerate(predicted):
        if predicted_used[predicted_index]:
            continue
        if any((not used) and item.kind == predicted_item.kind and _overlaps(predicted_item.span, item.span) for used, item in zip(gold_used, gold)):
            partial += 1
    return {
        "tp": true_positives,
        "fp": predicted_used.count(False),
        "fn": gold_used.count(False),
        "partial": partial,
        "EXACT_MATCH": true_positives,
        "PARTIAL_BOUNDARY_MATCH": partial,
        "UNSUPPORTED": predicted_used.count(False) - partial,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _empty_metrics() -> dict[str, int]:
    return {
        "tp": 0, "fp": 0, "fn": 0, "partial": 0,
        "EXACT_MATCH": 0, "PARTIAL_BOUNDARY_MATCH": 0, "UNSUPPORTED": 0,
    }


def _conservative_result(errors: list[dict[str, str]], *, malformed: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "errors": errors,
        "malformed_prediction_count": malformed,
        "scorer_version": SCORER_VERSION,
        "structure_precision": 0.0,
        "structure_recall": 0.0,
        "structure_f1": 0.0,
        "unsupported_structure_rate": None,
        "asserted_predicted_structures": 0,
        "unsupported_predicted_structures": 0,
        "partial_boundary_predicted_structures": 0,
        "unsupported_nonboundary_predicted_structures": 0,
        "outcomes": {"EXACT_MATCH": 0, "PARTIAL_BOUNDARY_MATCH": 0, "UNSUPPORTED": 0},
        "scorer_abstained": False,
        "abstention_rate": 0.0,
    }
    for kind in STRUCTURE_KINDS:
        result[kind] = _empty_metrics()
    return result


def score_dataset(gold: Mapping[str, Any], predicted: Mapping[str, Any] | object) -> dict[str, Any]:
    try:
        if not isinstance(predicted, Mapping):
            return _conservative_result([{"code": "invalid_prediction", "message": "prediction must be an object", "path": ""}], malformed=1)
        text = predicted.get("normalized_text") if isinstance(predicted.get("normalized_text"), str) else gold.get("normalized_text", "")
        if not isinstance(text, str):
            text = ""
        issues = _prediction_issues(predicted, text)
        if issues:
            return _conservative_result([_issue_dict(item) for item in issues], malformed=1)
        gold_structures = _extract(gold)
        predicted_structures = _extract(predicted)
        overall = _match(gold_structures, predicted_structures)
        asserted = len(predicted_structures)
        unsupported = overall["fp"]
        partial = overall["PARTIAL_BOUNDARY_MATCH"]
        scorer_abstained = any(
            isinstance(item, Mapping) and item.get("scope") == "document"
            for item in predicted["abstentions"]
        )
        precision = _ratio(overall["tp"], overall["tp"] + overall["fp"])
        recall = _ratio(overall["tp"], overall["tp"] + overall["fn"])
        result: dict[str, Any] = {
            "errors": [],
            "malformed_prediction_count": 0,
            "scorer_version": SCORER_VERSION,
            "structure_precision": precision,
            "structure_recall": recall,
            "structure_f1": _f1(precision, recall),
            "unsupported_structure_rate": None if asserted == 0 else unsupported / asserted,
            "asserted_predicted_structures": asserted,
            "unsupported_predicted_structures": unsupported,
            "partial_boundary_predicted_structures": partial,
            "unsupported_nonboundary_predicted_structures": unsupported - partial,
            "outcomes": {
                "EXACT_MATCH": overall["EXACT_MATCH"],
                "PARTIAL_BOUNDARY_MATCH": partial,
                "UNSUPPORTED": overall["UNSUPPORTED"],
            },
            "scorer_abstained": scorer_abstained,
            "abstention_rate": 1.0 if scorer_abstained else 0.0,
        }
        for kind in STRUCTURE_KINDS:
            result[kind] = _match([item for item in gold_structures if item.kind == kind], [item for item in predicted_structures if item.kind == kind])
        return result
    except Exception as exc:  # Predictions must never crash evaluation.
        return _conservative_result([{"code": "scorer_error", "message": str(exc), "path": ""}], malformed=1)


def aggregate_scores(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate Dataset 1 results while preserving strict and partial outcomes."""
    total = _empty_metrics()
    for result in results:
        for kind in STRUCTURE_KINDS:
            metrics = result.get(kind, {})
            if isinstance(metrics, Mapping):
                for key in total:
                    value = metrics.get(key, 0)
                    if isinstance(value, int):
                        total[key] += value
    precision = _ratio(total["tp"], total["tp"] + total["fp"])
    recall = _ratio(total["tp"], total["tp"] + total["fn"])
    abstained = sum(1 for result in results if result.get("scorer_abstained") is True)
    return {
        "scorer_version": SCORER_VERSION,
        "document_count": len(results),
        "malformed_prediction_count": sum(int(result.get("malformed_prediction_count", 0)) for result in results),
        "exact": {"precision": precision, "recall": recall, "f1": _f1(precision, recall)},
        "boundary_tolerant": {
            "partial_boundary_matches": total["PARTIAL_BOUNDARY_MATCH"],
            "unsupported_nonboundary": total["UNSUPPORTED"],
        },
        "outcomes": {key: total[key] for key in ("EXACT_MATCH", "PARTIAL_BOUNDARY_MATCH", "UNSUPPORTED")},
        "corpus_abstention_rate": _ratio(abstained, len(results)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Dataset 1 predictions against gold without throwing.")
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--predicted", type=Path)
    args = parser.parse_args()
    if args.gold is None or args.predicted is None:
        print(json.dumps({"scorer_version": SCORER_VERSION, "status": "ready"}, sort_keys=True))
        return
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    predicted = json.loads(args.predicted.read_text(encoding="utf-8"))
    print(json.dumps(score_dataset(gold, predicted), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
