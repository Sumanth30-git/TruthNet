"""Read-only integrity audit for NLI evaluation gold and annotator artifacts.

Does not load models, does not write gold or prediction files, and does not
measure news-verdict accuracy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.nli_evaluation_common import (
        GROUP_FIELDS,
        LABELS,
        LABEL_SET,
        NLI_RELATION_SCHEMA_PATH,
        STORY_FIELD,
        annotator_comparable_record,
        annotation_status,
        canonical_url_is_http,
        cohen_kappa,
        collision_groups,
        ensure_repo_on_path,
        gold_relation,
        is_labeled,
        is_nonempty_string,
        labeled_class_counts,
        raw_agreement,
        read_jsonl,
        write_json_report,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from nli_evaluation_common import (
        GROUP_FIELDS,
        LABELS,
        LABEL_SET,
        NLI_RELATION_SCHEMA_PATH,
        STORY_FIELD,
        annotator_comparable_record,
        annotation_status,
        canonical_url_is_http,
        cohen_kappa,
        collision_groups,
        ensure_repo_on_path,
        gold_relation,
        is_labeled,
        is_nonempty_string,
        labeled_class_counts,
        raw_agreement,
        read_jsonl,
        write_json_report,
    )

ensure_repo_on_path()

from scripts.dataset1_schema import validate_against_schema
from scripts.evaluate_nli_relations import _validate_gold_record, _validate_prediction_record


AUDIT_VERSION = "1.0.0"
_IDENTIFIER_FIELDS = ("sample_id", *GROUP_FIELDS)


def _finding(
    findings: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    sample_ids: Sequence[str] | None = None,
) -> None:
    findings.append({
        "severity": severity,
        "code": code,
        "message": message,
        "sample_ids": list(sample_ids or []),
    })


def _records_from_entries(entries: Sequence[tuple[int, object]]) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for line, raw in entries:
        if isinstance(raw, dict):
            records.append((line, raw))
    return records


def _sample_id_issues(
    records: Sequence[tuple[int, Mapping[str, Any]]],
    findings: list[dict[str, Any]],
    *,
    source: str,
) -> dict[str, Any]:
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    missing: list[int] = []
    for line, record in records:
        sample_id = record.get("sample_id")
        if not is_nonempty_string(sample_id):
            missing.append(line)
            continue
        sample_id = str(sample_id)
        if sample_id in seen:
            duplicates.append(sample_id)
        else:
            seen[sample_id] = line
    if missing:
        _finding(
            findings,
            "error",
            "missing_sample_id",
            f"{source} has {len(missing)} record(s) without a non-empty sample_id",
        )
    if duplicates:
        _finding(
            findings,
            "error",
            "duplicate_sample_id",
            f"{source} has duplicate sample_id values: {', '.join(sorted(set(duplicates)))}",
            sorted(set(duplicates)),
        )
    return {
        "count": len(records),
        "unique_sample_ids": len(seen),
        "missing_sample_id_lines": missing,
        "duplicate_sample_ids": sorted(set(duplicates)),
    }


def _identifier_issues(records: Sequence[Mapping[str, Any]], findings: list[dict[str, Any]]) -> dict[str, Any]:
    missing: dict[str, list[str]] = {field: [] for field in _IDENTIFIER_FIELDS if field != "sample_id"}
    invalid_urls: list[str] = []
    for record in records:
        sample_id = record.get("sample_id") if is_nonempty_string(record.get("sample_id")) else None
        for field in missing:
            if not is_nonempty_string(record.get(field)):
                if sample_id:
                    missing[field].append(str(sample_id))
                else:
                    missing[field].append("<missing sample_id>")
        if "canonical_url" in record and is_nonempty_string(record.get("canonical_url")):
            if not canonical_url_is_http(record.get("canonical_url")):
                if sample_id:
                    invalid_urls.append(str(sample_id))
    for field, sample_ids in missing.items():
        if sample_ids:
            _finding(
                findings,
                "error",
                f"missing_{field}",
                f"{len(sample_ids)} gold record(s) are missing a non-empty {field}",
                sample_ids,
            )
    if invalid_urls:
        _finding(
            findings,
            "error",
            "invalid_canonical_url",
            "canonical_url must be an absolute HTTP(S) URL",
            invalid_urls,
        )
    return {
        "missing": {field: ids for field, ids in missing.items() if ids},
        "invalid_canonical_url_sample_ids": invalid_urls,
    }


def _gold_validity(
    records: Sequence[tuple[int, Mapping[str, Any]]],
    findings: list[dict[str, Any]],
    schema: Mapping[str, Any] | None,
) -> dict[str, Any]:
    schema_invalid: list[str] = []
    evaluator_invalid: list[str] = []
    for line, record in records:
        sample_id = record.get("sample_id") if is_nonempty_string(record.get("sample_id")) else f"line-{line}"
        if schema is not None:
            schema_issues = validate_against_schema(record, schema, base_dir=NLI_RELATION_SCHEMA_PATH.parent)
            if schema_issues:
                schema_invalid.append(str(sample_id))
        evaluator_errors: list[dict[str, Any]] = []
        if _validate_gold_record(record, line, evaluator_errors) is None:
            evaluator_invalid.append(str(sample_id))
            for error in evaluator_errors:
                _finding(
                    findings,
                    "error",
                    "invalid_gold_label",
                    error["message"],
                    [str(sample_id)],
                )
        elif annotation_status(record) == "unannotated":
            evaluator_invalid.append(str(sample_id))
            _finding(
                findings,
                "error",
                "unannotated_not_evaluator_gold",
                "unannotated records are schema-legal but are not valid evaluator gold",
                [str(sample_id)],
            )
    if schema_invalid:
        _finding(
            findings,
            "error",
            "schema_invalid",
            f"{len(schema_invalid)} gold record(s) do not match nli_relation.schema.json",
            schema_invalid,
        )
    labeled_missing_class = [
        str(record.get("sample_id"))
        for _, record in records
        if is_labeled(record) and (
            not isinstance(gold_relation(record), str) or gold_relation(record) not in LABEL_SET
        )
    ]
    null_required = [
        str(record.get("sample_id"))
        for _, record in records
        if annotation_status(record) in {"unresolved", "abstained"} and gold_relation(record) is not None
    ]
    return {
        "schema_invalid_sample_ids": schema_invalid,
        "evaluator_invalid_sample_ids": evaluator_invalid,
        "labeled_with_invalid_relation": labeled_missing_class,
        "excluded_with_non_null_relation": null_required,
        "valid_evaluator_gold": not schema_invalid and not evaluator_invalid,
    }


def _class_distribution(records: Sequence[Mapping[str, Any]], findings: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(annotation_status(record) or "missing" for record in records)
    labeled_counts = labeled_class_counts(records)
    missing_classes = [label for label in LABELS if labeled_counts[label] == 0]
    if missing_classes and any(is_labeled(record) for record in records):
        _finding(
            findings,
            "warning",
            "missing_labeled_class",
            "labeled gold is missing class(es): " + ", ".join(missing_classes),
        )
    return {
        "annotation_status": dict(status_counts),
        "labeled_relations": labeled_counts,
        "eligible_labeled_count": sum(labeled_counts.values()),
        "excluded_unresolved": int(status_counts.get("unresolved", 0)),
        "excluded_abstained": int(status_counts.get("abstained", 0)),
        "missing_labeled_classes": missing_classes,
    }


def _duplicate_syndication_note(collisions: Mapping[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    dup = collisions["duplicate_or_syndication_group_id"]
    story = collisions[STORY_FIELD]
    singleton_only = dup["collision_group_count"] == 0 and dup["unique_value_count"] > 0
    if singleton_only:
        _finding(
            findings,
            "warning",
            "duplicate_groups_are_singletons",
            "duplicate_or_syndication_group_id values are unique per row and do not encode copies or syndication",
        )
    return {
        "duplicate_or_syndication_group_id": {
            "collision_group_count": dup["collision_group_count"],
            "unique_value_count": dup["unique_value_count"],
            "singleton_count": dup["singleton_count"],
            "encodes_copies": not singleton_only,
        },
        "story_or_event_cluster_id": {
            "collision_group_count": story["collision_group_count"],
            "unique_value_count": story["unique_value_count"],
        },
        "note": (
            "Duplicate/syndication IDs are per-sample singletons; story clustering is the grouping key for splits."
            if singleton_only
            else "Review duplicate/syndication collisions before treating rows as independent."
        ),
    }


def _annotator_agreement(
    annotator_a: Sequence[Mapping[str, Any]] | None,
    annotator_b: Sequence[Mapping[str, Any]] | None,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    if annotator_a is None and annotator_b is None:
        return {"status": "not_supplied"}
    if annotator_a is None or annotator_b is None:
        _finding(findings, "warning", "annotator_file_missing", "only one annotator file was supplied")
        return {"status": "incomplete", "reason": "both annotator files are required for agreement"}

    by_a = {record["sample_id"]: record for record in annotator_a if is_nonempty_string(record.get("sample_id"))}
    by_b = {record["sample_id"]: record for record in annotator_b if is_nonempty_string(record.get("sample_id"))}
    shared = sorted(set(by_a) & set(by_b))
    only_a = sorted(set(by_a) - set(by_b))
    only_b = sorted(set(by_b) - set(by_a))
    labels_a = [by_a[sample_id].get("gold_relation") for sample_id in shared]
    labels_b = [by_b[sample_id].get("gold_relation") for sample_id in shared]
    disagreements = [
        {
            "sample_id": sample_id,
            "annotator_a": by_a[sample_id].get("gold_relation"),
            "annotator_b": by_b[sample_id].get("gold_relation"),
        }
        for sample_id in shared
        if by_a[sample_id].get("gold_relation") != by_b[sample_id].get("gold_relation")
    ]
    comparable_a = [annotator_comparable_record(by_a[sample_id]) for sample_id in shared]
    comparable_b = [annotator_comparable_record(by_b[sample_id]) for sample_id in shared]
    identical_except_annotator = comparable_a == comparable_b and not only_a and not only_b
    same_order = [record.get("sample_id") for record in annotator_a] == [record.get("sample_id") for record in annotator_b]
    agreement = raw_agreement(labels_a, labels_b)
    kappa = cohen_kappa(labels_a, labels_b)
    if identical_except_annotator:
        _finding(
            findings,
            "warning",
            "identical_annotator_files",
            "Annotator A and B records are identical after ignoring the annotator name. "
            f"Raw agreement is {agreement} and Cohen's kappa is {kappa}. "
            "This does not prove independent double annotation.",
            shared,
        )
    elif disagreements:
        _finding(
            findings,
            "info",
            "annotator_disagreement",
            f"{len(disagreements)} sample(s) differ between annotator files",
            [item["sample_id"] for item in disagreements],
        )
    return {
        "status": "computed",
        "paired_sample_count": len(shared),
        "only_in_annotator_a": only_a,
        "only_in_annotator_b": only_b,
        "same_sample_order": same_order,
        "identical_except_annotator_name": identical_except_annotator,
        "raw_agreement": agreement,
        "cohen_kappa": kappa,
        "disagreements": disagreements,
        "independence_supported": False if identical_except_annotator else None,
        "note": (
            "Identical labels and identical records (except annotator name) are consistent with a copied label set. "
            "Kappa of 1.0 does not prove two independent annotators."
            if identical_except_annotator
            else "Agreement is reported on the files as they are; no labels were altered."
        ),
    }


def _prediction_coverage(
    gold_records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]] | None,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    if predictions is None:
        return {"status": "not_supplied"}
    gold_by_id = {
        str(record["sample_id"]): record
        for record in gold_records
        if is_nonempty_string(record.get("sample_id"))
    }
    pred_by_id: dict[str, Mapping[str, Any]] = {}
    invalid: list[str] = []
    for index, record in enumerate(predictions, 1):
        errors: list[dict[str, Any]] = []
        sample_id = record.get("sample_id") if is_nonempty_string(record.get("sample_id")) else f"line-{index}"
        if _validate_prediction_record(record, index, errors) is None:
            invalid.append(str(sample_id))
            for error in errors:
                _finding(findings, "error", "invalid_prediction", error["message"], [str(sample_id)])
            continue
        pred_by_id[str(record["sample_id"])] = record
    labeled_ids = [
        str(record["sample_id"])
        for record in gold_records
        if is_labeled(record) and is_nonempty_string(record.get("sample_id"))
    ]
    missing = [sample_id for sample_id in labeled_ids if sample_id not in pred_by_id]
    extra = sorted(set(pred_by_id) - set(gold_by_id))
    if missing:
        _finding(
            findings,
            "warning",
            "missing_predictions_for_labeled_gold",
            f"{len(missing)} labeled gold sample(s) have no stored prediction",
            missing,
        )
    return {
        "status": "checked",
        "prediction_count": len(predictions),
        "valid_prediction_count": len(pred_by_id),
        "invalid_prediction_sample_ids": invalid,
        "missing_for_labeled_gold": missing,
        "unmatched_prediction_sample_ids": extra,
        "valid_stored_predictions": not invalid and not extra,
    }


def audit_records(
    gold_records: Sequence[Mapping[str, Any]],
    annotator_a: Sequence[Mapping[str, Any]] | None = None,
    annotator_b: Sequence[Mapping[str, Any]] | None = None,
    predictions: Sequence[Mapping[str, Any]] | None = None,
    *,
    schema: Mapping[str, Any] | None = None,
    parse_errors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for error in parse_errors:
        _finding(findings, "error", "parse_error", str(error.get("message")), [])
    gold_with_lines = [(index, record) for index, record in enumerate(gold_records, 1)]
    sample_summary = _sample_id_issues(gold_with_lines, findings, source="gold")
    identifier_summary = _identifier_issues(gold_records, findings)
    loaded_schema = schema
    if loaded_schema is None and NLI_RELATION_SCHEMA_PATH.is_file():
        loaded_schema = json.loads(NLI_RELATION_SCHEMA_PATH.read_text(encoding="utf-8"))
    validity = _gold_validity(gold_with_lines, findings, loaded_schema)
    distribution = _class_distribution(gold_records, findings)
    collisions = collision_groups(gold_records)
    duplicate_note = _duplicate_syndication_note(collisions, findings)
    agreement = _annotator_agreement(annotator_a, annotator_b, findings)
    prediction_summary = _prediction_coverage(gold_records, predictions, findings)
    errors = [item for item in findings if item["severity"] == "error"]
    return {
        "audit_version": AUDIT_VERSION,
        "audit_status": "failed" if errors else "completed",
        "valid_evaluator_input": validity["valid_evaluator_gold"] and not errors,
        "counts": {
            "gold_records": len(gold_records),
            **sample_summary,
        },
        "identifiers": identifier_summary,
        "gold_label_validity": validity,
        "class_distribution": distribution,
        "group_diagnostics": collisions,
        "duplicate_or_syndication": duplicate_note,
        "annotator_agreement": agreement,
        "predictions": prediction_summary,
        "group_leakage": {
            "status": "not_applicable_no_splits",
            "note": "No split files are assigned for this corpus; cross-split leakage is not computed.",
        },
        "findings": findings,
        "limitations": [
            "This audit inspects NLI textual-relation gold. It does not measure news-verdict accuracy.",
            "Annotator agreement on identical files is not evidence of independent annotation.",
            "Story-clustered pairs are not independent samples.",
            "Existing gold and prediction files are not modified.",
        ],
    }


def audit_files(
    gold_path: Path,
    annotator_a_path: Path | None = None,
    annotator_b_path: Path | None = None,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    gold_entries, gold_errors = read_jsonl(gold_path, "gold")
    parse_errors = list(gold_errors)
    gold_records = [record for _, record in _records_from_entries(gold_entries)]
    for line, raw in gold_entries:
        if not isinstance(raw, dict):
            parse_errors.append({
                "source": "gold",
                "line": line,
                "message": "record must be a JSON object",
            })

    annotator_a = None
    annotator_b = None
    if annotator_a_path is not None:
        entries, errors = read_jsonl(annotator_a_path, "annotator_a")
        parse_errors.extend(errors)
        annotator_a = [record for _, record in _records_from_entries(entries)]
    if annotator_b_path is not None:
        entries, errors = read_jsonl(annotator_b_path, "annotator_b")
        parse_errors.extend(errors)
        annotator_b = [record for _, record in _records_from_entries(entries)]
    predictions = None
    if predictions_path is not None:
        entries, errors = read_jsonl(predictions_path, "predictions")
        parse_errors.extend(errors)
        predictions = [record for _, record in _records_from_entries(entries)]

    report = audit_records(
        gold_records,
        annotator_a,
        annotator_b,
        predictions,
        parse_errors=parse_errors,
    )
    report["inputs"] = {
        "gold": str(gold_path),
        "annotator_a": str(annotator_a_path) if annotator_a_path else None,
        "annotator_b": str(annotator_b_path) if annotator_b_path else None,
        "predictions": str(predictions_path) if predictions_path else None,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only integrity audit for NLI evaluation gold (not news-verdict accuracy)."
    )
    parser.add_argument("--gold", required=True, help="canonical gold JSONL")
    parser.add_argument("--annotator-a", help="optional annotator A JSONL")
    parser.add_argument("--annotator-b", help="optional annotator B JSONL")
    parser.add_argument("--predictions", help="optional stored predictions JSONL")
    parser.add_argument("--output", default="-", help="report JSON path, or - for stdout (default: -)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_files(
        Path(args.gold),
        Path(args.annotator_a) if args.annotator_a else None,
        Path(args.annotator_b) if args.annotator_b else None,
        Path(args.predictions) if args.predictions else None,
    )
    try:
        write_json_report(report, args.output)
    except ValueError as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot write report: {exc.strerror or exc.__class__.__name__}", file=__import__("sys").stderr)
        return 2
    return 2 if report["audit_status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
