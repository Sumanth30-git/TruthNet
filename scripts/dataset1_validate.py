"""Validation API and CLI for Dataset 1 artifacts.

The CLI is intentionally useful before data exists: an empty scaffold is valid
as scaffolding, explicitly marked ``empty_scaffold``, and never a valid corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

try:  # Supports both ``python -m`` and direct script execution.
    from scripts.dataset1_common import (
        ValidationIssue, exact_duplicate_ids, read_jsonl, run_duplicate_qc,
        validate_adjudication,
        validate_claim_annotation, validate_collection_candidate, validate_collection_cross_file_integrity,
        validate_document, validate_provenance,
        validate_cross_file_integrity, validate_story_cluster_records,
        validate_split_records,
    )
    from scripts.dataset1_schema import load_schema, validate_against_schema
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI use
    from dataset1_common import (
        ValidationIssue, exact_duplicate_ids, read_jsonl, run_duplicate_qc,
        validate_adjudication,
        validate_claim_annotation, validate_collection_candidate, validate_collection_cross_file_integrity,
        validate_document, validate_provenance,
        validate_cross_file_integrity, validate_story_cluster_records,
        validate_split_records,
    )
    from dataset1_schema import load_schema, validate_against_schema


def _serialise(issues: list[ValidationIssue]) -> list[dict[str, str]]:
    return [{"code": item.code, "message": item.message, "path": item.path} for item in issues]


def validate_dataset(root: Path) -> dict[str, Any]:
    """Validate all available Dataset 1 artifacts without mutating them."""
    documents = read_jsonl(root / "documents" / "normalized.jsonl")
    provenance = read_jsonl(root / "provenance" / "sources.jsonl")
    candidates = read_jsonl(root / "provenance" / "candidates.jsonl")
    clusters = read_jsonl(root / "clusters" / "story_clusters.jsonl")
    annotator_a = read_jsonl(root / "annotations" / "annotator_a.jsonl")
    annotator_b = read_jsonl(root / "annotations" / "annotator_b.jsonl")
    annotations = read_jsonl(root / "annotations" / "adjudicated.jsonl")
    split_rows = {
        "development": read_jsonl(root / "splits" / "development.jsonl"),
        "validation": read_jsonl(root / "splits" / "validation.jsonl"),
        "frozen_test": read_jsonl(root / "splits" / "frozen_test.jsonl"),
    }
    issues: list[ValidationIssue] = []
    documents_by_id = {row.get("document_id"): row for row in documents}

    def add_schema_issues(rows: list[dict[str, Any]], schema_name: str, label: str) -> None:
        schema = load_schema(schema_name)
        for index, row in enumerate(rows):
            issues.extend(
                ValidationIssue(item.code, item.message, f"{label}[{index}].{item.path}".rstrip("."))
                for item in validate_against_schema(row, schema)
            )

    # Schema checks are part of the workflow, not advisory definitions.
    add_schema_issues(documents, "document.schema.json", "documents")
    add_schema_issues(provenance, "provenance.schema.json", "provenance")
    add_schema_issues(candidates, "collection_candidate.schema.json", "candidates")
    add_schema_issues(clusters, "story_cluster.schema.json", "clusters")
    add_schema_issues(annotator_a, "claim_annotation.schema.json", "annotator_a")
    add_schema_issues(annotator_b, "claim_annotation.schema.json", "annotator_b")
    add_schema_issues(annotations, "adjudication.schema.json", "adjudications")
    for split_name, rows in split_rows.items():
        add_schema_issues(rows, "split_assignment.schema.json", f"splits.{split_name}")

    for index, row in enumerate(documents):
        issues.extend(ValidationIssue(item.code, item.message, f"documents[{index}].{item.path}".rstrip(".")) for item in validate_document(row))
    for duplicate_id in sorted(exact_duplicate_ids(documents)):
        issues.append(ValidationIssue("duplicate_document_id", f"duplicate document ID: {duplicate_id}", "documents"))
    for index, row in enumerate(provenance):
        issues.extend(ValidationIssue(item.code, item.message, f"provenance[{index}].{item.path}".rstrip(".")) for item in validate_provenance(row))
    for index, row in enumerate(candidates):
        issues.extend(ValidationIssue(item.code, item.message, f"candidates[{index}].{item.path}".rstrip(".")) for item in validate_collection_candidate(row))
    cluster_issues = validate_story_cluster_records(documents, clusters)
    if documents:
        issues.extend(cluster_issues)
    for label, rows in (("annotator_a", annotator_a), ("annotator_b", annotator_b)):
        for index, row in enumerate(rows):
            document = documents_by_id.get(row.get("document_id"))
            issues.extend(ValidationIssue(item.code, item.message, f"{label}[{index}].{item.path}".rstrip(".")) for item in validate_claim_annotation(row, document))
    for index, row in enumerate(annotations):
        document_id = row.get("document_id")
        document = documents_by_id.get(document_id)
        issues.extend(ValidationIssue(item.code, item.message, f"annotations[{index}].{item.path}".rstrip(".")) for item in validate_adjudication(row, document))
    issues.extend(validate_cross_file_integrity(documents, provenance, annotator_a, annotator_b, annotations))
    issues.extend(validate_collection_cross_file_integrity(documents, provenance, candidates))
    issues.extend(validate_story_cluster_records(documents, clusters))
    issues.extend(validate_split_records(documents, split_rows, clusters))
    qc = run_duplicate_qc(documents, clusters, split_rows)
    if documents and qc["status"] != "pass":
        issues.append(ValidationIssue("qc_gate", f"duplicate/QC status is {qc['status']}", "qc"))
    populated = bool(documents)
    blocking_issues = [item for item in issues if item.code != "dataset_not_populated"]
    return {
        "dataset_populated": populated,
        "validation_state": "populated" if populated else "empty_scaffold",
        "valid": not blocking_issues,
        "issues": _serialise(issues),
        "qc": qc,
        "document_count": len(documents),
        "provenance_count": len(provenance),
        "candidate_count": len(candidates),
        "cluster_count": len(clusters),
        "annotator_a_count": len(annotator_a),
        "annotator_b_count": len(annotator_b),
        "adjudication_count": len(annotations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Dataset 1 artifacts without changing them.")
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/claim_understanding"))
    args = parser.parse_args()
    result = validate_dataset(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["dataset_populated"] and not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
