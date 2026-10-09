"""Deterministic freeze and verification for the Dataset 1 release.

This module is deliberately Dataset-1-only.  It records content identity from a
closed inventory of canonical artifacts; the optional freeze timestamp is
provenance metadata and is never part of that identity.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:  # Supports both ``python -m`` and direct script execution.
    from scripts.dataset1_common import (
        DATASET_ID, DATASET_VERSION, NORMALIZATION_VERSION, SCORER_VERSION,
        SOURCE_PROVENANCE_VERSION, SPLIT_VERSION, TARGET_SPLIT_COUNTS,
        read_jsonl, sha256_file, sha256_text,
    )
    from scripts.dataset1_schema import validate_against_schema
    from scripts.dataset1_validate import validate_dataset
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI use
    from dataset1_common import (
        DATASET_ID, DATASET_VERSION, NORMALIZATION_VERSION, SCORER_VERSION,
        SOURCE_PROVENANCE_VERSION, SPLIT_VERSION, TARGET_SPLIT_COUNTS,
        read_jsonl, sha256_file, sha256_text,
    )
    from dataset1_schema import validate_against_schema
    from dataset1_validate import validate_dataset


FREEZE_MANIFEST_SCHEMA_VERSION = "1.0"
MANIFEST_RELATIVE_PATH = "freeze/MANIFEST.json"
SCHEMA_NAMES = (
    "adjudication.schema.json",
    "claim_annotation.schema.json",
    "document.schema.json",
    "collection_candidate.schema.json",
    "freeze_manifest.schema.json",
    "provenance.schema.json",
    "split_assignment.schema.json",
    "story_cluster.schema.json",
)
CANONICAL_ARTIFACT_PATHS = tuple(sorted((
    "DATASET_1_SPEC.md",
    "SOURCE_POLICY.md",
    "SOURCING_WORKFLOW.md",
    "annotations/adjudicated.jsonl",
    "annotations/annotator_a.jsonl",
    "annotations/annotator_b.jsonl",
    "clusters/story_clusters.jsonl",
    "documents/normalized.jsonl",
    "provenance/sources.jsonl",
    "provenance/candidates.jsonl",
    "qc/.gitkeep",
    "splits/development.jsonl",
    "splits/frozen_test.jsonl",
    "splits/validation.jsonl",
    *(f"schemas/{name}" for name in SCHEMA_NAMES),
)))
FROZEN_REQUIRED_FIELDS = {
    "status", "dataset_id", "freeze_manifest_schema_version", "dataset_version",
    "document_count", "claim_count", "split_counts", "document_ids",
    "split_membership", "cluster_count", "cluster_integrity",
    "provenance_completeness", "annotation_completeness",
    "adjudication_completeness", "adjudication_status_counts", "qc_status",
    "schema_hashes", "artifacts", "artifact_sha256", "validation_summary",
    "contract", "content_identity", "timestamp", "required_fields",
    "annotation_version", "adjudication_version", "normalization_version",
    "source_provenance_version", "split_version", "scorer_version",
}


def _result(status: str, reasons: list[dict[str, str]], **extra: Any) -> dict[str, Any]:
    return {"status": status, "verified": status == "VERIFIED", "failure_reasons": reasons, **extra}


def _reason(code: str, message: str, path: str = "") -> dict[str, str]:
    return {"code": code, "message": message, "path": path}


def _json_canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _relative_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )


def _inventory(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    artifacts: list[dict[str, Any]] = []
    reasons: list[dict[str, str]] = []
    for relative_path in CANONICAL_ARTIFACT_PATHS:
        path = root / relative_path
        exists = path.is_file()
        entry: dict[str, Any] = {"expected": True, "relative_path": relative_path, "exists": exists}
        if exists:
            entry.update({"size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        else:
            entry.update({"size_bytes": None, "sha256": None})
            reasons.append(_reason("MISSING_ARTIFACT", "required frozen artifact is missing", relative_path))
        artifacts.append(entry)
    expected = set(CANONICAL_ARTIFACT_PATHS) | {MANIFEST_RELATIVE_PATH}
    for relative_path in _relative_files(root):
        if relative_path not in expected:
            reasons.append(_reason("UNEXPECTED_ARTIFACT", "artifact is outside the closed freeze inventory", relative_path))
    return artifacts, reasons


def _safe_validate(root: Path) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    try:
        return validate_dataset(root), []
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return None, [_reason("DATASET_INPUT_ERROR", str(exc))]
    except Exception as exc:  # Ordinary bad data must not make freeze APIs throw.
        return None, [_reason("DATASET_VALIDATION_ERROR", str(exc))]


def _configuration_reasons(root: Path) -> list[dict[str, str]]:
    """Reject malformed frozen schemas instead of merely hashing their bytes."""
    reasons: list[dict[str, str]] = []
    for name in SCHEMA_NAMES:
        relative_path = f"schemas/{name}"
        try:
            schema = json.loads((root / relative_path).read_text(encoding="utf-8"))
            if not isinstance(schema, Mapping):
                reasons.append(_reason("MALFORMED_ARTIFACT", "schema must be a JSON object", relative_path))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            reasons.append(_reason("MALFORMED_ARTIFACT", str(exc), relative_path))
    return reasons


def _read_rows(root: Path, relative_path: str) -> list[dict[str, Any]]:
    return read_jsonl(root / relative_path)


def _labels(validation: Mapping[str, Any]) -> dict[str, str]:
    codes = {item.get("code") for item in validation.get("issues", []) if isinstance(item, Mapping)}
    return {
        "cluster_integrity": "invalid" if codes & {"cluster_disagreement", "cluster_leakage", "duplicate_cluster_membership"} else ("incomplete" if codes & {"missing_canonical_clusters", "missing_cluster_membership"} else "complete"),
        "provenance_completeness": "incomplete" if codes & {"missing_provenance", "orphan_provenance", "duplicate_provenance", "provenance_source_mismatch"} else "complete",
        "annotation_completeness": "incomplete" if codes & {"incomplete_double_annotation", "annotator_independence", "unknown_annotation"} else "complete",
        "adjudication_completeness": "incomplete" if codes & {"incomplete_adjudication", "duplicate_adjudication", "adjudication_annotation_mismatch"} else "complete",
    }


def _snapshot(root: Path, validation: Mapping[str, Any], artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    documents = _read_rows(root, "documents/normalized.jsonl")
    clusters = _read_rows(root, "clusters/story_clusters.jsonl")
    adjudications = _read_rows(root, "annotations/adjudicated.jsonl")
    split_rows = {name: _read_rows(root, f"splits/{name}.jsonl") for name in TARGET_SPLIT_COUNTS}
    split_membership = {name: sorted(str(row.get("document_id", "")) for row in rows) for name, rows in split_rows.items()}
    claim_count = sum(
        len(row.get("gold_annotation", {}).get("claims", []))
        for row in adjudications if isinstance(row.get("gold_annotation"), Mapping)
    )
    status_counts: dict[str, int] = {}
    for row in adjudications:
        status = row.get("status")
        if isinstance(status, str):
            status_counts[status] = status_counts.get(status, 0) + 1
    labels = _labels(validation)
    return {
        "document_count": len(documents),
        "claim_count": claim_count,
        "split_counts": {name: len(rows) for name, rows in split_rows.items()},
        "document_ids": sorted(str(row.get("document_id", "")) for row in documents),
        "split_membership": split_membership,
        "cluster_count": len(clusters),
        "adjudication_status_counts": dict(sorted(status_counts.items())),
        **labels,
        "qc_status": str(validation.get("qc", {}).get("status", "unknown")),
        "schema_hashes": {name: sha256_file(root / "schemas" / name) for name in SCHEMA_NAMES},
        "artifacts": artifacts,
        "artifact_sha256": {item["relative_path"]: item["sha256"] for item in artifacts},
        "validation_summary": {
            "valid": validation.get("valid") is True,
            "validation_state": validation.get("validation_state"),
            "issue_codes": sorted({str(item.get("code")) for item in validation.get("issues", []) if isinstance(item, Mapping)}),
            "document_count": validation.get("document_count"),
            "provenance_count": validation.get("provenance_count"),
            "candidate_count": validation.get("candidate_count"),
            "cluster_count": validation.get("cluster_count"),
            "annotator_a_count": validation.get("annotator_a_count"),
            "annotator_b_count": validation.get("annotator_b_count"),
            "adjudication_count": validation.get("adjudication_count"),
        },
    }


def _content_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"timestamp", "content_identity", "status", "required_fields"}
    return {key: value for key, value in manifest.items() if key not in excluded}


def _content_identity(manifest: Mapping[str, Any]) -> str:
    return sha256_text(_json_canonical(_content_payload(manifest)))


def _manifest(root: Path, validation: Mapping[str, Any], artifacts: list[dict[str, Any]], timestamp: str) -> dict[str, Any]:
    snapshot = _snapshot(root, validation, artifacts)
    manifest: dict[str, Any] = {
        "status": "frozen",
        "dataset_id": DATASET_ID,
        "freeze_manifest_schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "annotation_version": "1.0",
        "adjudication_version": "1.0",
        "normalization_version": NORMALIZATION_VERSION,
        "source_provenance_version": SOURCE_PROVENANCE_VERSION,
        "split_version": SPLIT_VERSION,
        "scorer_version": SCORER_VERSION,
        **snapshot,
        "contract": {
            "canonical_artifact_paths": list(CANONICAL_ARTIFACT_PATHS),
            "manifest_relative_path": MANIFEST_RELATIVE_PATH,
            "target_split_counts": dict(TARGET_SPLIT_COUNTS),
        },
        "timestamp": timestamp,
        "required_fields": sorted(FROZEN_REQUIRED_FIELDS),
    }
    manifest["content_identity"] = _content_identity(manifest)
    return manifest


def _freeze_gates(root: Path) -> tuple[dict[str, Any] | None, list[dict[str, str]], list[dict[str, Any]]]:
    validation, reasons = _safe_validate(root)
    artifacts, inventory_reasons = _inventory(root)
    reasons.extend(inventory_reasons)
    reasons.extend(_configuration_reasons(root))
    if validation is None:
        return None, reasons, artifacts
    if not validation.get("dataset_populated"):
        reasons.append(_reason("DATASET_NOT_POPULATED", "empty scaffolding is valid but cannot be frozen"))
    if validation.get("valid") is not True:
        reasons.append(_reason("VALIDATION_GATE_FAILED", "Dataset 1 validation did not pass"))
        for issue in validation.get("issues", []):
            if isinstance(issue, Mapping):
                reasons.append(_reason(f"VALIDATION_{str(issue.get('code', 'ISSUE')).upper()}", str(issue.get("message", "validation issue")), str(issue.get("path", ""))))
    if validation.get("qc", {}).get("status") != "pass":
        reasons.append(_reason("QC_GATE_FAILED", "duplicate/QC gate must pass before freezing"))
    return validation, _deduplicate_reasons(reasons), artifacts


def _deduplicate_reasons(reasons: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    result = []
    for reason in reasons:
        key = (reason["code"], reason["message"], reason["path"])
        if key not in seen:
            seen.add(key)
            result.append(reason)
    return result


def freeze_dataset(root: Path, *, timestamp: str | None = None) -> dict[str, Any]:
    """Create or replace a manifest only after every populated-corpus gate passes."""
    root = Path(root)
    validation, reasons, artifacts = _freeze_gates(root)
    if reasons:
        return _result("NOT_FROZEN", reasons, validation=validation)
    assert validation is not None
    freeze_timestamp = timestamp or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest = _manifest(root, validation, artifacts, freeze_timestamp)
    manifest_path = root / MANIFEST_RELATIVE_PATH
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    except OSError as exc:
        return _result("NOT_FROZEN", [_reason("MANIFEST_WRITE_ERROR", str(exc), MANIFEST_RELATIVE_PATH)], validation=validation)
    return _result("FROZEN", [], manifest=manifest, manifest_path=MANIFEST_RELATIVE_PATH, validation=validation)


def _load_manifest(root: Path) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    path = root / MANIFEST_RELATIVE_PATH
    if not path.is_file():
        return None, [_reason("MISSING_MANIFEST", "freeze manifest is missing", MANIFEST_RELATIVE_PATH)]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, [_reason("MALFORMED_MANIFEST", str(exc), MANIFEST_RELATIVE_PATH)]
    if not isinstance(value, dict):
        return None, [_reason("MALFORMED_MANIFEST", "manifest must be a JSON object", MANIFEST_RELATIVE_PATH)]
    return value, []


def verify_freeze(root: Path) -> dict[str, Any]:
    """Verify a frozen Dataset 1 release without throwing on bad input."""
    root = Path(root)
    manifest, reasons = _load_manifest(root)
    if manifest is None:
        return _result("NOT_VERIFIED", reasons)
    schema_path = root / "schemas" / "freeze_manifest.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        for issue in validate_against_schema(manifest, schema, base_dir=schema_path.parent):
            reasons.append(_reason("MALFORMED_MANIFEST", issue.message, issue.path))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        reasons.append(_reason("MANIFEST_SCHEMA_ERROR", str(exc), "schemas/freeze_manifest.schema.json"))
    missing_fields = sorted(FROZEN_REQUIRED_FIELDS - set(manifest))
    for field in missing_fields:
        reasons.append(_reason("INCOMPLETE_MANIFEST", "required frozen manifest field is missing", field))
    if manifest.get("status") != "frozen":
        reasons.append(_reason("INVALID_FREEZE_STATUS", "manifest status must be frozen", "status"))
    if not isinstance(manifest.get("artifacts"), list):
        reasons.append(_reason("INCOMPLETE_MANIFEST", "artifact inventory must be a list", "artifacts"))
        return _result("NOT_VERIFIED", _deduplicate_reasons(reasons))
    expected_paths = list(CANONICAL_ARTIFACT_PATHS)
    manifest_paths = [item.get("relative_path") for item in manifest["artifacts"] if isinstance(item, Mapping)]
    if manifest_paths != expected_paths:
        reasons.append(_reason("ARTIFACT_INVENTORY_MISMATCH", "manifest artifact paths must exactly match the canonical ordered inventory", "artifacts"))
    artifacts, inventory_reasons = _inventory(root)
    reasons.extend(inventory_reasons)
    actual_by_path = {item["relative_path"]: item for item in artifacts}
    for item in manifest["artifacts"]:
        if not isinstance(item, Mapping):
            reasons.append(_reason("MALFORMED_MANIFEST", "artifact entry must be an object", "artifacts"))
            continue
        path = item.get("relative_path")
        current = actual_by_path.get(path)
        if not isinstance(path, str) or current is None:
            reasons.append(_reason("ARTIFACT_INVENTORY_MISMATCH", "artifact entry names an unknown path", str(path)))
            continue
        if item.get("expected") is not True or item.get("exists") is not True:
            reasons.append(_reason("INCOMPLETE_MANIFEST", "frozen artifact must be expected and present", path))
        if not current["exists"]:
            continue
        if item.get("sha256") != current["sha256"]:
            reasons.append(_reason("HASH_MISMATCH", "artifact SHA-256 does not match manifest", path))
        if item.get("size_bytes") != current["size_bytes"]:
            reasons.append(_reason("SIZE_MISMATCH", "artifact size does not match manifest", path))
        if manifest.get("artifact_sha256", {}).get(path) != item.get("sha256"):
            reasons.append(_reason("MANIFEST_HASH_MISMATCH", "artifact_sha256 does not agree with artifact inventory", path))
    if not isinstance(manifest.get("content_identity"), str) or manifest.get("content_identity") != _content_identity(manifest):
        reasons.append(_reason("CONTENT_IDENTITY_MISMATCH", "manifest content identity is invalid", "content_identity"))
    validation, validation_reasons = _safe_validate(root)
    reasons.extend(validation_reasons)
    if validation is not None:
        try:
            snapshot = _snapshot(root, validation, artifacts)
            for field in ("document_count", "claim_count", "split_counts", "document_ids", "split_membership", "cluster_count"):
                if manifest.get(field) != snapshot[field]:
                    reasons.append(_reason("COUNT_OR_SPLIT_MISMATCH", "current dataset value differs from frozen manifest", field))
            if manifest.get("validation_summary") != snapshot["validation_summary"]:
                reasons.append(_reason("VALIDATION_SUMMARY_MISMATCH", "current validation summary differs from frozen manifest", "validation_summary"))
            if manifest.get("qc_status") != snapshot["qc_status"]:
                reasons.append(_reason("QC_STATUS_MISMATCH", "current QC status differs from frozen manifest", "qc_status"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(_reason("DATASET_INPUT_ERROR", str(exc)))
    return _result("VERIFIED" if not reasons else "NOT_VERIFIED", _deduplicate_reasons(reasons), manifest=manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze or verify the deterministic Dataset 1 release.")
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/claim_understanding"))
    parser.add_argument("--verify", action="store_true", help="verify an existing freeze manifest")
    parser.add_argument("--timestamp", help="optional ISO-8601 freeze metadata; excluded from content identity")
    args = parser.parse_args()
    result = verify_freeze(args.root) if args.verify else freeze_dataset(args.root, timestamp=args.timestamp)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] not in {"FROZEN", "VERIFIED"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
