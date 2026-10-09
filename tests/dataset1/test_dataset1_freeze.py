from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.dataset1_common import normalize_document, sha256_text
from scripts.dataset1_freeze import (
    CANONICAL_ARTIFACT_PATHS,
    MANIFEST_RELATIVE_PATH,
    freeze_dataset,
    verify_freeze,
)


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "data" / "claim_understanding"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8", newline="\n")


def _span(text: str, fragment: str) -> dict[str, object]:
    start = text.index(fragment)
    return {"start": start, "end": start + len(fragment), "text": fragment}


def _annotation(document: dict, annotation_id: str, annotator_id: str) -> dict:
    text = document["normalized_text"]
    original = document["original_text"]
    return {
        "schema_version": "1.0", "document_id": document["document_id"],
        "annotation_id": annotation_id, "annotator_id": annotator_id,
        "annotation_version": "1.0", "normalized_text": text, "abstentions": [],
        "claims": [{
            "claim_id": f"claim-{document['document_id']}", "source_span": _span(text, text),
            "original_source_span": _span(original, original), "original_text": original,
            "claim_type": "company_announcement", "is_atomic": True,
            "is_verifiable_as_textual_claim": True, "entities": [], "relations": [],
            "temporal_annotations": [], "quantity_annotations": [], "location_annotations": [],
            "modifiers": [], "negation": None, "modality": None, "non_claim_spans": [],
        }],
    }


def _candidate(document: dict, index: int) -> dict:
    return {
        "schema_version": "1.0", "candidate_id": f"candidate-{index:03d}", "source_id": document["source_id"],
        "canonical_uri": f"https://example.org/{index}", "source_category": "publisher", "language": "en",
        "discovery": {"method": "publisher_archive", "route": f"fixture/{index}", "discovered_at": "2026-10-04T00:00:00Z", "discovered_by": "fixture"},
        "source_identity": {"identity_id": f"example-{index:03d}", "display_name": "Example", "canonical_domain": "example.org", "identity_basis": "publisher_domain"},
        "independence": {"relationship": "independent_reporting", "origin_uri": None, "origin_source_identity": None},
        "publication": {"precision": "not_available", "value": None, "verbatim": None},
        "collection": {"collected_at": "2026-10-04T00:00:00Z", "collector_id": "fixture", "access_status": "accessible", "content_sha256": sha256_text(document["original_text"]), "content_length": len(document["original_text"]), "content_format": "text"},
        "license": "example", "license_verification_status": "verified", "usage_assessment": "permitted",
        "eligibility_checks": {"english_language": True, "meaningful_text": True, "attributable_source": True, "stable_source_identity": True, "accessible": True, "license_reviewed": True, "source_independence_reviewed": True},
        "collection_status": "retained", "exclusion_reason": None,
        "duplicate_review": {"status": "unique", "canonical_candidate_id": None, "reviewed_by": "fixture", "reviewed_at": "2026-10-04T00:00:00Z"},
        "retained_document_id": document["document_id"], "notes": "",
    }


def _make_fixture(root: Path, *, populated: bool = True) -> Path:
    for relative in CANONICAL_ARTIFACT_PATHS:
        source = SOURCE_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copyfile(source, destination)
        else:
            destination.touch()
    if not populated:
        return root
    documents = []
    for index in range(230):
        original = f"Organization{index} announced unique-token-{index}."
        document = normalize_document({"document_id": f"doc-{index:03d}", "source_id": f"source-{index:03d}", "original_text": original})
        documents.append(document)
    provenance = [{
        "document_id": document["document_id"], "source_id": document["source_id"],
        "source_category": "publisher", "acquisition_date": "2026-10-04", "license": "example",
        "license_verification_status": "verified", "uri": f"https://example.org/{index}",
        "checksum": sha256_text(document["original_text"]), "collector_id": "fixture", "notes": "",
    } for index, document in enumerate(documents)]
    candidates = [_candidate(document, index) for index, document in enumerate(documents)]
    clusters = [{"document_id": document["document_id"], "story_cluster_id": f"cluster-{index:03d}", "cluster_version": "1.0"} for index, document in enumerate(documents)]
    annotator_a = [_annotation(document, f"a-{index:03d}", "annotator-a") for index, document in enumerate(documents)]
    annotator_b = [_annotation(document, f"b-{index:03d}", "annotator-b") for index, document in enumerate(documents)]
    adjudications = [{
        "schema_version": "1.0", "adjudication_id": f"adj-{index:03d}", "document_id": document["document_id"],
        "annotator_a_annotation_id": f"a-{index:03d}", "annotator_b_annotation_id": f"b-{index:03d}",
        "adjudicator_id": "adjudicator", "adjudication_version": "1.0", "status": "adjudicated",
        "rationale": "fixture adjudication", "gold_annotation": _annotation(document, f"gold-{index:03d}", "adjudicator"),
    } for index, document in enumerate(documents)]
    boundaries = {"development": (0, 80), "validation": (80, 130), "frozen_test": (130, 230)}
    splits = {
        name: [{
            "document_id": document["document_id"], "story_cluster_id": f"cluster-{index:03d}",
            "split_version": "dataset1-cluster-disjoint-v1", "assignment_seed": "fixture-seed",
        } for index, document in enumerate(documents[start:end], start)]
        for name, (start, end) in boundaries.items()
    }
    _write_jsonl(root / "documents/normalized.jsonl", documents)
    _write_jsonl(root / "provenance/sources.jsonl", provenance)
    _write_jsonl(root / "provenance/candidates.jsonl", candidates)
    _write_jsonl(root / "clusters/story_clusters.jsonl", clusters)
    _write_jsonl(root / "annotations/annotator_a.jsonl", annotator_a)
    _write_jsonl(root / "annotations/annotator_b.jsonl", annotator_b)
    _write_jsonl(root / "annotations/adjudicated.jsonl", adjudications)
    for name, rows in splits.items():
        _write_jsonl(root / f"splits/{name}.jsonl", rows)
    return root


def _manifest(root: Path) -> dict:
    return json.loads((root / MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))


def test_freeze_and_verification_record_deterministic_counts_and_hashes(tmp_path):
    root = _make_fixture(tmp_path / "dataset")

    frozen = freeze_dataset(root, timestamp="2026-10-09T00:00:00Z")
    manifest = _manifest(root)

    assert frozen["status"] == "FROZEN"
    assert verify_freeze(root)["status"] == "VERIFIED"
    assert manifest["document_count"] == 230
    assert manifest["claim_count"] == 230
    assert manifest["split_counts"] == {"development": 80, "validation": 50, "frozen_test": 100}
    assert [item["relative_path"] for item in manifest["artifacts"]] == list(CANONICAL_ARTIFACT_PATHS)
    assert all(item["sha256"] for item in manifest["artifacts"])


def test_verification_detects_modified_deleted_and_unexpected_artifacts(tmp_path):
    root = _make_fixture(tmp_path / "dataset")
    assert freeze_dataset(root, timestamp="2026-10-09T00:00:00Z")["status"] == "FROZEN"

    documents = root / "documents/normalized.jsonl"
    documents.write_text(documents.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert "HASH_MISMATCH" in {item["code"] for item in verify_freeze(root)["failure_reasons"]}

    assert freeze_dataset(root, timestamp="2026-10-09T00:00:00Z")["status"] == "FROZEN"
    documents.write_text(documents.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")
    assert freeze_dataset(root, timestamp="2026-10-09T00:00:00Z")["status"] == "FROZEN"
    (root / "provenance/sources.jsonl").unlink()
    assert "MISSING_ARTIFACT" in {item["code"] for item in verify_freeze(root)["failure_reasons"]}

    _make_fixture(root)
    assert freeze_dataset(root, timestamp="2026-10-09T00:00:00Z")["status"] == "FROZEN"
    (root / "documents/unexpected.txt").write_text("no", encoding="utf-8")
    assert "UNEXPECTED_ARTIFACT" in {item["code"] for item in verify_freeze(root)["failure_reasons"]}


def test_verification_handles_malformed_or_incomplete_manifest_safely(tmp_path):
    root = _make_fixture(tmp_path / "dataset")
    manifest_path = root / MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{", encoding="utf-8")
    assert "MALFORMED_MANIFEST" in {item["code"] for item in verify_freeze(root)["failure_reasons"]}

    assert freeze_dataset(root, timestamp="2026-10-09T00:00:00Z")["status"] == "FROZEN"
    manifest = _manifest(root)
    del manifest["content_identity"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "INCOMPLETE_MANIFEST" in {item["code"] for item in verify_freeze(root)["failure_reasons"]}


def test_content_identity_is_independent_of_timestamp_and_filesystem_creation_order(tmp_path):
    first = _make_fixture(tmp_path / "first")
    second = _make_fixture(tmp_path / "second")

    assert freeze_dataset(first, timestamp="2026-10-09T00:00:00Z")["status"] == "FROZEN"
    assert freeze_dataset(second, timestamp="2026-10-10T00:00:00Z")["status"] == "FROZEN"
    assert _manifest(first)["content_identity"] == _manifest(second)["content_identity"]


def test_freeze_is_blocked_by_qc_and_empty_scaffolding(tmp_path):
    empty = _make_fixture(tmp_path / "empty", populated=False)
    empty_result = freeze_dataset(empty)
    assert empty_result["status"] == "NOT_FROZEN"
    assert "DATASET_NOT_POPULATED" in {item["code"] for item in empty_result["failure_reasons"]}

    root = _make_fixture(tmp_path / "dataset")
    rows = [json.loads(line) for line in (root / "documents/normalized.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[1]["original_text"] = rows[0]["original_text"]
    rows[1]["normalized_text"] = rows[0]["normalized_text"]
    rows[1]["normalized_sha256"] = rows[0]["normalized_sha256"]
    rows[1]["normalized_to_original_offsets"] = rows[0]["normalized_to_original_offsets"]
    _write_jsonl(root / "documents/normalized.jsonl", rows)
    qc_result = freeze_dataset(root)
    assert qc_result["status"] == "NOT_FROZEN"
    assert "QC_GATE_FAILED" in {item["code"] for item in qc_result["failure_reasons"]}
