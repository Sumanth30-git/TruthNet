from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.dataset1_common import (
    NORMALIZATION_VERSION,
    assign_splits,
    exact_hash_duplicates,
    near_duplicate_pairs,
    normalize_document,
    map_normalized_span_to_original,
    normalized_text_duplicates,
    validate_claim_annotation,
    validate_document,
    validate_provenance,
    validate_story_cluster_records,
    validate_split_records,
)
from scripts.dataset1_schema import load_schema, validate_against_schema
from scripts.dataset1_score import score_dataset
from scripts.dataset1_validate import validate_dataset
import scripts.dataset1_validate as dataset1_validate_module


TEXT = "Acme launched Product Y in Bengaluru today."


def _span(text: str, fragment: str) -> dict[str, object]:
    start = text.index(fragment)
    return {"start": start, "end": start + len(fragment), "text": fragment}


def _document(document_id: str = "doc-1", *, synthetic: bool = False, cluster: str | None = None) -> dict:
    record = normalize_document({"document_id": document_id, "source_id": f"source-{document_id}", "original_text": TEXT})
    record["is_synthetic"] = synthetic
    if cluster is not None:
        record["story_cluster_id"] = cluster
    return record


def _annotation(document: dict) -> dict:
    return {
        "schema_version": "1.0", "document_id": document["document_id"], "annotation_id": "ann-1",
        "annotator_id": "annotator-a", "annotation_version": "1.0", "normalized_text": document["normalized_text"],
        "abstentions": [],
        "claims": [{
            "claim_id": "claim-1", "source_span": _span(document["normalized_text"], TEXT), "original_text": TEXT,
            "original_source_span": _span(document["original_text"], TEXT),
            "claim_type": "product_announcement", "is_atomic": True, "is_verifiable_as_textual_claim": True,
            "entities": [{"entity_type": "organization", "source_span": _span(document["normalized_text"], "Acme")}],
            "relations": [], "temporal_annotations": [], "quantity_annotations": [], "location_annotations": [],
            "modifiers": [], "negation": None, "modality": None, "non_claim_spans": [],
        }],
    }


def _issues(records) -> set[str]:
    return {issue.code for issue in records}


def test_valid_document_and_normalization_are_deterministic_with_offsets():
    raw = {"document_id": "doc-1", "source_id": "source-1", "original_text": "Cafe\u0301\r\nLine two"}
    first = normalize_document(raw)
    second = normalize_document(raw)

    assert first == second
    assert first["normalized_text"] == "Café\nLine two"
    assert first["normalization_version"] == NORMALIZATION_VERSION
    assert len(first["normalized_to_original_offsets"]) == len(first["normalized_text"]) + 1
    assert validate_document(first) == []


def test_malformed_document_and_checksum_mismatch_are_rejected():
    document = _document()
    document["normalized_sha256"] = "not-a-checksum"
    document["language"] = "fr"

    assert {"invalid_language", "checksum_mismatch"} <= _issues(validate_document(document))


def test_invalid_offsets_and_original_text_span_mismatch_are_rejected():
    document = _document()
    annotation = _annotation(document)
    annotation["claims"][0]["source_span"]["end"] = len(document["normalized_text"]) + 1
    annotation["claims"][0]["original_text"] = "wrong"

    assert {"invalid_offset", "original_text_mismatch"} <= _issues(validate_claim_annotation(annotation, document))


def test_duplicate_claims_and_invalid_enums_are_rejected():
    document = _document()
    annotation = _annotation(document)
    duplicate = deepcopy(annotation["claims"][0])
    duplicate["claim_id"] = "claim-2"
    duplicate["claim_type"] = "not-a-claim-type"
    annotation["claims"].append(duplicate)

    assert {"duplicate_claim", "invalid_enum"} <= _issues(validate_claim_annotation(annotation, document))


def test_relation_requires_grounded_entity_links_and_support_spans():
    document = _document()
    annotation = _annotation(document)
    annotation["claims"][0]["relations"] = [{
        "relation_type": "launch", "predicate": "launched", "subject_span": _span(document["normalized_text"], "Acme"),
        "object_span": {"start": 999, "end": 1000, "text": "missing"},
        "support_spans": [_span(document["normalized_text"], TEXT)],
    }]

    assert "invalid_offset" in _issues(validate_claim_annotation(annotation, document))


def test_provenance_contract_validates_required_fields_and_enum():
    provenance = {
        "document_id": "doc-1", "source_id": "source-1", "source_category": "publisher",
        "acquisition_date": "2026-10-04", "license": "example license", "license_verification_status": "verified",
        "uri": "https://example.org/article", "checksum": "a" * 64, "collector_id": "collector-1", "notes": "",
    }
    assert validate_provenance(provenance) == []
    provenance["license_verification_status"] = "perhaps"
    assert "invalid_enum" in _issues(validate_provenance(provenance))


def _populated_documents() -> list[dict]:
    return [_document(f"doc-{index}", cluster=f"cluster-{index}") for index in range(230)]


def _valid_split_rows(documents: list[dict]) -> dict[str, list[dict]]:
    result = {"development": [], "validation": [], "frozen_test": []}
    boundaries = (("development", 0, 80), ("validation", 80, 130), ("frozen_test", 130, 230))
    for split_name, start, end in boundaries:
        for document in documents[start:end]:
            result[split_name].append({
                "document_id": document["document_id"], "story_cluster_id": document["story_cluster_id"],
                "split_version": "dataset1-cluster-disjoint-v1", "assignment_seed": "fixed-seed",
            })
    return result


def _clusters(documents: list[dict]) -> list[dict]:
    return [
        {"document_id": document["document_id"], "story_cluster_id": document["story_cluster_id"], "cluster_version": "1.0"}
        for document in documents
    ]


def test_empty_dataset_reports_not_populated():
    assert [issue.code for issue in validate_split_records([], {})] == ["dataset_not_populated"]


def test_split_validation_rejects_cluster_leakage_and_synthetic_frozen_test():
    documents = _populated_documents()
    rows = _valid_split_rows(documents)
    documents[130]["is_synthetic"] = True
    documents[80]["story_cluster_id"] = documents[0]["story_cluster_id"]
    rows["validation"][0]["story_cluster_id"] = documents[0]["story_cluster_id"]

    issues = _issues(validate_split_records(documents, rows))
    assert {"cluster_leakage", "synthetic_frozen_test"} <= issues


def test_split_count_validation_and_assignment_are_deterministic():
    documents = _populated_documents()
    rows = _valid_split_rows(documents)
    clusters = _clusters(documents)
    rows["development"].pop()
    assert "split_count" in _issues(validate_split_records(documents, rows, clusters))

    assigned_once = assign_splits(documents, clusters, "dataset1-seed")
    assigned_twice = assign_splits(documents, clusters, "dataset1-seed")
    assert assigned_once == assigned_twice
    assert {name: len(values) for name, values in assigned_once.items()} == {"development": 80, "validation": 50, "frozen_test": 100}


def test_duplicate_helpers_find_exact_normalized_and_near_duplicates():
    left = _document("doc-a")
    right = _document("doc-b")
    right["normalized_text"] = left["normalized_text"]
    duplicate_id = _document("doc-a")

    assert "doc-a" in {row["document_id"] for row in [duplicate_id]}
    assert normalized_text_duplicates([left, right])
    assert exact_hash_duplicates([left, right])
    assert near_duplicate_pairs([left, right], threshold=1.0) == [("doc-a", "doc-b", 1.0)]


def test_story_cluster_records_require_complete_manual_membership():
    documents = [_document("doc-a"), _document("doc-b")]
    rows = [{"document_id": "doc-a", "story_cluster_id": "story-1", "cluster_version": "1.0"}]

    assert "missing_cluster_membership" in _issues(validate_story_cluster_records(documents, rows))


def _workflow_result(monkeypatch, documents: list[dict], *, clusters: list[dict] | None = None, splits: dict[str, list[dict]] | None = None, provenance: list[dict] | None = None, annotator_a: list[dict] | None = None, annotator_b: list[dict] | None = None, adjudications: list[dict] | None = None):
    rows = {
        ("documents", "normalized.jsonl"): documents,
        ("provenance", "sources.jsonl"): provenance or [],
        ("clusters", "story_clusters.jsonl"): clusters or [],
        ("annotations", "annotator_a.jsonl"): annotator_a or [],
        ("annotations", "annotator_b.jsonl"): annotator_b or [],
        ("annotations", "adjudicated.jsonl"): adjudications or [],
    }
    for split in ("development", "validation", "frozen_test"):
        rows[("splits", f"{split}.jsonl")] = (splits or {}).get(split, [])
    monkeypatch.setattr(dataset1_validate_module, "read_jsonl", lambda path: rows.get((path.parent.name, path.name), []))
    return validate_dataset(Path("in-memory-dataset1"))


def _issue_codes(result: dict) -> set[str]:
    return {item["code"] for item in result["issues"]}


def test_workflow_reports_empty_scaffold_as_valid_but_not_populated(monkeypatch):
    result = _workflow_result(monkeypatch, [])

    assert result["validation_state"] == "empty_scaffold"
    assert result["valid"] is True
    assert result["qc"]["status"] == "empty"


def test_workflow_requires_canonical_clusters_for_populated_corpus(monkeypatch):
    documents = _populated_documents()
    result = _workflow_result(monkeypatch, documents, splits=_valid_split_rows(documents))

    assert "missing_canonical_clusters" in _issue_codes(result)
    assert result["valid"] is False


def test_workflow_enforces_document_cluster_disagreement_and_split_leakage(monkeypatch):
    documents = _populated_documents()
    clusters = _clusters(documents)
    clusters[0]["story_cluster_id"] = "canonical-other"
    splits = _valid_split_rows(documents)
    splits["validation"][0]["story_cluster_id"] = splits["development"][0]["story_cluster_id"]
    result = _workflow_result(monkeypatch, documents, clusters=clusters, splits=splits)

    assert {"cluster_disagreement", "cluster_leakage"} <= _issue_codes(result)


def test_workflow_connects_cross_file_integrity_and_duplicate_qc(monkeypatch):
    left, right = _document("doc-a", cluster="story-a"), _document("doc-b", cluster="story-b")
    result = _workflow_result(monkeypatch, [left, right], clusters=_clusters([left, right]), annotator_a=[_annotation(left)])

    assert {"missing_provenance", "incomplete_double_annotation", "incomplete_adjudication", "qc_gate"} <= _issue_codes(result)
    assert result["qc"]["exact_normalized_duplicates"]
    assert result["qc"]["exact_hash_duplicates"]


def test_schema_validator_annotation_validator_and_scorer_share_abstentions():
    document = _document()
    annotation = _annotation(document)
    assert validate_against_schema(annotation, load_schema("claim_annotation.schema.json")) == []
    assert validate_claim_annotation(annotation, document) == []

    prediction = {
        "document_id": document["document_id"], "normalized_text": document["normalized_text"], "claims": [],
        "abstentions": [{"scope": "document", "reason": "insufficient_text", "claim_id": None}],
    }
    schema_prediction = deepcopy(annotation)
    schema_prediction["claims"] = []
    schema_prediction["abstentions"] = prediction["abstentions"]
    assert validate_against_schema(schema_prediction, load_schema("claim_annotation.schema.json")) == []
    assert validate_claim_annotation(schema_prediction, document) == []
    assert score_dataset(annotation, prediction)["scorer_abstained"] is True


def test_original_source_span_maps_across_nfc_and_crlf_normalization():
    document = normalize_document({"document_id": "doc-nfc", "source_id": "source-nfc", "original_text": "Cafe\u0301\r\nLine"})
    normalized_span = {"start": 0, "end": len("Café\n"), "text": "Café\n"}
    original_span = map_normalized_span_to_original(document, 0, len("Café\n"))
    annotation = {
        "schema_version": "1.0", "document_id": "doc-nfc", "annotation_id": "ann-nfc", "annotator_id": "a",
        "annotation_version": "1.0", "normalized_text": document["normalized_text"], "abstentions": [],
        "claims": [{
            "claim_id": "claim-nfc", "source_span": normalized_span, "original_source_span": original_span,
            "original_text": original_span["text"], "claim_type": "other", "is_atomic": True,
            "is_verifiable_as_textual_claim": True,
        }],
    }

    assert original_span == {"start": 0, "end": len("Cafe\u0301\r\n"), "text": "Cafe\u0301\r\n"}
    assert validate_against_schema(annotation, load_schema("claim_annotation.schema.json")) == []
    assert validate_claim_annotation(annotation, document) == []


@pytest.mark.parametrize("count,has_document_count_issue", [(229, True), (230, False), (231, True)])
def test_split_contract_enforces_exact_populated_document_count(count, has_document_count_issue):
    documents = [_document(f"doc-{index}", cluster=f"cluster-{index}") for index in range(count)]
    issues = _issues(validate_split_records(documents, _valid_split_rows(documents), _clusters(documents)))

    assert ("document_count" in issues) is has_document_count_issue


def test_split_contract_rejects_unassigned_and_duplicate_assignments():
    documents = _populated_documents()
    rows = _valid_split_rows(documents)
    rows["development"].pop()
    rows["validation"].append(deepcopy(rows["development"][0]))
    issues = _issues(validate_split_records(documents, rows, _clusters(documents)))

    assert {"unassigned_document", "duplicate_document_id", "split_count"} <= issues
