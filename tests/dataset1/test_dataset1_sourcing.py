from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from scripts.dataset1_common import (
    normalize_document,
    sha256_text,
    validate_collection_candidate,
    validate_collection_cross_file_integrity,
)
from scripts.dataset1_schema import load_schema, validate_against_schema
from scripts.dataset1_validate import validate_dataset


def _candidate(*, status: str = "eligible") -> dict:
    retained = status == "retained"
    text = "Example candidate text."
    return {
        "schema_version": "1.0", "candidate_id": "candidate-1", "source_id": "source-1",
        "canonical_uri": "https://news.example.org/article", "source_category": "publisher", "language": "en",
        "discovery": {"method": "publisher_archive", "route": "archive/2026", "discovered_at": "2026-10-04T10:15:00Z", "discovered_by": "collector-1"},
        "source_identity": {"identity_id": "example-news", "display_name": "Example News", "canonical_domain": "news.example.org", "identity_basis": "publisher_domain"},
        "independence": {"relationship": "independent_reporting", "origin_uri": None, "origin_source_identity": None},
        "publication": {"precision": "date", "value": "2026-10-03", "verbatim": "Oct. 3, 2026"},
        "collection": {"collected_at": "2026-10-04T10:16:00Z", "collector_id": "collector-1", "access_status": "accessible", "content_sha256": sha256_text(text), "content_length": len(text), "content_format": "html"},
        "license": "Example terms", "license_verification_status": "verified", "usage_assessment": "permitted",
        "eligibility_checks": {"english_language": True, "meaningful_text": True, "attributable_source": True, "stable_source_identity": True, "accessible": True, "license_reviewed": True, "source_independence_reviewed": True},
        "collection_status": status, "exclusion_reason": None,
        "duplicate_review": {"status": "unique", "canonical_candidate_id": None, "reviewed_by": "collector-1", "reviewed_at": "2026-10-04T10:17:00Z"},
        "retained_document_id": "doc-1" if retained else None, "notes": "",
    }


def _codes(issues) -> set[str]:
    return {issue.code for issue in issues}


def test_valid_eligible_candidate_satisfies_schema_and_contract():
    candidate = _candidate()

    assert validate_against_schema(candidate, load_schema("collection_candidate.schema.json")) == []
    assert validate_collection_candidate(candidate) == []


def test_candidate_contract_rejects_unexpected_fields_invalid_identity_and_language():
    candidate = _candidate()
    candidate["unexpected"] = True
    candidate["source_identity"]["identity_id"] = ""
    candidate["language"] = "fr"

    assert "schema_additional" in _codes(validate_against_schema(candidate, load_schema("collection_candidate.schema.json")))
    assert {"invalid_source_identity", "invalid_language"} <= _codes(validate_collection_candidate(candidate))


def test_publication_and_exclusion_are_explicit_and_never_invented():
    candidate = _candidate()
    candidate["publication"] = {"precision": "not_available", "value": "2026-10-03", "verbatim": None}
    assert "invalid_publication" in _codes(validate_collection_candidate(candidate))

    excluded = _candidate(status="excluded")
    excluded["eligibility_checks"]["meaningful_text"] = False
    excluded["exclusion_reason"] = "insufficient_text"
    assert validate_collection_candidate(excluded) == []
    excluded["exclusion_reason"] = None
    assert "missing_exclusion_reason" in _codes(validate_collection_candidate(excluded))


def test_duplicate_and_syndication_representation_requires_human_traceability():
    canonical = _candidate()
    duplicate = _candidate(status="excluded")
    duplicate["candidate_id"] = "candidate-2"
    duplicate["collection_status"] = "excluded"
    duplicate["exclusion_reason"] = "duplicate_exact"
    duplicate["duplicate_review"] = {"status": "exact_duplicate", "canonical_candidate_id": "candidate-1", "reviewed_by": "reviewer", "reviewed_at": "2026-10-04T10:17:00Z"}
    assert validate_collection_candidate(duplicate) == []
    assert validate_collection_cross_file_integrity([], [], [canonical, duplicate]) == []

    syndicated = _candidate()
    syndicated["independence"] = {"relationship": "syndicated_or_republished", "origin_uri": None, "origin_source_identity": None}
    assert "missing_origin" in _codes(validate_collection_candidate(syndicated))


def test_retained_candidate_requires_matching_document_and_provenance_checksum():
    text = "Example candidate text."
    candidate = _candidate(status="retained")
    document = {"document_id": "doc-1", "source_id": "source-1"}
    provenance = {"document_id": "doc-1", "source_id": "source-1", "checksum": sha256_text(text)}

    assert validate_collection_cross_file_integrity([document], [provenance], [candidate]) == []
    candidate["collection"]["content_sha256"] = "a" * 64
    assert "candidate_checksum_mismatch" in _codes(validate_collection_cross_file_integrity([document], [provenance], [candidate]))


def test_populated_candidate_workflow_requires_existing_document_provenance(tmp_path):
    document = normalize_document({"document_id": "doc-1", "source_id": "source-1", "original_text": "Example candidate text."})
    document_path = tmp_path / "documents" / "normalized.jsonl"
    document_path.parent.mkdir(parents=True)
    document_path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    candidate_path = tmp_path / "provenance" / "candidates.jsonl"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(json.dumps(_candidate(status="retained"), sort_keys=True) + "\n", encoding="utf-8")

    assert "missing_provenance" in {item["code"] for item in validate_dataset(tmp_path)["issues"]}


def test_dataset_validation_is_deterministic_and_compatible_with_empty_scaffold(tmp_path):
    candidate_path = tmp_path / "provenance" / "candidates.jsonl"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(json.dumps(_candidate(), sort_keys=True) + "\n", encoding="utf-8")

    first = validate_dataset(tmp_path)
    second = validate_dataset(tmp_path)
    assert first == second
    assert first["valid"] is True
    assert first["candidate_count"] == 1

    root = Path(__file__).resolve().parents[2] / "data" / "claim_understanding"
    scaffold = validate_dataset(root)
    assert scaffold["validation_state"] == "empty_scaffold"
    assert scaffold["valid"] is True
