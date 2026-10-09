"""Dataset 1-only contracts, validators, and deterministic utilities.

This module is deliberately standard-library-only and is never imported by the
TruthNet production runtime.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DATASET_ID = "claim_understanding_dataset_1"
NORMALIZATION_VERSION = "dataset1-nfc-lf-v1"
SPLIT_VERSION = "dataset1-cluster-disjoint-v1"
SCORER_VERSION = "dataset1-usr-v3"
DATASET_VERSION = "dataset1-eval-v1"
SOURCE_PROVENANCE_VERSION = "1.0"
TARGET_SPLIT_COUNTS = {"development": 80, "validation": 50, "frozen_test": 100}
TEMPORAL_NORMALIZATIONS = {"iso_date", "relative", "verbatim"}
NEAR_DUPLICATE_REVIEW_THRESHOLD = 0.9
CLAIM_TYPES = {"general_fact", "government_announcement", "company_announcement", "product_announcement", "financial_price", "scientific", "legal_court", "sports_result", "person_statement", "event", "organization_announcement", "other", "unknown"}
ENTITY_TYPES = {"person", "organization", "location", "country", "product", "event", "other"}
LICENSE_STATUSES = {"verified", "unverified", "not_applicable"}
COLLECTION_LICENSE_STATUSES = LICENSE_STATUSES | {"not_found"}
SOURCE_CATEGORIES = {"publisher", "broadcaster", "wire_service", "government", "official_organization", "licensed_archive", "other_documented"}
COLLECTION_STATUSES = {"eligible", "excluded", "retained"}
INDEPENDENCE_RELATIONSHIPS = {"primary_or_official", "independent_reporting", "syndicated_or_republished", "copy_or_mirror", "unknown_needs_review"}
DUPLICATE_REVIEW_STATUSES = {"not_assessed", "unique", "exact_duplicate", "near_duplicate", "same_syndicated_text", "needs_human_review"}
EXCLUSION_REASONS = {"unsupported_language", "insufficient_text", "unattributable_source", "unstable_source_identity", "inaccessible", "licensing_or_usage", "duplicate_exact", "duplicate_near", "syndicated_or_republished", "outside_scope", "other_documented"}
ABSTENTION_SCOPES = {"document", "claim"}
ABSTENTION_REASONS = {"insufficient_text", "ambiguous_structure", "not_applicable", "other"}
MODALITIES = {"asserted", "possible", "probable", "obligated", "permitted", "other"}
MODIFIER_TYPES = {"degree", "frequency", "scope", "comparative", "other"}
NON_CLAIM_REASONS = {"question", "opinion", "instruction", "fragment", "other"}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str = ""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalise_with_offsets(original_text: str) -> tuple[str, list[int]]:
    """NFC/LF text plus normalized-boundary -> original-boundary offsets."""
    chars: list[tuple[str, int]] = []
    index = 0
    while index < len(original_text):
        if original_text[index] == "\r":
            if index + 1 < len(original_text) and original_text[index + 1] == "\n":
                chars.append(("\n", index + 2)); index += 2
            else:
                chars.append(("\n", index + 1)); index += 1
        else:
            chars.append((original_text[index], index + 1)); index += 1
    raw = ""; normalized = ""; offsets = [0]
    for char, original_end in chars:
        raw += char
        updated = unicodedata.normalize("NFC", raw)
        shared = 0
        while shared < min(len(normalized), len(updated)) and normalized[shared] == updated[shared]:
            shared += 1
        offsets = offsets[:shared + 1] + [original_end] * (len(updated) - shared)
        normalized = updated
    return normalized, offsets


def normalize_document(record: Mapping[str, Any]) -> dict[str, Any]:
    if not _nonempty(record.get("document_id")) or not _nonempty(record.get("source_id")):
        raise ValueError("document_id and source_id must be non-empty strings")
    if not isinstance(record.get("original_text"), str):
        raise ValueError("original_text must be a string")
    if "is_synthetic" in record and not isinstance(record["is_synthetic"], bool):
        raise ValueError("is_synthetic must be boolean")
    normalized, offsets = _normalise_with_offsets(record["original_text"])
    return {"schema_version": "1.0", "dataset_id": DATASET_ID, "document_id": record["document_id"], "source_id": record["source_id"], "language": record.get("language", "en"), "original_text": record["original_text"], "normalized_text": normalized, "normalized_to_original_offsets": offsets, "normalization_version": NORMALIZATION_VERSION, "normalized_sha256": sha256_text(normalized), "is_synthetic": record.get("is_synthetic", False)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try: row = json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict): raise ValueError(f"{path}:{number}: JSONL rows must be objects")
            rows.append(row)
    return rows


def _span_issues(span: object, text: str, path: str) -> list[ValidationIssue]:
    if not isinstance(span, Mapping): return [ValidationIssue("invalid_span", "span must be an object", path)]
    start, end, stated = span.get("start"), span.get("end"), span.get("text")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(text):
        return [ValidationIssue("invalid_offset", "span offsets are outside source text", path)]
    if stated != text[start:end]: return [ValidationIssue("span_text_mismatch", "span text does not match source text", path)]
    return []


def validate_document(record: Mapping[str, Any]) -> list[ValidationIssue]:
    issues = []
    for key in ("document_id", "source_id", "original_text", "normalized_text", "normalized_sha256"):
        if not _nonempty(record.get(key)): issues.append(ValidationIssue("missing_field", f"{key} must be a non-empty string", key))
    if record.get("language") != "en": issues.append(ValidationIssue("invalid_language", "Dataset 1 documents must be English", "language"))
    if not isinstance(record.get("is_synthetic"), bool): issues.append(ValidationIssue("invalid_synthetic_flag", "is_synthetic must be boolean", "is_synthetic"))
    if isinstance(record.get("original_text"), str) and isinstance(record.get("normalized_text"), str):
        expected = normalize_document({"document_id": record.get("document_id") or "x", "source_id": record.get("source_id") or "x", "original_text": record["original_text"], "language": "en", "is_synthetic": False})
        if record["normalized_text"] != expected["normalized_text"]: issues.append(ValidationIssue("normalization_mismatch", "normalized_text is not deterministic NFC/LF output", "normalized_text"))
        if record.get("normalized_to_original_offsets") != expected["normalized_to_original_offsets"]: issues.append(ValidationIssue("offset_map_mismatch", "offset map does not match normalization", "normalized_to_original_offsets"))
        if record.get("normalized_sha256") != sha256_text(record["normalized_text"]): issues.append(ValidationIssue("checksum_mismatch", "normalized SHA-256 does not match text", "normalized_sha256"))
    return issues


def validate_provenance(record: Mapping[str, Any]) -> list[ValidationIssue]:
    issues = []
    for key in ("document_id", "source_id", "source_category", "acquisition_date", "license", "license_verification_status", "uri", "checksum", "collector_id"):
        if not _nonempty(record.get(key)): issues.append(ValidationIssue("missing_field", f"{key} is required", key))
    if record.get("license_verification_status") not in LICENSE_STATUSES: issues.append(ValidationIssue("invalid_enum", "invalid license verification status", "license_verification_status"))
    try: date.fromisoformat(record.get("acquisition_date", ""))
    except (TypeError, ValueError): issues.append(ValidationIssue("invalid_date", "acquisition_date must be ISO-8601", "acquisition_date"))
    if _nonempty(record.get("uri")) and not re.match(r"^(https?://|urn:)", record["uri"]): issues.append(ValidationIssue("invalid_uri", "uri must be http(s) or URN", "uri"))
    if _nonempty(record.get("checksum")) and not re.fullmatch(r"[0-9a-f]{64}", record["checksum"]): issues.append(ValidationIssue("invalid_checksum", "checksum must be SHA-256 hex", "checksum"))
    return issues


def _is_timestamp(value: object) -> bool:
    """Accept only timezone-bearing ISO-8601 timestamps used by the ledger."""
    if not _nonempty(value) or "T" not in value:
        return False
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def validate_collection_candidate(record: Mapping[str, Any]) -> list[ValidationIssue]:
    """Validate a Stage 3A collection-ledger record without collecting data."""
    issues: list[ValidationIssue] = []
    for key in ("candidate_id", "source_id", "canonical_uri", "source_category", "language", "collection_status"):
        if not _nonempty(record.get(key)):
            issues.append(ValidationIssue("missing_field", f"{key} is required", key))
    if record.get("language") != "en":
        issues.append(ValidationIssue("invalid_language", "Dataset 1 candidates must be English", "language"))
    if record.get("source_category") not in SOURCE_CATEGORIES:
        issues.append(ValidationIssue("invalid_enum", "invalid source category", "source_category"))
    if _nonempty(record.get("canonical_uri")) and not re.match(r"^(https?://|urn:)", str(record["canonical_uri"])):
        issues.append(ValidationIssue("invalid_uri", "canonical_uri must be http(s) or URN", "canonical_uri"))
    discovery = record.get("discovery")
    if not isinstance(discovery, Mapping):
        issues.append(ValidationIssue("invalid_discovery", "discovery must be an object", "discovery"))
    else:
        for key in ("method", "route", "discovered_at", "discovered_by"):
            if not _nonempty(discovery.get(key)):
                issues.append(ValidationIssue("missing_field", f"discovery.{key} is required", f"discovery.{key}"))
        if _nonempty(discovery.get("discovered_at")) and not _is_timestamp(discovery["discovered_at"]):
            issues.append(ValidationIssue("invalid_timestamp", "discovery timestamp must include a timezone", "discovery.discovered_at"))
    identity = record.get("source_identity")
    if not isinstance(identity, Mapping):
        issues.append(ValidationIssue("invalid_source_identity", "source_identity must be an object", "source_identity"))
    else:
        for key in ("identity_id", "display_name", "identity_basis"):
            if not _nonempty(identity.get(key)):
                issues.append(ValidationIssue("invalid_source_identity", f"source_identity.{key} is required", f"source_identity.{key}"))
        if identity.get("identity_basis") not in {"publisher_domain", "official_organization", "licensed_archive", "other_documented"}:
            issues.append(ValidationIssue("invalid_source_identity", "invalid source identity basis", "source_identity.identity_basis"))
    independence = record.get("independence")
    if not isinstance(independence, Mapping):
        issues.append(ValidationIssue("invalid_independence", "independence must be an object", "independence"))
    else:
        relationship = independence.get("relationship")
        if relationship not in INDEPENDENCE_RELATIONSHIPS:
            issues.append(ValidationIssue("invalid_independence", "invalid independence relationship", "independence.relationship"))
        if relationship in {"syndicated_or_republished", "copy_or_mirror"}:
            if not _nonempty(independence.get("origin_uri")) or not _nonempty(independence.get("origin_source_identity")):
                issues.append(ValidationIssue("missing_origin", "syndicated or copied material must identify its origin", "independence"))
    publication = record.get("publication")
    if not isinstance(publication, Mapping):
        issues.append(ValidationIssue("invalid_publication", "publication must be an object", "publication"))
    else:
        precision, value = publication.get("precision"), publication.get("value")
        if precision == "date":
            try:
                date.fromisoformat(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                issues.append(ValidationIssue("invalid_publication", "date precision requires ISO date", "publication.value"))
        elif precision == "timestamp":
            if not _is_timestamp(value):
                issues.append(ValidationIssue("invalid_publication", "timestamp precision requires timezone-bearing ISO timestamp", "publication.value"))
        elif precision in {"not_available", "uncertain"}:
            if value is not None:
                issues.append(ValidationIssue("invalid_publication", "unavailable or uncertain publication information must have null value", "publication.value"))
        else:
            issues.append(ValidationIssue("invalid_publication", "invalid publication precision", "publication.precision"))
    collection = record.get("collection")
    if not isinstance(collection, Mapping):
        issues.append(ValidationIssue("invalid_collection", "collection must be an object", "collection"))
    else:
        if not _is_timestamp(collection.get("collected_at")):
            issues.append(ValidationIssue("invalid_timestamp", "collection timestamp must include a timezone", "collection.collected_at"))
        if not _nonempty(collection.get("collector_id")):
            issues.append(ValidationIssue("missing_field", "collection.collector_id is required", "collection.collector_id"))
        digest, length = collection.get("content_sha256"), collection.get("content_length")
        if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            issues.append(ValidationIssue("invalid_checksum", "content_sha256 must be SHA-256 hex or null", "collection.content_sha256"))
        if length is not None and (not isinstance(length, int) or isinstance(length, bool) or length < 0):
            issues.append(ValidationIssue("invalid_content_length", "content_length must be a non-negative integer or null", "collection.content_length"))
    license_status = record.get("license_verification_status")
    usage = record.get("usage_assessment")
    if license_status not in COLLECTION_LICENSE_STATUSES or usage not in {"permitted", "requires_review", "prohibited", "unknown"}:
        issues.append(ValidationIssue("invalid_enum", "invalid licensing or usage assessment", "license_verification_status"))
    checks = record.get("eligibility_checks")
    if not isinstance(checks, Mapping):
        issues.append(ValidationIssue("invalid_eligibility", "eligibility_checks must be an object", "eligibility_checks"))
    else:
        required_checks = ("english_language", "meaningful_text", "attributable_source", "stable_source_identity", "accessible", "license_reviewed", "source_independence_reviewed")
        for key in required_checks:
            if not isinstance(checks.get(key), bool):
                issues.append(ValidationIssue("invalid_eligibility", f"eligibility_checks.{key} must be boolean", f"eligibility_checks.{key}"))
    status = record.get("collection_status")
    reason, retained = record.get("exclusion_reason"), record.get("retained_document_id")
    if status not in COLLECTION_STATUSES:
        issues.append(ValidationIssue("invalid_enum", "invalid collection status", "collection_status"))
    if status == "excluded":
        if reason not in EXCLUSION_REASONS:
            issues.append(ValidationIssue("missing_exclusion_reason", "excluded candidate needs an explicit exclusion reason", "exclusion_reason"))
        if retained is not None:
            issues.append(ValidationIssue("invalid_retention", "excluded candidate cannot name a retained document", "retained_document_id"))
    else:
        if reason is not None:
            issues.append(ValidationIssue("invalid_exclusion_reason", "eligible or retained candidate cannot have an exclusion reason", "exclusion_reason"))
        required_checks = ("english_language", "meaningful_text", "attributable_source", "stable_source_identity", "accessible", "license_reviewed", "source_independence_reviewed")
        if not isinstance(checks, Mapping) or not all(checks.get(key) is True for key in required_checks):
            issues.append(ValidationIssue("ineligible_candidate", "eligible or retained candidate requires every eligibility check", "eligibility_checks"))
        if status == "retained":
            if not _nonempty(retained):
                issues.append(ValidationIssue("missing_retained_document", "retained candidate must name its document", "retained_document_id"))
            if usage != "permitted" or license_status not in {"verified", "not_applicable"}:
                issues.append(ValidationIssue("invalid_retention", "retained candidate requires permitted, verified licensing", "usage_assessment"))
            if not isinstance(collection, Mapping) or collection.get("access_status") != "accessible" or not _nonempty(collection.get("content_sha256")) or not isinstance(collection.get("content_length"), int) or collection["content_length"] <= 0:
                issues.append(ValidationIssue("invalid_retention", "retained candidate requires accessible, meaningful collected content", "collection"))
        elif retained is not None:
            issues.append(ValidationIssue("invalid_retention", "eligible candidate cannot name a retained document", "retained_document_id"))
    duplicate = record.get("duplicate_review")
    if not isinstance(duplicate, Mapping):
        issues.append(ValidationIssue("invalid_duplicate_review", "duplicate_review must be an object", "duplicate_review"))
    else:
        duplicate_status, canonical_id = duplicate.get("status"), duplicate.get("canonical_candidate_id")
        if duplicate_status not in DUPLICATE_REVIEW_STATUSES:
            issues.append(ValidationIssue("invalid_duplicate_review", "invalid duplicate review status", "duplicate_review.status"))
        if duplicate_status in {"exact_duplicate", "near_duplicate", "same_syndicated_text"} and not _nonempty(canonical_id):
            issues.append(ValidationIssue("missing_canonical_candidate", "duplicate review must identify a canonical candidate", "duplicate_review.canonical_candidate_id"))
        if duplicate_status in {"unique", "not_assessed", "needs_human_review"} and canonical_id is not None:
            issues.append(ValidationIssue("invalid_duplicate_review", "non-duplicate review cannot name a canonical candidate", "duplicate_review.canonical_candidate_id"))
        if status == "retained" and duplicate_status != "unique":
            issues.append(ValidationIssue("invalid_retention", "retained candidate must be the reviewed unique canonical candidate", "duplicate_review.status"))
    return issues


def validate_collection_cross_file_integrity(
    documents: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> list[ValidationIssue]:
    """Connect retained Stage 3A candidates to existing document provenance."""
    issues: list[ValidationIssue] = []
    candidate_ids = [row.get("candidate_id") for row in candidates if _nonempty(row.get("candidate_id"))]
    if len(candidate_ids) != len(set(candidate_ids)):
        issues.append(ValidationIssue("duplicate_candidate_id", "candidate IDs must be unique", "candidates"))
    candidate_by_id = {str(row.get("candidate_id")): row for row in candidates if _nonempty(row.get("candidate_id"))}
    for index, row in enumerate(candidates):
        duplicate = row.get("duplicate_review")
        if isinstance(duplicate, Mapping) and _nonempty(duplicate.get("canonical_candidate_id")):
            canonical_id = str(duplicate["canonical_candidate_id"])
            if canonical_id == row.get("candidate_id") or canonical_id not in candidate_by_id:
                issues.append(ValidationIssue("invalid_canonical_candidate", "duplicate review must refer to a different known candidate", f"candidates[{index}].duplicate_review.canonical_candidate_id"))
    if not documents:
        return issues
    documents_by_id = {str(row.get("document_id")): row for row in documents if _nonempty(row.get("document_id"))}
    provenance_by_doc = {str(row.get("document_id")): row for row in provenance if _nonempty(row.get("document_id"))}
    retained = [row for row in candidates if row.get("collection_status") == "retained"]
    retained_by_doc: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(retained):
        document_id = row.get("retained_document_id")
        if document_id not in documents_by_id:
            issues.append(ValidationIssue("orphan_retained_candidate", "retained candidate references unknown document", f"candidates[{index}].retained_document_id"))
            continue
        retained_by_doc.setdefault(str(document_id), []).append(row)
        document = documents_by_id[str(document_id)]
        if row.get("source_id") != document.get("source_id"):
            issues.append(ValidationIssue("candidate_source_mismatch", "retained candidate source_id must match document", f"candidates[{index}].source_id"))
        provenance_row = provenance_by_doc.get(str(document_id))
        collection = row.get("collection")
        if provenance_row is not None and isinstance(collection, Mapping) and collection.get("content_sha256") != provenance_row.get("checksum"):
            issues.append(ValidationIssue("candidate_checksum_mismatch", "retained candidate checksum must match provenance checksum", f"candidates[{index}].collection.content_sha256"))
    for document_id in documents_by_id:
        rows = retained_by_doc.get(document_id, [])
        if not rows:
            issues.append(ValidationIssue("missing_collection_candidate", "document has no retained collection candidate", "candidates"))
        elif len(rows) > 1:
            issues.append(ValidationIssue("duplicate_retained_candidate", "document has multiple retained collection candidates", "candidates"))
    return issues


def _abstention_issues(items: object, claim_ids: set[str], path: str = "abstentions") -> list[ValidationIssue]:
    if not isinstance(items, list): return [ValidationIssue("invalid_abstention", "abstentions must be a list", path)]
    issues = []; seen = set()
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        if not isinstance(item, Mapping): issues.append(ValidationIssue("invalid_abstention", "abstention must be an object", item_path)); continue
        scope, reason, claim_id = item.get("scope"), item.get("reason"), item.get("claim_id")
        if scope not in ABSTENTION_SCOPES: issues.append(ValidationIssue("invalid_abstention", "invalid abstention scope", item_path))
        if reason not in ABSTENTION_REASONS: issues.append(ValidationIssue("invalid_abstention", "invalid abstention reason", item_path))
        if scope == "claim" and claim_id not in claim_ids: issues.append(ValidationIssue("invalid_abstention", "claim abstention must reference a claim", item_path))
        if scope == "document" and claim_id is not None: issues.append(ValidationIssue("invalid_abstention", "document abstention cannot name a claim", item_path))
        key = (scope, claim_id)
        if key in seen: issues.append(ValidationIssue("duplicate_abstention", "duplicate abstention scope", item_path))
        seen.add(key)
    return issues


def _controlled_structure_issues(key: str, value: object, text: str, path: str) -> list[ValidationIssue]:
    if not isinstance(value, Mapping): return [ValidationIssue("invalid_structure", "structure must be an object", path)]
    issues = _span_issues(value.get("source_span"), text, f"{path}.source_span")
    required = {"temporal_annotations": ("value", "normalization"), "quantity_annotations": ("value", "unit", "normalized_unit"), "location_annotations": ("text", "normalized_name"), "modifiers": ("modifier_type", "value"), "negation": ("value",), "modality": ("value",), "non_claim_spans": ("reason",)}
    for field in required.get(key, ()): 
        if field not in value: issues.append(ValidationIssue("missing_field", f"{field} is required", f"{path}.{field}"))
    if key == "temporal_annotations" and value.get("normalization") not in TEMPORAL_NORMALIZATIONS: issues.append(ValidationIssue("invalid_enum", "invalid temporal normalization", path))
    if key == "quantity_annotations" and (not isinstance(value.get("value"), (int, float)) or isinstance(value.get("value"), bool) or not isinstance(value.get("unit"), (str, type(None))) or not isinstance(value.get("normalized_unit"), (str, type(None)))): issues.append(ValidationIssue("invalid_quantity", "quantity requires numeric value and string/null units", path))
    if key == "location_annotations" and (not _nonempty(value.get("text")) or not isinstance(value.get("normalized_name"), (str, type(None)))): issues.append(ValidationIssue("invalid_location", "location requires text and normalized_name", path))
    if key == "modifiers" and (value.get("modifier_type") not in MODIFIER_TYPES or not _nonempty(value.get("value"))): issues.append(ValidationIssue("invalid_modifier", "invalid modifier type or value", path))
    if key == "negation" and not isinstance(value.get("value"), bool): issues.append(ValidationIssue("invalid_negation", "negation value must be boolean", path))
    if key == "modality" and value.get("value") not in MODALITIES: issues.append(ValidationIssue("invalid_modality", "invalid modality", path))
    if key == "non_claim_spans" and value.get("reason") not in NON_CLAIM_REASONS: issues.append(ValidationIssue("invalid_non_claim", "invalid non-claim reason", path))
    return issues


def _structure_issues(claim: Mapping[str, Any], text: str, path: str) -> list[ValidationIssue]:
    issues = []
    entities = claim.get("entities", [])
    if not isinstance(entities, list): issues.append(ValidationIssue("invalid_structure", "entities must be a list", f"{path}.entities"))
    else:
        for i, value in enumerate(entities):
            item = f"{path}.entities[{i}]"
            if not isinstance(value, Mapping) or value.get("entity_type") not in ENTITY_TYPES: issues.append(ValidationIssue("invalid_enum", "invalid EntityType", item))
            else: issues.extend(_span_issues(value.get("source_span"), text, f"{item}.source_span"))
    relations = claim.get("relations", [])
    if not isinstance(relations, list): issues.append(ValidationIssue("invalid_structure", "relations must be a list", f"{path}.relations"))
    else:
        for i, value in enumerate(relations):
            item = f"{path}.relations[{i}]"
            if not isinstance(value, Mapping): issues.append(ValidationIssue("invalid_structure", "relation must be an object", item)); continue
            if not _nonempty(value.get("relation_type")) or not _nonempty(value.get("predicate")): issues.append(ValidationIssue("invalid_relation", "relation_type and predicate are required", item))
            issues.extend(_span_issues(value.get("subject_span"), text, f"{item}.subject_span")); issues.extend(_span_issues(value.get("object_span"), text, f"{item}.object_span"))
            supports = value.get("support_spans")
            if not isinstance(supports, list) or not supports: issues.append(ValidationIssue("missing_support", "relation requires support_spans", item))
            else:
                for j, span in enumerate(supports): issues.extend(_span_issues(span, text, f"{item}.support_spans[{j}]"))
            modality = value.get("modality")
            if not isinstance(value.get("negated", False), bool) or not isinstance(value.get("qualifiers", []), list) or not (modality is None or modality in MODALITIES):
                issues.append(ValidationIssue("invalid_relation", "relation fields have invalid types", item))
    for key in ("temporal_annotations", "quantity_annotations", "location_annotations", "modifiers", "non_claim_spans"):
        values = claim.get(key, [])
        if not isinstance(values, list): issues.append(ValidationIssue("invalid_structure", f"{key} must be a list", f"{path}.{key}"))
        else:
            for i, value in enumerate(values): issues.extend(_controlled_structure_issues(key, value, text, f"{path}.{key}[{i}]"))
    for key in ("negation", "modality"):
        value = claim.get(key)
        if value is not None: issues.extend(_controlled_structure_issues(key, value, text, f"{path}.{key}"))
    return issues


def validate_claim_annotation(record: Mapping[str, Any], document: Mapping[str, Any] | None = None, *, require_original_spans: bool = True) -> list[ValidationIssue]:
    issues = []; normalized = record.get("normalized_text"); original = None; offsets = None
    if document is not None:
        if record.get("document_id") != document.get("document_id"): issues.append(ValidationIssue("document_mismatch", "annotation document_id does not match document", "document_id"))
        if normalized != document.get("normalized_text"): issues.append(ValidationIssue("normalized_text_mismatch", "annotation normalized_text must equal document normalized_text", "normalized_text"))
        normalized, original, offsets = document.get("normalized_text"), document.get("original_text"), document.get("normalized_to_original_offsets")
    if not _nonempty(record.get("document_id")) or not _nonempty(record.get("annotation_id")) or not _nonempty(record.get("annotator_id")): issues.append(ValidationIssue("missing_field", "document_id, annotation_id, annotator_id are required", ""))
    if not isinstance(normalized, str): return issues + [ValidationIssue("missing_text", "normalized text is required", "normalized_text")]
    claims = record.get("claims")
    if not isinstance(claims, list): return issues + [ValidationIssue("invalid_claims", "claims must be a list", "claims")]
    claim_ids = set(); bounds = set()
    for i, claim in enumerate(claims):
        path = f"claims[{i}]"
        if not isinstance(claim, Mapping): issues.append(ValidationIssue("invalid_claim", "claim must be an object", path)); continue
        claim_id = claim.get("claim_id")
        if not _nonempty(claim_id) or claim_id in claim_ids: issues.append(ValidationIssue("duplicate_claim" if claim_id in claim_ids else "missing_field", "claim_id must be unique", f"{path}.claim_id"))
        else: claim_ids.add(claim_id)
        span = claim.get("source_span"); issues.extend(_span_issues(span, normalized, f"{path}.source_span"))
        if isinstance(span, Mapping) and isinstance(span.get("start"), int) and isinstance(span.get("end"), int):
            pair = (span["start"], span["end"])
            if pair in bounds: issues.append(ValidationIssue("duplicate_claim", "duplicate claim source span", f"{path}.source_span"))
            bounds.add(pair)
        if claim.get("claim_type") not in CLAIM_TYPES: issues.append(ValidationIssue("invalid_enum", "invalid ClaimType", f"{path}.claim_type"))
        for key in ("is_atomic", "is_verifiable_as_textual_claim"):
            if not isinstance(claim.get(key), (bool, type(None))): issues.append(ValidationIssue("invalid_claim", f"{key} must be boolean or null", f"{path}.{key}"))
        if require_original_spans:
            raw_span = claim.get("original_source_span")
            if not isinstance(original, str) or not isinstance(offsets, list): issues.append(ValidationIssue("missing_original_provenance", "document original text and offset map are required", path))
            else:
                issues.extend(_span_issues(raw_span, original, f"{path}.original_source_span"))
                if isinstance(span, Mapping) and isinstance(raw_span, Mapping) and all(isinstance(span.get(k), int) for k in ("start", "end")):
                    try:
                        if (raw_span.get("start"), raw_span.get("end")) != (offsets[span["start"]], offsets[span["end"]]): issues.append(ValidationIssue("offset_provenance_mismatch", "original span must map from normalized span", path))
                    except (IndexError, TypeError): pass
                original_claim = claim.get("original_text")
                if isinstance(raw_span, Mapping) and original_claim != raw_span.get("text"):
                    issues.append(ValidationIssue("original_text_mismatch", "original_text must equal original source span text", f"{path}.original_text"))
                elif isinstance(original_claim, str):
                    mapped_original = None
                    try:
                        if isinstance(span, Mapping) and isinstance(offsets, list) and all(isinstance(span.get(k), int) for k in ("start", "end")):
                            mapped_original = original[offsets[span["start"]]:offsets[span["end"]]]
                    except (IndexError, TypeError):
                        mapped_original = None
                    if mapped_original is not None and original_claim != mapped_original:
                        issues.append(ValidationIssue("original_text_mismatch", "original_text must equal the original slice of the normalized span", f"{path}.original_text"))
                    elif mapped_original is None and original_claim not in original:
                        issues.append(ValidationIssue("original_text_mismatch", "original_text must be a verbatim slice of document original_text", f"{path}.original_text"))
        issues.extend(_structure_issues(claim, normalized, path))
    issues.extend(_abstention_issues(record.get("abstentions", []), claim_ids))
    return issues


ADJUDICATION_STATUSES = {"adjudicated", "abstained", "needs_review"}


def validate_adjudication(record: Mapping[str, Any], document: Mapping[str, Any] | None = None) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for key in ("adjudication_id", "document_id", "annotator_a_annotation_id", "annotator_b_annotation_id", "adjudicator_id", "adjudication_version", "rationale"):
        if not _nonempty(record.get(key)):
            issues.append(ValidationIssue("missing_field", f"{key} is required", key))
    if record.get("status") not in ADJUDICATION_STATUSES:
        issues.append(ValidationIssue("invalid_enum", "invalid adjudication status", "status"))
    if record.get("annotator_a_annotation_id") and record.get("annotator_a_annotation_id") == record.get("annotator_b_annotation_id"):
        issues.append(ValidationIssue("annotator_independence", "A/B annotation IDs must differ", "annotator_b_annotation_id"))
    if document is not None and record.get("document_id") != document.get("document_id"):
        issues.append(ValidationIssue("document_mismatch", "adjudication document_id does not match document", "document_id"))
    gold = record.get("gold_annotation")
    if gold is not None:
        issues.extend(validate_claim_annotation(gold, document))
        if isinstance(gold, Mapping) and gold.get("document_id") != record.get("document_id"):
            issues.append(ValidationIssue("document_mismatch", "gold annotation document_id does not match adjudication", "gold_annotation.document_id"))
    elif record.get("status") == "adjudicated":
        issues.append(ValidationIssue("missing_field", "adjudicated records require gold_annotation", "gold_annotation"))
    return issues


def exact_duplicate_ids(documents: Iterable[Mapping[str, Any]]) -> set[str]:
    counts = Counter(str(row.get("document_id", "")) for row in documents); return {key for key, value in counts.items() if key and value > 1}


def exact_hash_duplicates(documents: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in documents:
        if _nonempty(row.get("normalized_sha256")) and _nonempty(row.get("document_id")): grouped[row["normalized_sha256"]].append(row["document_id"])
    return {key: value for key, value in grouped.items() if len(value) > 1}


def normalized_text_duplicates(documents: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in documents:
        if isinstance(row.get("normalized_text"), str) and _nonempty(row.get("document_id")): grouped[sha256_text(row["normalized_text"])].append(row["document_id"])
    return {key: value for key, value in grouped.items() if len(value) > 1}


def near_duplicate_pairs(documents: Sequence[Mapping[str, Any]], threshold: float = 0.9) -> list[tuple[str, str, float]]:
    if not 0 < threshold <= 1: raise ValueError("threshold must be in (0, 1]")
    values = [(row["document_id"], set(re.findall(r"\w+", row.get("normalized_text", "").casefold()))) for row in documents if _nonempty(row.get("document_id"))]
    pairs = []
    for index, (left_id, left) in enumerate(values):
        for right_id, right in values[index + 1:]:
            score = len(left & right) / len(left | right) if left or right else 1.0
            if score >= threshold: pairs.append((left_id, right_id, score))
    return pairs


def validate_story_cluster_records(documents: Sequence[Mapping[str, Any]], clusters: Sequence[Mapping[str, Any]]) -> list[ValidationIssue]:
    if not documents: return [ValidationIssue("dataset_not_populated", "dataset not populated; clusters are not evaluated")]
    issues = []; document_ids = {row.get("document_id") for row in documents}; assigned = {}
    for index, row in enumerate(clusters):
        path = f"clusters[{index}]"; document_id = row.get("document_id"); cluster = row.get("story_cluster_id")
        if document_id not in document_ids: issues.append(ValidationIssue("unknown_document", "cluster record references unknown document", path))
        elif document_id in assigned: issues.append(ValidationIssue("duplicate_cluster_membership", "document has multiple cluster records", path))
        else: assigned[document_id] = cluster
        if not _nonempty(cluster): issues.append(ValidationIssue("invalid_story_cluster_id", "story_cluster_id is required", path))
        if not _nonempty(row.get("cluster_version")): issues.append(ValidationIssue("missing_cluster_version", "cluster_version is required", path))
    missing = document_ids - set(assigned)
    if missing: issues.append(ValidationIssue("missing_cluster_membership", f"{len(missing)} documents have no cluster record", "clusters"))
    for document in documents:
        embedded = document.get("story_cluster_id"); canonical = assigned.get(document.get("document_id"))
        if embedded is not None and embedded != canonical: issues.append(ValidationIssue("cluster_disagreement", "document cluster disagrees with canonical cluster record", str(document.get("document_id"))))
    return issues


def validate_split_records(documents: Sequence[Mapping[str, Any]], split_rows: Mapping[str, Sequence[Mapping[str, Any]]], clusters: Sequence[Mapping[str, Any]] = ()) -> list[ValidationIssue]:
    if not documents: return [ValidationIssue("dataset_not_populated", "dataset not populated; split targets are not evaluated")]
    issues = []; by_id = {row.get("document_id"): row for row in documents if _nonempty(row.get("document_id"))}
    if len(documents) != 230: issues.append(ValidationIssue("document_count", "populated corpus requires exactly 230 documents", "documents"))
    if len(by_id) != len(documents): issues.append(ValidationIssue("duplicate_document_id", "documents contain duplicate or invalid IDs", "documents"))
    if not clusters:
        issues.append(ValidationIssue("missing_canonical_clusters", "populated datasets require canonical cluster records", "clusters"))
    canonical = {row.get("document_id"): row.get("story_cluster_id") for row in clusters if _nonempty(row.get("document_id"))}
    seen = {}; cluster_splits = {}; metadata = set()
    for split, target in TARGET_SPLIT_COUNTS.items():
        rows = split_rows.get(split, [])
        if len(rows) != target: issues.append(ValidationIssue("split_count", f"{split} requires exactly {target} documents", split))
        for index, row in enumerate(rows):
            path = f"{split}[{index}]"; doc_id = row.get("document_id")
            if doc_id not in by_id: issues.append(ValidationIssue("unknown_document", "split references unknown document", path)); continue
            if doc_id in seen: issues.append(ValidationIssue("duplicate_document_id", f"{doc_id} appears in {seen[doc_id]} and {split}", path))
            seen[doc_id] = split; cluster = row.get("story_cluster_id")
            if cluster != canonical.get(doc_id): issues.append(ValidationIssue("cluster_disagreement", "split cluster must equal canonical cluster record", path))
            if cluster in cluster_splits and cluster_splits[cluster] != split: issues.append(ValidationIssue("cluster_leakage", f"cluster {cluster} crosses splits", path))
            cluster_splits[cluster] = split
            split_version, assignment_seed = row.get("split_version"), row.get("assignment_seed")
            if not isinstance(split_version, str) or not isinstance(assignment_seed, str):
                issues.append(ValidationIssue("invalid_split_metadata", "split_version and assignment_seed must be strings", path))
            else:
                metadata.add((split_version, assignment_seed))
            if split == "frozen_test" and by_id[doc_id].get("is_synthetic") is True: issues.append(ValidationIssue("synthetic_frozen_test", "synthetic documents are forbidden in frozen test", path))
    if set(seen) != set(by_id): issues.append(ValidationIssue("unassigned_document", "every document must appear in exactly one split", "splits"))
    if len(metadata) != 1 or next(iter(metadata), (None, None))[0] != SPLIT_VERSION: issues.append(ValidationIssue("nondeterministic_split_metadata", "split metadata must have one version and seed", "splits"))
    return issues


def assign_splits(documents: Sequence[Mapping[str, Any]], clusters: Sequence[Mapping[str, Any]], seed: str) -> dict[str, list[dict[str, Any]]]:
    """Deterministically assign manually clustered documents; never invent clusters."""
    if not isinstance(seed, str) or not seed:
        raise TypeError("assign_splits requires a non-empty string seed")
    if validate_story_cluster_records(documents, clusters): raise ValueError("canonical cluster records are incomplete or invalid")
    if len(documents) != 230: raise ValueError("exactly 230 documents are required")
    canonical = {row["document_id"]: row["story_cluster_id"] for row in clusters}; memberships: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for document in documents: memberships[canonical[document["document_id"]]].append(document)
    remaining = dict(TARGET_SPLIT_COUNTS); result = {name: [] for name in TARGET_SPLIT_COUNTS}
    for cluster, members in sorted(memberships.items(), key=lambda item: sha256_text(f"{seed}:{item[0]}")):
        choices = [name for name in TARGET_SPLIT_COUNTS if remaining[name] >= len(members)]
        if not choices: raise ValueError("cluster sizes cannot satisfy exact split targets")
        split = max(choices, key=lambda name: (remaining[name], name)); remaining[split] -= len(members)
        result[split].extend({"document_id": member["document_id"], "story_cluster_id": cluster, "split_version": SPLIT_VERSION, "assignment_seed": seed} for member in members)
    if any(remaining.values()): raise ValueError("cluster sizes cannot satisfy exact split targets")
    return result


def map_normalized_span_to_original(document: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
    """Map a normalized half-open span onto original_text using the offset map.

    Never compares original_text to normalized_text. The returned text is the
    original slice, which may differ after Unicode or newline normalization.
    """
    original = document.get("original_text")
    offsets = document.get("normalized_to_original_offsets")
    if not isinstance(original, str) or not isinstance(offsets, list):
        raise ValueError("document original text and offset map are required")
    original_start = offsets[start]
    original_end = offsets[end]
    return {"start": original_start, "end": original_end, "text": original[original_start:original_end]}


def validate_cross_file_integrity(
    documents: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
    annotator_a: Sequence[Mapping[str, Any]],
    annotator_b: Sequence[Mapping[str, Any]],
    adjudications: Sequence[Mapping[str, Any]],
) -> list[ValidationIssue]:
    if not documents:
        return []
    issues: list[ValidationIssue] = []
    document_ids = [row.get("document_id") for row in documents if _nonempty(row.get("document_id"))]
    document_id_set = set(document_ids)
    provenance_by_doc: dict[str, int] = {}
    for index, row in enumerate(provenance):
        doc_id = row.get("document_id")
        path = f"provenance[{index}]"
        if doc_id not in document_id_set:
            issues.append(ValidationIssue("orphan_provenance", "provenance references unknown document", path))
            continue
        if doc_id in provenance_by_doc:
            issues.append(ValidationIssue("duplicate_provenance", "document has multiple provenance records", path))
        provenance_by_doc[str(doc_id)] = index
        document = next((item for item in documents if item.get("document_id") == doc_id), None)
        if document is not None and row.get("source_id") != document.get("source_id"):
            issues.append(ValidationIssue("provenance_source_mismatch", "provenance source_id must match document", path))
    missing_prov = sorted(document_id_set - set(provenance_by_doc))
    if missing_prov:
        issues.append(ValidationIssue("missing_provenance", f"{len(missing_prov)} documents have no provenance record", "provenance"))

    def _index_annotations(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
        indexed: dict[str, Mapping[str, Any]] = {}
        seen_ids: set[str] = set()
        annotated: set[str] = set()
        identities: set[str] = set()
        for index, row in enumerate(rows):
            path = f"{label}[{index}]"
            doc_id = row.get("document_id")
            annotation_id = row.get("annotation_id")
            annotator_id = row.get("annotator_id")
            if isinstance(annotator_id, str):
                identities.add(annotator_id)
            if isinstance(annotation_id, str):
                if annotation_id in seen_ids:
                    issues.append(ValidationIssue("duplicate_annotation_id", f"duplicate {label} annotation ID", path))
                seen_ids.add(annotation_id)
                indexed[annotation_id] = row
            if isinstance(doc_id, str):
                annotated.add(doc_id)
                if doc_id not in document_id_set:
                    issues.append(ValidationIssue("unknown_document", f"{label} references unknown document", path))
        if annotated != document_id_set:
            issues.append(ValidationIssue("incomplete_double_annotation", f"{label} does not annotate every document", label))
        return indexed, identities

    a_by_id, a_identities = _index_annotations(annotator_a, "annotator_a")
    b_by_id, b_identities = _index_annotations(annotator_b, "annotator_b")
    shared_identities = a_identities & b_identities
    if shared_identities:
        issues.append(ValidationIssue("annotator_independence", "the same annotator identity cannot be used for both A and B", "annotator_id"))
    a_by_doc = {row.get("document_id"): row for row in annotator_a if isinstance(row.get("document_id"), str)}
    b_by_doc = {row.get("document_id"): row for row in annotator_b if isinstance(row.get("document_id"), str)}
    for doc_id in document_id_set:
        left, right = a_by_doc.get(doc_id), b_by_doc.get(doc_id)
        if left and right and left.get("annotator_id") == right.get("annotator_id") and _nonempty(left.get("annotator_id")):
            issues.append(ValidationIssue("annotator_independence", "A/B annotator identities must differ for each document", str(doc_id)))

    adjudicated_docs: set[str] = set()
    for index, row in enumerate(adjudications):
        path = f"adjudications[{index}]"
        doc_id = row.get("document_id")
        if doc_id not in document_id_set:
            issues.append(ValidationIssue("unknown_document", "adjudication references unknown document", path))
        else:
            if doc_id in adjudicated_docs:
                issues.append(ValidationIssue("duplicate_adjudication", "document has multiple adjudication records", path))
            adjudicated_docs.add(str(doc_id))
        a_id, b_id = row.get("annotator_a_annotation_id"), row.get("annotator_b_annotation_id")
        a_row, b_row = a_by_id.get(a_id), b_by_id.get(b_id)
        if a_id not in a_by_id:
            issues.append(ValidationIssue("unknown_annotation", "adjudication A annotation_id is not in annotator_a", path))
        elif a_row.get("document_id") != doc_id:
            issues.append(ValidationIssue("adjudication_annotation_mismatch", "adjudication A annotation belongs to a different document", path))
        if b_id not in b_by_id:
            issues.append(ValidationIssue("unknown_annotation", "adjudication B annotation_id is not in annotator_b", path))
        elif b_row.get("document_id") != doc_id:
            issues.append(ValidationIssue("adjudication_annotation_mismatch", "adjudication B annotation belongs to a different document", path))
        if a_row and b_row and a_row.get("annotator_id") == b_row.get("annotator_id") and _nonempty(a_row.get("annotator_id")):
            issues.append(ValidationIssue("annotator_independence", "adjudication links A/B records from the same annotator identity", path))
    missing_adj = document_id_set - adjudicated_docs
    if missing_adj:
        issues.append(ValidationIssue("incomplete_adjudication", f"{len(missing_adj)} documents have no adjudication record", "adjudications"))
    return issues


def run_duplicate_qc(
    documents: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]] = (),
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    *,
    near_duplicate_threshold: float = NEAR_DUPLICATE_REVIEW_THRESHOLD,
) -> dict[str, Any]:
    if not documents:
        return {
            "status": "empty",
            "exact_normalized_duplicates": [],
            "exact_hash_duplicates": [],
            "duplicate_cluster_membership": [],
            "near_duplicate_candidates": [],
            "split_leakage": [],
        }
    exact_text = [{"normalized_sha256": key, "document_ids": value} for key, value in sorted(normalized_text_duplicates(documents).items())]
    exact_hash = [{"normalized_sha256": key, "document_ids": value} for key, value in sorted(exact_hash_duplicates(documents).items())]
    cluster_conflicts = [issue.path for issue in validate_story_cluster_records(documents, clusters) if issue.code == "duplicate_cluster_membership"]
    near = [
        {"left_document_id": left, "right_document_id": right, "score": score, "action": "review_candidate"}
        for left, right, score in near_duplicate_pairs(documents, threshold=near_duplicate_threshold)
    ]
    leakage = []
    if split_rows:
        leakage = [issue.message for issue in validate_split_records(documents, split_rows, clusters) if issue.code == "cluster_leakage"]
    if exact_text or exact_hash or cluster_conflicts:
        status = "fail"
    elif near or leakage:
        status = "needs_review"
    else:
        status = "pass"
    return {
        "status": status,
        "exact_normalized_duplicates": exact_text,
        "exact_hash_duplicates": exact_hash,
        "duplicate_cluster_membership": cluster_conflicts,
        "near_duplicate_candidates": near,
        "split_leakage": leakage,
    }


def completeness_labels(
    documents: Sequence[Mapping[str, Any]],
    issues: Sequence[ValidationIssue],
) -> dict[str, str]:
    if not documents:
        unevaluated = "not_evaluated"
        return {
            "cluster_integrity": unevaluated,
            "provenance_completeness": unevaluated,
            "annotation_completeness": unevaluated,
            "adjudication_completeness": unevaluated,
        }
    codes = {issue.code for issue in issues}

    def _label(invalid: set[str], incomplete: set[str]) -> str:
        if codes & invalid:
            return "invalid" if "cluster" in next(iter(invalid)) else "incomplete"
        if codes & incomplete:
            return "incomplete"
        return "complete"

    cluster_codes = {"unknown_document", "duplicate_cluster_membership", "invalid_story_cluster_id", "cluster_disagreement"}
    cluster_missing = {"missing_cluster_membership", "missing_cluster_version"}
    if codes & cluster_codes:
        cluster = "invalid"
    elif codes & cluster_missing:
        cluster = "incomplete"
    else:
        cluster = "complete"
    provenance = "incomplete" if codes & {"missing_provenance", "orphan_provenance", "duplicate_provenance", "provenance_source_mismatch"} else "complete"
    annotation = "incomplete" if codes & {"incomplete_double_annotation", "annotator_independence", "unknown_annotation"} else "complete"
    adjudication = "incomplete" if codes & {"incomplete_adjudication", "duplicate_adjudication", "adjudication_annotation_mismatch"} else "complete"
    return {
        "cluster_integrity": cluster,
        "provenance_completeness": provenance,
        "annotation_completeness": annotation,
        "adjudication_completeness": adjudication,
    }
