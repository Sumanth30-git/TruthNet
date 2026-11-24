"""Shared helpers for NLI evaluation integrity and split tooling.

These utilities are evaluation-only. They must not be imported by the TruthNet
production runtime. Gold labels describe textual NLI relation, not claim truth.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit


LABELS = ("contradiction", "entailment", "neutral")
LABEL_SET = frozenset(LABELS)
GROUP_FIELDS = (
    "claim_cluster_id",
    "story_or_event_cluster_id",
    "source_id",
    "article_id",
    "canonical_url",
    "duplicate_or_syndication_group_id",
)
STORY_FIELD = "story_or_event_cluster_id"
LEAKAGE_FIELDS = (
    "source_id",
    "article_id",
    "canonical_url",
    "duplicate_or_syndication_group_id",
)
PROTECTED_BASENAMES = frozenset({
    "ANNOTATION_GUIDE.md",
    "EVALUATION.md",
    "batch1.jsonl",
    "batch1_predictions.jsonl",
    "batch2.jsonl",
    "batch2_adjudicated.jsonl",
    "batch2_annotator_a.jsonl",
    "batch2_annotator_b.jsonl",
    "batch2_evaluation_gold.jsonl",
    "batch2_prediction_input.jsonl",
    "batch2_predictions.jsonl",
    "nli_relation.schema.json",
})
NLI_RELATION_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "news_evaluation"
    / "contracts"
    / "nli_relation.schema.json"
)


def ensure_repo_on_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_protected_output_path(path: Path) -> bool:
    return path.name in PROTECTED_BASENAMES


def assert_safe_output_path(path: Path) -> None:
    if is_protected_output_path(path):
        raise ValueError(
            f"refusing to write {path.name}: existing gold, prediction, schema, "
            "and evaluation documentation files must not be overwritten"
        )
    if path.exists():
        raise ValueError(f"refusing to overwrite existing file: {path}")


def annotation_status(record: Mapping[str, Any]) -> str | None:
    provenance = record.get("annotation_provenance")
    if not isinstance(provenance, dict):
        return None
    status = provenance.get("annotation_status")
    return status if isinstance(status, str) else None


def is_labeled(record: Mapping[str, Any]) -> bool:
    return annotation_status(record) == "labeled"


def gold_relation(record: Mapping[str, Any]) -> object:
    return record.get("gold_relation")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric constant {value}")


def read_jsonl(path: Path, source: str) -> tuple[list[tuple[int, object]], list[dict[str, Any]]]:
    entries: list[tuple[int, object]] = []
    errors: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    errors.append({
                        "source": source,
                        "line": line_number,
                        "sample_id": None,
                        "message": "blank lines are not valid JSONL records",
                    })
                    continue
                try:
                    entries.append((line_number, json.loads(line, parse_constant=_reject_json_constant)))
                except json.JSONDecodeError as exc:
                    errors.append({
                        "source": source,
                        "line": line_number,
                        "sample_id": None,
                        "message": f"malformed JSON: {exc.msg}",
                    })
                except ValueError as exc:
                    errors.append({
                        "source": source,
                        "line": line_number,
                        "sample_id": None,
                        "message": f"malformed JSON: {exc}",
                    })
    except OSError as exc:
        errors.append({
            "source": source,
            "line": 0,
            "sample_id": None,
            "message": f"cannot read input file: {exc.strerror or exc.__class__.__name__}",
        })
    return entries, errors


def write_json_report(report: Mapping[str, Any], output: str) -> None:
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output == "-":
        sys.stdout.write(serialized)
        return
    path = Path(output)
    assert_safe_output_path(path)
    path.write_text(serialized, encoding="utf-8", newline="\n")


def collision_groups(records: Sequence[Mapping[str, Any]], fields: Sequence[str] = GROUP_FIELDS) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        values: dict[str, set[str]] = defaultdict(set)
        missing = 0
        for record in records:
            value = record.get(field)
            sample_id = record.get("sample_id")
            if not is_nonempty_string(value):
                missing += 1
                continue
            if is_nonempty_string(sample_id):
                values[str(value)].add(str(sample_id))
        collisions = [
            {"group_id": group_id, "sample_ids": sorted(sample_ids), "count": len(sample_ids)}
            for group_id, sample_ids in sorted(values.items())
            if len(sample_ids) > 1
        ]
        result[field] = {
            "collision_group_count": len(collisions),
            "unique_value_count": len(values),
            "missing_count": missing,
            "singleton_count": sum(1 for sample_ids in values.values() if len(sample_ids) == 1),
            "groups": collisions,
        }
    return result


def story_groups(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        story = record.get(STORY_FIELD)
        if is_nonempty_string(story):
            grouped[str(story)].append(record)
    return {story: grouped[story] for story in sorted(grouped)}


def labeled_class_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        if not is_labeled(record):
            continue
        relation = gold_relation(record)
        if isinstance(relation, str) and relation in LABEL_SET:
            counts[relation] += 1
    return {label: int(counts[label]) for label in LABELS}


def canonical_url_is_http(value: object) -> bool:
    if not is_nonempty_string(value):
        return False
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and not any(char.isspace() for char in value)


def annotator_comparable_record(record: Mapping[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(record, ensure_ascii=False))
    provenance = copied.get("annotation_provenance")
    if isinstance(provenance, dict):
        provenance.pop("annotator", None)
        provenance.pop("annotator_id", None)
        copied["annotation_provenance"] = provenance
    return copied


def raw_agreement(labels_a: Sequence[object], labels_b: Sequence[object]) -> float | None:
    if len(labels_a) != len(labels_b) or not labels_a:
        return None
    return sum(left == right for left, right in zip(labels_a, labels_b)) / len(labels_a)


def cohen_kappa(labels_a: Sequence[object], labels_b: Sequence[object]) -> float | None:
    if len(labels_a) != len(labels_b) or not labels_a:
        return None
    n = len(labels_a)
    observed = sum(left == right for left, right in zip(labels_a, labels_b)) / n
    categories = sorted(set(labels_a) | set(labels_b), key=lambda item: (item is None, str(item)))
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    expected = sum((count_a[category] / n) * (count_b[category] / n) for category in categories)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)
