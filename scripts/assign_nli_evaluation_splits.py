"""Deterministic group-aware split feasibility for NLI evaluation gold.

Groups by story_or_event_cluster_id. Rows from the same story stay together.
Source, article, URL, and duplicate-group identifiers must not cross splits.
This tool does not randomly split individual rows.

Ten-story Batch 2 gold is too small for train/dev/test. The default is to print
a feasibility report and refuse to write split files.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.nli_evaluation_common import (
        LABELS,
        LEAKAGE_FIELDS,
        STORY_FIELD,
        annotation_status,
        assert_safe_output_path,
        ensure_repo_on_path,
        is_nonempty_string,
        labeled_class_counts,
        read_jsonl,
        story_groups,
        write_json_report,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from nli_evaluation_common import (
        LABELS,
        LEAKAGE_FIELDS,
        STORY_FIELD,
        annotation_status,
        assert_safe_output_path,
        ensure_repo_on_path,
        is_nonempty_string,
        labeled_class_counts,
        read_jsonl,
        story_groups,
        write_json_report,
    )

ensure_repo_on_path()


SPLIT_VERSION = "nli-eval-story-group-v1"
MIN_STORY_GROUPS = 30
REQUESTED_SPLITS = ("development", "test", "frozen_test")
SPLIT_FILENAMES = {
    "development": "development.jsonl",
    "test": "test.jsonl",
    "frozen_test": "frozen_test.jsonl",
}


def _gold_records(entries: Sequence[tuple[int, object]], errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line, raw in entries:
        if not isinstance(raw, dict):
            errors.append({"source": "gold", "line": line, "message": "record must be a JSON object"})
            continue
        records.append(raw)
    return records


def _story_assignment(story_ids: Sequence[str]) -> dict[str, str]:
    return {
        story_id: REQUESTED_SPLITS[index % len(REQUESTED_SPLITS)]
        for index, story_id in enumerate(story_ids)
    }


def _leakage(records: Sequence[Mapping[str, Any]], sample_to_split: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    leaked = False
    for field in LEAKAGE_FIELDS:
        values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for record in records:
            sample_id = record.get("sample_id")
            value = record.get(field)
            if not is_nonempty_string(sample_id) or not is_nonempty_string(value):
                continue
            split = sample_to_split.get(str(sample_id))
            if split is None:
                continue
            values[str(value)][split].add(str(sample_id))
        collisions = [
            {
                "group_id": group_id,
                "splits": {split: sorted(ids) for split, ids in sorted(split_ids.items())},
            }
            for group_id, split_ids in sorted(values.items())
            if len(split_ids) > 1
        ]
        result[field] = {"collision_group_count": len(collisions), "groups": collisions}
        if collisions:
            leaked = True
    return {"leaked": leaked, "fields": result}


def _split_class_coverage(
    records: Sequence[Mapping[str, Any]],
    sample_to_split: Mapping[str, str],
) -> dict[str, Any]:
    by_split: dict[str, list[Mapping[str, Any]]] = {split: [] for split in REQUESTED_SPLITS}
    for record in records:
        sample_id = record.get("sample_id")
        if not is_nonempty_string(sample_id):
            continue
        split = sample_to_split.get(str(sample_id))
        if split in by_split:
            by_split[split].append(record)
    coverage: dict[str, Any] = {}
    missing: dict[str, list[str]] = {}
    for split in REQUESTED_SPLITS:
        counts = labeled_class_counts(by_split[split])
        present = {label for label, count in counts.items() if count}
        absent = [label for label in LABELS if label not in present]
        coverage[split] = {
            "record_count": len(by_split[split]),
            "labeled_class_counts": counts,
            "missing_labeled_classes": absent,
        }
        if absent:
            missing[split] = absent
    return {"coverage": coverage, "missing": missing}


def assess_splits(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups = story_groups(records)
    story_ids = list(groups)
    group_summaries = []
    for story_id in story_ids:
        rows = groups[story_id]
        sample_ids = [str(row["sample_id"]) for row in rows if is_nonempty_string(row.get("sample_id"))]
        group_summaries.append({
            STORY_FIELD: story_id,
            "sample_ids": sample_ids,
            "sample_count": len(sample_ids),
            "labeled_class_counts": labeled_class_counts(rows),
            "source_ids": sorted({str(row["source_id"]) for row in rows if is_nonempty_string(row.get("source_id"))}),
            "article_ids": sorted({str(row["article_id"]) for row in rows if is_nonempty_string(row.get("article_id"))}),
            "canonical_urls": sorted({str(row["canonical_url"]) for row in rows if is_nonempty_string(row.get("canonical_url"))}),
        })

    limitations = [
        "Metrics and splits apply to NLI textual relation only, not news-verdict accuracy.",
        "Rows that share a story_or_event_cluster_id are not independent samples.",
        f"Writing official development/test/frozen_test files requires at least {MIN_STORY_GROUPS} story groups "
        "and all three labeled classes in every requested split.",
    ]
    base = {
        "split_version": SPLIT_VERSION,
        "min_story_groups": MIN_STORY_GROUPS,
        "requested_splits": list(REQUESTED_SPLITS),
        "story_group_count": len(story_ids),
        "record_count": len(records),
        "labeled_class_counts": labeled_class_counts(records),
        "groups": group_summaries,
        "assignments": None,
        "wrote_files": False,
        "output_files": [],
        "limitations": limitations,
    }

    unlabeled_only = all(annotation_status(record) != "labeled" for record in records)
    if unlabeled_only:
        return {
            **base,
            "status": "invalid_input",
            "reason": "gold has no labeled records",
        }
    if len(story_ids) < MIN_STORY_GROUPS:
        return {
            **base,
            "status": "insufficient_groups",
            "reason": (
                f"This corpus has {len(story_ids)} story groups, below the minimum of {MIN_STORY_GROUPS}. "
                "Generating train/dev/test files would be misleading."
            ),
        }

    sample_to_split: dict[str, str] = {}
    story_to_split = _story_assignment(story_ids)
    for story_id, rows in groups.items():
        split = story_to_split[story_id]
        for row in rows:
            if is_nonempty_string(row.get("sample_id")):
                sample_to_split[str(row["sample_id"])] = split

    leakage = _leakage(records, sample_to_split)
    if leakage["leaked"]:
        return {
            **base,
            "status": "cross_split_leakage",
            "reason": "A source_id, article_id, canonical_url, or duplicate group would appear in more than one split.",
            "assignments": sample_to_split,
            "leakage": leakage["fields"],
        }

    class_coverage = _split_class_coverage(records, sample_to_split)
    if class_coverage["missing"]:
        return {
            **base,
            "status": "insufficient_class_coverage",
            "reason": "Every requested split must contain labeled entailment, contradiction, and neutral records.",
            "assignments": sample_to_split,
            "split_class_coverage": class_coverage["coverage"],
        }

    return {
        **base,
        "status": "ready",
        "reason": "Story-grouped splits satisfy the minimum group count, class coverage, and leakage checks.",
        "assignments": sample_to_split,
        "split_class_coverage": class_coverage["coverage"],
        "leakage": leakage["fields"],
    }


def write_split_files(report: Mapping[str, Any], write_dir: Path) -> dict[str, Any]:
    updated = dict(report)
    if report.get("status") != "ready" or not report.get("assignments"):
        updated["wrote_files"] = False
        updated["output_files"] = []
        return updated
    assignments: Mapping[str, str] = report["assignments"]
    by_split: dict[str, list[dict[str, str]]] = {split: [] for split in REQUESTED_SPLITS}
    for sample_id in sorted(assignments):
        split = assignments[sample_id]
        by_split[split].append({"sample_id": sample_id, "split": split})
    write_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[str] = []
    for split in REQUESTED_SPLITS:
        path = write_dir / SPLIT_FILENAMES[split]
        assert_safe_output_path(path)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in by_split[split]),
            encoding="utf-8",
            newline="\n",
        )
        output_files.append(str(path))
    updated["wrote_files"] = True
    updated["output_files"] = output_files
    return updated


def assign_from_records(
    records: Sequence[Mapping[str, Any]],
    write_dir: Path | None = None,
) -> dict[str, Any]:
    report = assess_splits(records)
    if write_dir is not None:
        if report["status"] != "ready":
            report = dict(report)
            report["wrote_files"] = False
            report["write_refused"] = True
            return report
        return write_split_files(report, write_dir)
    return report


def assign_from_file(gold_path: Path, write_dir: Path | None = None) -> dict[str, Any]:
    entries, errors = read_jsonl(gold_path, "gold")
    if errors:
        return {
            "status": "invalid_input",
            "reason": "gold JSONL failed to parse",
            "validation_errors": errors,
            "story_group_count": 0,
            "min_story_groups": MIN_STORY_GROUPS,
            "requested_splits": list(REQUESTED_SPLITS),
            "assignments": None,
            "wrote_files": False,
            "output_files": [],
        }
    parse_errors: list[dict[str, Any]] = []
    records = _gold_records(entries, parse_errors)
    if parse_errors:
        return {
            "status": "invalid_input",
            "reason": "gold JSONL contains non-object records",
            "validation_errors": parse_errors,
            "story_group_count": 0,
            "min_story_groups": MIN_STORY_GROUPS,
            "requested_splits": list(REQUESTED_SPLITS),
            "assignments": None,
            "wrote_files": False,
            "output_files": [],
        }
    report = assign_from_records(records, write_dir)
    report["gold"] = str(gold_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic story-grouped NLI split feasibility. Does not randomly split rows."
    )
    parser.add_argument("--gold", required=True, help="canonical gold JSONL")
    parser.add_argument("--output", default="-", help="feasibility report JSON path, or - for stdout (default: -)")
    parser.add_argument(
        "--write-dir",
        help="optional directory for development.jsonl, test.jsonl, and frozen_test.jsonl; writes only when feasible",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    write_dir = Path(args.write_dir) if args.write_dir else None
    try:
        report = assign_from_file(Path(args.gold), write_dir)
        write_json_report(report, args.output)
    except ValueError as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot write output: {exc.strerror or exc.__class__.__name__}", file=__import__("sys").stderr)
        return 2
    if report["status"] == "invalid_input":
        return 2
    if write_dir is not None and report["status"] != "ready":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
