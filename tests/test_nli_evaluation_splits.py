from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts.assign_nli_evaluation_splits import (
    MIN_STORY_GROUPS,
    REQUESTED_SPLITS,
    assign_from_file,
    assign_from_records,
    main,
)


REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "data" / "news_evaluation" / "pilot"
SCRIPT_PATH = REPO / "scripts" / "assign_nli_evaluation_splits.py"


def _gold(
    sample_id: str,
    relation: str | None,
    *,
    status: str = "labeled",
    story_id: str = "story-1",
    source_id: str = "source-1",
    article_id: str = "article-1",
    copy_group: str | None = None,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "claim_cluster_id": f"claim-{sample_id}",
        "story_or_event_cluster_id": story_id,
        "source_id": source_id,
        "article_id": article_id,
        "canonical_url": f"https://news.example/{article_id}",
        "duplicate_or_syndication_group_id": copy_group or f"dup-{sample_id}",
        "premise": f"Premise for {sample_id}.",
        "hypothesis": f"Hypothesis for {sample_id}.",
        "gold_relation": relation,
        "annotation_provenance": {
            "annotation_method": "human_adjudication",
            "annotation_status": status,
            "annotation_version": "1.0",
            "adjudication_id": f"adjudication-{sample_id}",
        },
    }


def _synthetic_corpus(story_count: int = 30, *, shared_source_pair: bool = False, drop_contradiction_outside_dev: bool = False) -> list[dict[str, object]]:
    labels = ("entailment", "contradiction", "neutral", "entailment")
    records: list[dict[str, object]] = []
    for index in range(story_count):
        story_id = f"story-{index:02d}"
        source_id = f"source-{index:02d}"
        if shared_source_pair and index == 1:
            source_id = "source-00"
        article_id = f"article-{index:02d}"
        for sample_index, label in enumerate(labels):
            if drop_contradiction_outside_dev and label == "contradiction" and index % 3 != 0:
                label = "entailment"
            records.append(
                _gold(
                    f"s-{index:02d}-{sample_index}",
                    label,
                    story_id=story_id,
                    source_id=source_id,
                    article_id=article_id,
                )
            )
    return records


def _pilot_hashes() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PILOT.iterdir()
        if path.is_file()
    }


def test_synthetic_thirty_stories_write_group_closed_splits(tmp_path: Path) -> None:
    records = _synthetic_corpus(30)
    report = assign_from_records(records, write_dir=tmp_path)

    assert report["status"] == "ready"
    assert report["wrote_files"] is True
    assert report["story_group_count"] == 30
    assignments = report["assignments"]
    assert assignments is not None
    for index in range(30):
        sample_ids = [f"s-{index:02d}-{sample_index}" for sample_index in range(4)]
        splits = {assignments[sample_id] for sample_id in sample_ids}
        assert len(splits) == 1
    sources_by_split: dict[str, set[str]] = {split: set() for split in REQUESTED_SPLITS}
    for record in records:
        split = assignments[str(record["sample_id"])]
        sources_by_split[split].add(str(record["source_id"]))
    overlap = sources_by_split["development"] & sources_by_split["test"]
    overlap |= sources_by_split["development"] & sources_by_split["frozen_test"]
    overlap |= sources_by_split["test"] & sources_by_split["frozen_test"]
    assert not overlap
    for split in REQUESTED_SPLITS:
        path = tmp_path / f"{split}.jsonl"
        assert path.is_file()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows
        assert {row["split"] for row in rows} == {split}


def test_assignment_is_deterministic() -> None:
    records = _synthetic_corpus(30)
    first = assign_from_records(records)
    second = assign_from_records(list(reversed(records)))

    assert first["status"] == "ready"
    assert first["assignments"] == second["assignments"]


def test_shared_source_across_stories_refuses_write(tmp_path: Path) -> None:
    records = _synthetic_corpus(30, shared_source_pair=True)
    report = assign_from_records(records, write_dir=tmp_path)

    assert report["status"] == "cross_split_leakage"
    assert report["wrote_files"] is False
    assert report.get("write_refused") is True
    assert list(tmp_path.iterdir()) == []


def test_missing_class_in_a_split_refuses_write(tmp_path: Path) -> None:
    records = _synthetic_corpus(30, drop_contradiction_outside_dev=True)
    report = assign_from_records(records, write_dir=tmp_path)

    assert report["status"] == "insufficient_class_coverage"
    assert report["wrote_files"] is False
    assert list(tmp_path.iterdir()) == []


def test_real_batch2_gold_is_insufficient_and_writes_nothing(tmp_path: Path) -> None:
    before = _pilot_hashes()
    write_dir = tmp_path / "splits"
    report = assign_from_file(PILOT / "batch2_evaluation_gold.jsonl", write_dir=write_dir)

    assert report["status"] == "insufficient_groups"
    assert report["story_group_count"] == 10
    assert report["min_story_groups"] == MIN_STORY_GROUPS
    assert report["wrote_files"] is False
    assert not write_dir.exists()
    assert _pilot_hashes() == before


def test_cli_batch2_stdout_does_not_write_splits(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    before = _pilot_hashes()
    code = main([
        "--gold",
        str(PILOT / "batch2_evaluation_gold.jsonl"),
        "--output",
        "-",
        "--write-dir",
        str(tmp_path / "out"),
    ])
    report = json.loads(capsys.readouterr().out)

    assert code == 1
    assert report["status"] == "insufficient_groups"
    assert report["wrote_files"] is False
    assert list(tmp_path.iterdir()) == []
    assert _pilot_hashes() == before


def test_cli_without_write_dir_leaves_disk_untouched(capsys: pytest.CaptureFixture[str]) -> None:
    before = _pilot_hashes()
    code = main(["--gold", str(PILOT / "batch2_evaluation_gold.jsonl"), "--output", "-"])
    report = json.loads(capsys.readouterr().out)

    assert code == 0
    assert report["status"] == "insufficient_groups"
    assert _pilot_hashes() == before


def test_split_script_has_no_production_imports() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    assert all(not name.startswith(("backend", "frontend")) for name in imports)
