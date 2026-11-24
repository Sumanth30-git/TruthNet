from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_nli_evaluation import audit_files, audit_records, main
from scripts.nli_evaluation_common import PROTECTED_BASENAMES, cohen_kappa, write_json_report


REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "data" / "news_evaluation" / "pilot"
SCRIPT_PATH = REPO / "scripts" / "audit_nli_evaluation.py"


def _gold(
    sample_id: str,
    relation: str | None,
    *,
    status: str = "labeled",
    story_id: str = "story-1",
    source_id: str = "source-1",
    article_id: str = "article-1",
    copy_group: str = "copy-1",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "claim_cluster_id": f"claim-{sample_id}",
        "story_or_event_cluster_id": story_id,
        "source_id": source_id,
        "article_id": article_id,
        "canonical_url": f"https://news.example/{article_id}",
        "duplicate_or_syndication_group_id": copy_group,
        "premise": f"Premise for {sample_id}.",
        "hypothesis": f"Hypothesis for {sample_id}.",
        "gold_relation": relation,
        "annotation_provenance": {
            "annotation_method": "human_adjudication",
            "annotation_status": status,
            "annotation_version": "1.0",
            "adjudication_id": f"adjudication-{sample_id}",
        },
        "evaluation_metadata": {"retrieval_origin": "fetched_page"},
    }


def _annotator(sample_id: str, relation: object, annotator: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "gold_relation": relation,
        "annotation_provenance": {
            "annotation_status": "annotated",
            "annotator": annotator,
            "method": "independent_text_only_annotation",
            "source": "ANNOTATION_GUIDE.md",
        },
    }


def _pilot_hashes() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PILOT.iterdir()
        if path.is_file()
    }


def test_duplicate_sample_ids_are_errors() -> None:
    gold = [_gold("s1", "entailment"), _gold("s1", "neutral", source_id="source-2", article_id="article-2")]

    report = audit_records(gold)

    assert report["audit_status"] == "failed"
    assert any(item["code"] == "duplicate_sample_id" for item in report["findings"])


def test_missing_story_and_source_ids_are_errors() -> None:
    gold = [_gold("s1", "entailment")]
    gold[0]["story_or_event_cluster_id"] = " "
    gold[0]["source_id"] = ""

    report = audit_records(gold)

    assert report["audit_status"] == "failed"
    codes = {item["code"] for item in report["findings"]}
    assert "missing_story_or_event_cluster_id" in codes
    assert "missing_source_id" in codes


def test_invalid_gold_status_combination_is_an_error() -> None:
    gold = [_gold("s1", "entailment", status="abstained")]

    report = audit_records(gold)

    assert report["audit_status"] == "failed"
    assert report["gold_label_validity"]["valid_evaluator_gold"] is False
    assert any(item["code"] == "invalid_gold_label" for item in report["findings"])


def test_identical_annotator_files_are_reported_without_inventing_labels() -> None:
    gold = [_gold("s1", "entailment"), _gold("s2", "contradiction", source_id="s2", article_id="a2", copy_group="c2")]
    annotator_a = [_annotator("s1", "entailment", "annotator_a"), _annotator("s2", "contradiction", "annotator_a")]
    annotator_b = [_annotator("s1", "entailment", "annotator_b"), _annotator("s2", "contradiction", "annotator_b")]

    report = audit_records(gold, annotator_a, annotator_b)
    agreement = report["annotator_agreement"]

    assert agreement["identical_except_annotator_name"] is True
    assert agreement["raw_agreement"] == 1.0
    assert agreement["cohen_kappa"] == 1.0
    assert agreement["independence_supported"] is False
    assert any(item["code"] == "identical_annotator_files" for item in report["findings"])
    assert "does not prove independent" in agreement["note"].lower() or "does not prove" in str(report["findings"])


def test_disagreeing_annotators_are_reported() -> None:
    gold = [_gold("s1", "entailment")]
    annotator_a = [_annotator("s1", "entailment", "annotator_a")]
    annotator_b = [_annotator("s1", "neutral", "annotator_b")]

    report = audit_records(gold, annotator_a, annotator_b)

    assert report["annotator_agreement"]["identical_except_annotator_name"] is False
    assert report["annotator_agreement"]["disagreements"][0]["sample_id"] == "s1"
    assert cohen_kappa(["entailment"], ["neutral"]) == 0.0


def test_tiny_canonical_copy_is_valid_evaluator_gold(tmp_path: Path) -> None:
    gold = [
        _gold("s1", "entailment"),
        _gold("s2", "contradiction", source_id="source-2", article_id="article-2", copy_group="copy-2"),
        _gold("s3", "neutral", source_id="source-3", article_id="article-3", copy_group="copy-3"),
        _gold("s4", None, status="abstained", source_id="source-4", article_id="article-4", copy_group="copy-4"),
    ]
    path = tmp_path / "tiny_gold.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in gold), encoding="utf-8", newline="\n")

    report = audit_files(path)

    assert report["audit_status"] == "completed"
    assert report["valid_evaluator_input"] is True
    assert report["class_distribution"]["eligible_labeled_count"] == 3
    assert report["class_distribution"]["excluded_abstained"] == 1
    assert PILOT.joinpath("batch2_evaluation_gold.jsonl").is_file()


def test_real_batch2_gold_audit_is_read_only() -> None:
    before = _pilot_hashes()
    report = audit_files(
        PILOT / "batch2_evaluation_gold.jsonl",
        PILOT / "batch2_annotator_a.jsonl",
        PILOT / "batch2_annotator_b.jsonl",
        PILOT / "batch2_predictions.jsonl",
    )

    assert report["audit_status"] == "completed"
    assert report["valid_evaluator_input"] is True
    assert report["counts"]["gold_records"] == 40
    assert report["annotator_agreement"]["identical_except_annotator_name"] is True
    assert report["annotator_agreement"]["cohen_kappa"] == 1.0
    assert report["duplicate_or_syndication"]["duplicate_or_syndication_group_id"]["encodes_copies"] is False
    assert report["group_diagnostics"]["story_or_event_cluster_id"]["collision_group_count"] == 10
    assert _pilot_hashes() == before


def test_cli_refuses_protected_output_names(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps(_gold("s1", "entailment")) + "\n", encoding="utf-8")
    protected = tmp_path / "batch2_evaluation_gold.jsonl"

    code = main(["--gold", str(gold), "--output", str(protected)])

    assert code == 2
    assert "refusing to write" in capsys.readouterr().err
    assert not protected.exists()
    assert "batch2_evaluation_gold.jsonl" in PROTECTED_BASENAMES


def test_cli_stdout_does_not_modify_pilot_files(capsys: pytest.CaptureFixture[str]) -> None:
    before = _pilot_hashes()
    code = main([
        "--gold",
        str(PILOT / "batch2_evaluation_gold.jsonl"),
        "--annotator-a",
        str(PILOT / "batch2_annotator_a.jsonl"),
        "--annotator-b",
        str(PILOT / "batch2_annotator_b.jsonl"),
        "--output",
        "-",
    ])
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert code == 0
    assert report["audit_status"] == "completed"
    assert _pilot_hashes() == before


def test_write_json_report_refuses_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="overwrite"):
        write_json_report({"ok": True}, str(path))


def test_audit_script_has_no_production_imports() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    assert all(not name.startswith(("backend", "frontend")) for name in imports)
