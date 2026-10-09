"""Dataset 1-only deterministic cluster-disjoint split assignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # Supports both ``python -m`` and direct script execution.
    from scripts.dataset1_common import TARGET_SPLIT_COUNTS, assign_splits, read_jsonl
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI use
    from dataset1_common import TARGET_SPLIT_COUNTS, assign_splits, read_jsonl


def write_splits(documents_path: Path, output_dir: Path, seed: str, clusters_path: Path | None = None) -> dict[str, int]:
    documents = read_jsonl(documents_path)
    if not documents:
        return {name: 0 for name in TARGET_SPLIT_COUNTS}
    if clusters_path is None:
        raise ValueError("--clusters is required when assigning splits for a populated Dataset 1 corpus")
    clusters = read_jsonl(clusters_path)
    splits = assign_splits(documents, clusters, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        (output_dir / f"{split_name}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8", newline="\n"
        )
    return {name: len(rows) for name, rows in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign Dataset 1 cluster-disjoint splits.")
    parser.add_argument("documents", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--clusters", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_splits(args.documents, args.output_dir, args.seed, args.clusters), sort_keys=True))


if __name__ == "__main__":
    main()
