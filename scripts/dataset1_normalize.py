"""CLI and API for deterministic Dataset 1 document normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # Supports both ``python -m`` and direct script execution.
    from scripts.dataset1_common import normalize_document, read_jsonl
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI use
    from dataset1_common import normalize_document, read_jsonl


def normalize_jsonl(input_path: Path, output_path: Path) -> int:
    rows = [normalize_document(row) for row in read_jsonl(input_path)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8", newline="\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Dataset 1 raw document JSONL deterministically.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"normalized_documents={normalize_jsonl(args.input, args.output)}")


if __name__ == "__main__":
    main()
