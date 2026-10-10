"""Fetch explicitly selected public Dataset 1 candidate pages.

This is deliberately not a crawler: it follows no links, accepts only explicit
JSONL seed rows, and never assigns a final split or creates a normalized
document.  It produces candidate-ledger rows whose eligibility decision remains
the human-provided decision in the seed.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen


USER_AGENT = "TruthNet-Dataset1-Stage3B/1.0 (controlled candidate collection)"
TEXT_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


class _TextExtractor(HTMLParser):
    """Conservative visible-text extractor for collection checksums only."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "template"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "template"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


def _normalise_visible_text(raw: bytes, content_type: str) -> str:
    decoded = raw.decode("utf-8", errors="replace")
    if content_type.startswith("text/plain"):
        return unicodedata.normalize("NFC", decoded.replace("\r\n", "\n").replace("\r", "\n"))
    parser = _TextExtractor()
    parser.feed(decoded)
    return unicodedata.normalize("NFC", parser.text())


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    stream = sys.stdin if path is None else path.open(encoding="utf-8")
    try:
        rows = [json.loads(line) for line in stream if line.strip()]
    finally:
        if path is not None:
            stream.close()
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("seed JSONL rows must be objects")
    return rows


def collect_candidate(
    seed: Mapping[str, Any],
    *,
    collected_at: str,
    fetch: Callable[[str], tuple[bytes, str]] | None = None,
) -> dict[str, Any]:
    """Return one ledger row after fetching its already-approved public URL."""
    candidate = dict(seed)
    uri = candidate.get("canonical_uri")



    if fetch is None:
        content_type = response.headers.get_content_type().lower()
        raw = response.read()
    else:
        raw, content_type = fetch(uri)

    content_type = content_type.lower().split(";")[0].strip()
    if content_type not in TEXT_CONTENT_TYPES:
        raise ValueError(f"unsupported public content type: {content_type}")

    text = _normalise_visible_text(raw, content_type)




    if not text.strip():
        raise ValueError("candidate page contains no meaningful textual content")
    candidate["collection"] = {
        "collected_at": collected_at,
        "collector_id": candidate.get("collection", {}).get("collector_id", "stage3b-controlled-collection"),
        "access_status": "accessible",
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "content_length": len(text),
        "content_format": "text",
    }
    notes = candidate.get("notes", "")
    scope = "collection_checksum_scope=visible_text_nfc_lf_v1"
    candidate["notes"] = f"{notes}; {scope}".strip("; ")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch explicit public Dataset 1 candidate pages into ledger rows.")
    parser.add_argument("input", help="seed JSONL path, or - for stdin")
    parser.add_argument("output", type=Path, help="output candidate JSONL")
    parser.add_argument("--collected-at", required=True, help="timezone-bearing ISO-8601 collection timestamp")
    args = parser.parse_args()
    seeds = _read_jsonl(None if args.input == "-" else Path(args.input))
    rows = [collect_candidate(seed, collected_at=args.collected_at) for seed in seeds]
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
