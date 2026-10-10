from __future__ import annotations

import hashlib

import pytest

from scripts.dataset1_collect_candidates import collect_candidate


def _seed() -> dict:
    return {
        "candidate_id": "candidate-1", "canonical_uri": "https://example.org/article",
        "collection": {"collector_id": "collector-1"}, "notes": "topic=test",
    }


def test_collection_fetches_only_explicit_url_and_records_deterministic_visible_text_checksum():
    calls: list[str] = []

    def fetch(uri: str) -> tuple[bytes, str]:
        calls.append(uri)
        return b"<html><body>Useful <b>candidate</b><script>ignored()</script> text.</body></html>", "text/html"

    result = collect_candidate(_seed(), collected_at="2026-10-10T00:00:00Z", fetch=fetch)

    assert calls == ["https://example.org/article"]
    assert result["collection"] == {
        "collected_at": "2026-10-10T00:00:00Z", "collector_id": "collector-1", "access_status": "accessible",
        "content_sha256": hashlib.sha256(b"Useful candidate text.").hexdigest(), "content_length": 22,
        "content_format": "text",
    }
    assert result["notes"] == "topic=test; collection_checksum_scope=visible_text_nfc_lf_v1"


def test_collection_rejects_non_text_or_empty_public_content():
    with pytest.raises(ValueError, match="unsupported"):
        collect_candidate(_seed(), collected_at="2026-10-10T00:00:00Z", fetch=lambda _: (b"binary", "application/pdf"))
    with pytest.raises(ValueError, match="meaningful"):
        collect_candidate(_seed(), collected_at="2026-10-10T00:00:00Z", fetch=lambda _: (b"<script>x</script>", "text/html"))
