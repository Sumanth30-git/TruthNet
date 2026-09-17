"""Controlled local smoke test for the existing TruthNet news baseline.

This script is intentionally outside pytest. It may download the model on first
use and must be run only with MODEL_INFERENCE_ENABLED=true.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import model_inference_enabled
from backend.pipelines.news_pipeline import analyze_news
from backend.schemas import SignalStatus, Verdict

DEFAULT_TEXT = "A public agency announced a routine update to its community service schedule."


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one local TruthNet news-baseline smoke test.")
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Non-sensitive text for local testing. The default is a harmless public-style sentence.",
    )
    arguments = parser.parse_args()

    if not model_inference_enabled():
        print("MODEL_INFERENCE_ENABLED must be true to run this smoke test.", file=sys.stderr)
        return 2

    response = analyze_news(arguments.text)
    signal = response.signals[0]
    print(response.model_dump_json(indent=2))

    if response.verdict != Verdict.INCONCLUSIVE:
        print("Unexpected final verdict from smoke test.", file=sys.stderr)
        return 1
    if signal.status != SignalStatus.COMPLETE:
        print("News model smoke test did not complete successfully.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
