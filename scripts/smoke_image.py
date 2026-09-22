"""Controlled local smoke test for the existing TruthNet AI-image baseline.

This script is intentionally outside pytest. It may download the model on first
use and must be run only with MODEL_INFERENCE_ENABLED=true and a local image.
"""

import argparse
import mimetypes
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.config import ALLOWED_IMAGE_TYPES, model_inference_enabled
from backend.main import app
from backend.schemas import SignalStatus, Verdict

AI_IMAGE_MODEL_ID = "capcheck/ai-human-generated-image-detection"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one local TruthNet AI-image baseline smoke test."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to a local JPEG, PNG, or WebP image to analyze.",
    )
    arguments = parser.parse_args()

    if not model_inference_enabled():
        print("MODEL_INFERENCE_ENABLED must be true to run this smoke test.", file=sys.stderr)
        return 2

    image_path = arguments.image.expanduser().resolve()
    if not image_path.is_file():
        print(f"Image file does not exist: {image_path}", file=sys.stderr)
        return 2

    content_type, _ = mimetypes.guess_type(image_path.name)
    if content_type not in ALLOWED_IMAGE_TYPES:
        print("Image must have a .jpg, .jpeg, .png, or .webp extension.", file=sys.stderr)
        return 2

    with TestClient(app) as client:
        response = client.post(
            "/api/image/analyze",
            files={"image": (image_path.name, image_path.read_bytes(), content_type)},
        )

    try:
        payload = response.json()
    except ValueError:
        print(response.text, file=sys.stderr)
        return 1

    print(response.text)

    if response.status_code != 200:
        print("Image smoke test request did not succeed.", file=sys.stderr)
        return 1
    if payload.get("verdict") != Verdict.INCONCLUSIVE.value:
        print("Unexpected final verdict from smoke test.", file=sys.stderr)
        return 1

    signal = next(
        (item for item in payload.get("signals", []) if item.get("model_id") == AI_IMAGE_MODEL_ID),
        None,
    )
    if signal is None or signal.get("status") != SignalStatus.COMPLETE.value:
        print("AI-image model smoke test did not complete successfully.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
