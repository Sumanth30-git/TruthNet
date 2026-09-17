import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_WIDTH = 4_096
MAX_IMAGE_HEIGHT = 4_096
MAX_IMAGE_PIXELS = 12_000_000
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
IMAGE_TYPE_TO_FORMAT = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
APP_NAME = "TruthNet V2"
API_VERSION = "0.1.0"
_TRUTHY_ENV = {"1", "true", "yes", "on"}


def model_inference_enabled() -> bool:
    """Read MODEL_INFERENCE_ENABLED at call time. Default is off (safe for tests)."""
    return os.getenv("MODEL_INFERENCE_ENABLED", "false").strip().lower() in _TRUTHY_ENV
