import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_WIDTH = 4_096
MAX_IMAGE_HEIGHT = 4_096
MAX_IMAGE_PIXELS = 12_000_000
# Haar-cascade detector parameters and the minimum size for a usable face gate.
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5
FACE_DETECTION_MIN_SIZE_PX = 32
FACE_QUALITY_MIN_DIMENSION_PX = 80
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
IMAGE_TYPE_TO_FORMAT = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
APP_NAME = "TruthNet V2"
API_VERSION = "0.1.0"
_TRUTHY_ENV = {"1", "true", "yes", "on"}
DEFAULT_NEWS_NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"
DEFAULT_NEWS_NLI_MAX_TOKENS = 512
DEFAULT_NEWS_NLI_MAX_SOURCE_CHARS = 6_000


def model_inference_enabled() -> bool:
    """Read MODEL_INFERENCE_ENABLED at call time. Default is off (safe for tests)."""
    return os.getenv("MODEL_INFERENCE_ENABLED", "false").strip().lower() in _TRUTHY_ENV


def news_nli_model_id() -> str:
    """Read the NLI model selection without loading a model."""
    return os.getenv("NEWS_NLI_MODEL_ID", DEFAULT_NEWS_NLI_MODEL_ID).strip() or DEFAULT_NEWS_NLI_MODEL_ID


def news_nli_max_tokens() -> int:
    """Keep NLI pair inputs within the supported 512-token model context."""
    try:
        value = int(os.getenv("NEWS_NLI_MAX_TOKENS", str(DEFAULT_NEWS_NLI_MAX_TOKENS)))
        return value if 64 <= value <= DEFAULT_NEWS_NLI_MAX_TOKENS else DEFAULT_NEWS_NLI_MAX_TOKENS
    except ValueError:
        return DEFAULT_NEWS_NLI_MAX_TOKENS


def news_nli_max_source_chars() -> int:
    """Bound web-text preprocessing before tokenizer truncation."""
    try:
        value = int(os.getenv("NEWS_NLI_MAX_SOURCE_CHARS", str(DEFAULT_NEWS_NLI_MAX_SOURCE_CHARS)))
        return value if 256 <= value <= 20_000 else DEFAULT_NEWS_NLI_MAX_SOURCE_CHARS
    except ValueError:
        return DEFAULT_NEWS_NLI_MAX_SOURCE_CHARS
