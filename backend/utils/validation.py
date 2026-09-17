from dataclasses import dataclass
from io import BytesIO
import warnings

from PIL import Image, UnidentifiedImageError

from backend.config import (
    ALLOWED_IMAGE_TYPES,
    IMAGE_TYPE_TO_FORMAT,
    MAX_IMAGE_HEIGHT,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_WIDTH,
)


@dataclass
class ValidatedImage:
    """An in-memory, decoded RGB image safe to pass to a future model adapter."""

    image: Image.Image
    content_type: str
    width: int
    height: int
    file_size_bytes: int


def validate_image_upload(content_type: str | None) -> None:
    if content_type not in ALLOWED_IMAGE_TYPES:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_TYPES))
        raise ValueError(f"Unsupported image type. Upload JPEG, PNG, or WebP ({allowed}).")


def decode_and_validate_image(content: bytes, content_type: str) -> ValidatedImage:
    """Decode, verify, size-limit, and normalize an uploaded image in memory."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as source:
                source.verify()

            with Image.open(BytesIO(content)) as source:
                detected_format = source.format
                expected_format = IMAGE_TYPE_TO_FORMAT[content_type]
                if detected_format != expected_format:
                    raise ValueError(
                        f"Image content does not match the declared {content_type} type."
                    )
                width, height = source.size
                if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
                    raise ValueError(
                        f"Image dimensions exceed the {MAX_IMAGE_WIDTH}×{MAX_IMAGE_HEIGHT} limit."
                    )
                if width * height > MAX_IMAGE_PIXELS:
                    raise ValueError(
                        f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel safety limit."
                    )
                normalized = source.convert("RGB")
                normalized.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("Image is too large to process safely.") from error
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ValueError("Uploaded file is corrupt or is not a valid image.") from error

    return ValidatedImage(
        image=normalized,
        content_type=content_type,
        width=width,
        height=height,
        file_size_bytes=len(content),
    )
