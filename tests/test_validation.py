import pytest

from backend.utils.validation import validate_image_upload


def test_accepts_supported_image_type() -> None:
    validate_image_upload("image/jpeg")


def test_rejects_unsupported_image_type() -> None:
    with pytest.raises(ValueError, match="Unsupported image type"):
        validate_image_upload("image/gif")
