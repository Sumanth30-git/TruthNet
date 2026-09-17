from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.config import ALLOWED_IMAGE_TYPES, MAX_UPLOAD_BYTES
from backend.schemas import AnalysisResponse
from backend.services import image_service
from backend.utils.validation import decode_and_validate_image, validate_image_upload

router = APIRouter(prefix="/api/image", tags=["image"])


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(image: UploadFile = File(...)) -> AnalysisResponse:
    try:
        validate_image_upload(image.content_type)
        content = await image.read(MAX_UPLOAD_BYTES + 1)
        if not content:
            raise ValueError("The uploaded image is empty.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("The uploaded image exceeds the 10 MB limit.")
        validated_image = decode_and_validate_image(content, image.content_type or "")
        try:
            return image_service.analyze(validated_image)
        finally:
            validated_image.image.close()
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    finally:
        await image.close()
