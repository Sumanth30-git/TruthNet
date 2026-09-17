from fastapi import APIRouter

from backend.schemas import AnalysisResponse
from backend.services import video_service

router = APIRouter(prefix="/api/video", tags=["video"])


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_video() -> AnalysisResponse:
    return video_service.analyze()
