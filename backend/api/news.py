from fastapi import APIRouter

from backend.schemas import AnalysisResponse, NewsAnalysisRequest
from backend.services import news_service

router = APIRouter(prefix="/api/news", tags=["news"])


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_news(request: NewsAnalysisRequest) -> AnalysisResponse:
    return news_service.analyze(request.text.strip())
