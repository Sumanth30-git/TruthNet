from backend.pipelines.news_pipeline import analyze_news
from backend.schemas import AnalysisResponse


def analyze(text: str) -> AnalysisResponse:
    return analyze_news(text)
