from backend.pipelines.video_pipeline import analyze_video
from backend.schemas import AnalysisResponse


def analyze() -> AnalysisResponse:
    return analyze_video()
