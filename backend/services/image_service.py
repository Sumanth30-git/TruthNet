from backend.pipelines.image_pipeline import analyze_image
from backend.schemas import AnalysisResponse
from backend.utils.validation import ValidatedImage


def analyze(image: ValidatedImage) -> AnalysisResponse:
    return analyze_image(image)
