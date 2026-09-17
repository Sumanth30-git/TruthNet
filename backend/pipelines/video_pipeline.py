from backend.schemas import AnalysisResponse, Uncertainty, Verdict


def analyze_video() -> AnalysisResponse:
    return AnalysisResponse(
        verdict=Verdict.NOT_IMPLEMENTED,
        confidence=None,
        uncertainty=Uncertainty.NOT_ASSESSED,
        limitations=["Video detection is a future phase and is not implemented."],
        message="Video analysis is coming soon.",
    )
