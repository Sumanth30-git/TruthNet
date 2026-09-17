from backend.schemas import AnalysisResponse, ModelInfo, SignalResult, Uncertainty, Verdict


def inconclusive_result(
    message: str,
    signals: list[SignalResult] | None = None,
    model_versions: list[ModelInfo] | None = None,
    limitations: list[str] | None = None,
) -> AnalysisResponse:
    """Safe default until a calibrated, validated fusion policy exists."""
    return AnalysisResponse(
        verdict=Verdict.INCONCLUSIVE,
        confidence=None,
        uncertainty=Uncertainty.HIGH,
        signals=signals or [],
        model_versions=model_versions or [],
        limitations=[
            "No calibrated evidence fusion has run.",
            "A final result is intentionally not forced into authentic or fake.",
            *(limitations or []),
        ],
        message=message,
    )
