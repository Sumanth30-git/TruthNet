from fastapi import APIRouter

from backend.config import API_VERSION, model_inference_enabled
from backend.models.registry import registry
from backend.schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        model_inference_enabled=model_inference_enabled(),
        models=registry.readiness(),
    )
