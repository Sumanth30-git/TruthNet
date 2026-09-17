from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    LIKELY_AUTHENTIC = "likely_authentic"
    LIKELY_AI_GENERATED = "likely_ai_generated"
    LIKELY_MANIPULATED = "likely_manipulated"
    INCONCLUSIVE = "inconclusive"
    NOT_IMPLEMENTED = "not_implemented"


class Uncertainty(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    NOT_ASSESSED = "not_assessed"


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class SignalStatus(str, Enum):
    NOT_RUN = "not_run"
    NOT_APPLICABLE = "not_applicable"
    COMPLETE = "complete"
    FAILED = "failed"


class ModelLoadStatus(str, Enum):
    NOT_LOADED = "not_loaded"
    LOADED = "loaded"
    FAILED = "failed"


class ModelInfo(BaseModel):
    model_id: str
    model_version: str
    content_type: ContentType


class ModelReadiness(ModelInfo):
    status: ModelLoadStatus
    message: str | None = None


class SignalResult(BaseModel):
    model_id: str
    model_version: str
    prediction: str
    raw_score: float | None = Field(default=None, ge=0, le=1)
    calibrated_score: float | None = Field(default=None, ge=0, le=1)
    content_type: ContentType
    status: SignalStatus
    inference_time_ms: float | None = Field(default=None, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    kind: str
    summary: str
    source: str | None = None


class AnalysisResponse(BaseModel):
    verdict: Verdict
    confidence: float | None = Field(default=None, ge=0, le=1)
    uncertainty: Uncertainty
    signals: list[SignalResult] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    model_versions: list[ModelInfo] = Field(default_factory=list)
    message: str | None = None


class NewsAnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


class HealthResponse(BaseModel):
    status: str
    version: str
    model_inference_enabled: bool
    models: list[ModelReadiness] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
