from datetime import datetime
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


class EvidenceStance(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    INSUFFICIENT = "insufficient"


class EvidenceRelevance(str, Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    NOT_ASSESSED = "not_assessed"


class EvidenceAnalysisStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    UNAVAILABLE = "unavailable"
    NO_SOURCES = "no_sources"
    NOT_RUN = "not_run"


class Claim(BaseModel):
    claim_id: str
    text: str
    extraction_method: str = "sentence_segmentation"


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


class NLIProbabilities(BaseModel):
    contradiction: float = Field(ge=0, le=1)
    entailment: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)


class NLIEvidenceSignal(BaseModel):
    """The NLI relation between retrieved source text (premise) and claim (hypothesis)."""

    model_id: str
    model_version: str
    prediction: str
    status: SignalStatus
    probabilities: NLIProbabilities | None = None
    selected_confidence: float | None = Field(default=None, ge=0, le=1)
    inference_time_ms: float | None = Field(default=None, ge=0)
    source_text_truncated: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    kind: str
    summary: str
    source: str | None = None
    claim_id: str | None = None
    claim: str | None = None
    source_title: str | None = None
    source_url: str | None = None
    publisher: str | None = None
    published_at: str | None = None
    retrieved_at: datetime | None = None
    stance: EvidenceStance = EvidenceStance.INSUFFICIENT
    relevance: EvidenceRelevance = EvidenceRelevance.NOT_ASSESSED
    analysis_status: EvidenceAnalysisStatus = EvidenceAnalysisStatus.NOT_RUN
    reason: str | None = None
    nli_signal: NLIEvidenceSignal | None = None


class AnalysisResponse(BaseModel):
    verdict: Verdict
    confidence: float | None = Field(default=None, ge=0, le=1)
    uncertainty: Uncertainty
    signals: list[SignalResult] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    model_versions: list[ModelInfo] = Field(default_factory=list)
    search: "SearchResponse | None" = None
    claims: list[Claim] = Field(default_factory=list)
    evidence_analysis_status: EvidenceAnalysisStatus = EvidenceAnalysisStatus.NOT_RUN
    message: str | None = None


class NewsAnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


class SearchResult(BaseModel):
    """A retrieved source candidate; it is not a finding about the claim."""

    title: str
    url: str
    snippet: str | None = None
    publisher: str | None = None
    published_at: str | None = None
    retrieved_at: datetime


class SearchResponse(BaseModel):
    status: str
    provider: str
    results: list[SearchResult] = Field(default_factory=list)
    message: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_inference_enabled: bool
    models: list[ModelReadiness] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
