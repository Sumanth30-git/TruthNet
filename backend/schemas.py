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


class ClaimType(str, Enum):
    GENERAL_FACT = "general_fact"
    GOVERNMENT_ANNOUNCEMENT = "government_announcement"
    COMPANY_ANNOUNCEMENT = "company_announcement"
    PRODUCT_ANNOUNCEMENT = "product_announcement"
    FINANCIAL_PRICE = "financial_price"
    SCIENTIFIC = "scientific"
    LEGAL_COURT = "legal_court"
    SPORTS_RESULT = "sports_result"
    PERSON_STATEMENT = "person_statement"
    EVENT = "event"
    ORGANIZATION_ANNOUNCEMENT = "organization_announcement"
    OTHER = "other"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    OFFICIAL_PRIMARY = "official_primary"
    ESTABLISHED_NEWS = "established_news"
    SPECIALIST_FINANCIAL = "specialist_financial"
    RESEARCH_INSTITUTION = "research_institution"
    REGULATORY_OR_LEGAL = "regulatory_or_legal"
    ORGANIZATION = "organization"
    KNOWN_PUBLISHER = "known_publisher"
    SOCIAL_OFFICIAL = "social_official"
    SOCIAL_UNVERIFIED = "social_unverified"
    UNKNOWN = "unknown"


class AuthorityLevel(str, Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"
    UNKNOWN = "unknown"


class SourceRelationship(str, Enum):
    DIRECT_PRIMARY = "direct_primary"
    OFFICIAL_STATEMENT = "official_statement"
    DIRECT_DATA_PROVIDER = "direct_data_provider"
    INDEPENDENT_REPORTING = "independent_reporting"
    SECONDARY_REPORTING = "secondary_reporting"
    COMMENTARY = "commentary"
    UNVERIFIED_REPOST = "unverified_repost"
    UNKNOWN = "unknown"


class Officiality(str, Enum):
    VERIFIED_OFFICIAL = "verified_official"
    LIKELY_OFFICIAL = "likely_official"
    NOT_ESTABLISHED = "not_established"
    UNVERIFIED = "unverified"
    NOT_APPLICABLE = "not_applicable"


class ClaimSourceRelevance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOT_ASSESSED = "not_assessed"


class MatchStatus(str, Enum):
    MATCH = "match"
    PARTIAL_MATCH = "partial_match"
    MISMATCH = "mismatch"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class Freshness(str, Enum):
    CURRENT = "current"
    RECENT = "recent"
    STALE = "stale"
    UNKNOWN = "unknown"


class NumericalStatus(str, Enum):
    EXACT_MATCH = "exact_match"
    WITHIN_TOLERANCE = "within_tolerance"
    MISMATCH = "mismatch"
    UNIT_MISMATCH = "unit_mismatch"
    NOT_ASSESSED = "not_assessed"


class EvidenceIndependence(str, Enum):
    INDEPENDENT = "independent"
    SAME_GROUP = "same_group"
    POSSIBLE_COPY = "possible_copy"
    UNKNOWN = "unknown"


class EvidenceStrength(str, Enum):
    INSUFFICIENT = "insufficient"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class EvidenceAggregateState(str, Enum):
    VERY_STRONG_SUPPORT = "very_strong_support"
    STRONG_SUPPORT = "strong_support"
    MODERATE_SUPPORT = "moderate_support"
    WEAK_SUPPORT = "weak_support"
    VERY_STRONG_CONTRADICTION = "very_strong_contradiction"
    STRONG_CONTRADICTION = "strong_contradiction"
    MODERATE_CONTRADICTION = "moderate_contradiction"
    WEAK_CONTRADICTION = "weak_contradiction"
    BALANCED_OR_CONFLICTING = "balanced_or_conflicting"
    INSUFFICIENT = "insufficient"


class ClaimNumber(BaseModel):
    raw_text: str
    value: float
    unit: str | None = None
    attribute: str | None = None


class ClaimContext(BaseModel):
    claim_id: str
    factual_statement: str
    subject: str | None = None
    claim_type: ClaimType = ClaimType.UNKNOWN
    event: str | None = None
    location: str | None = None
    time_reference: str | None = None
    numerical_values: list[ClaimNumber] = Field(default_factory=list)
    relevant_attributes: dict[str, str] = Field(default_factory=dict)
    extraction_notes: list[str] = Field(default_factory=list)


class SourceProfile(BaseModel):
    publisher: str | None = None
    domain: str | None = None
    platform: str | None = None
    source_type: SourceType = SourceType.UNKNOWN
    authority_level: AuthorityLevel = AuthorityLevel.UNKNOWN
    primary_source: bool | None = None
    officiality: Officiality = Officiality.NOT_APPLICABLE
    source_relationship: SourceRelationship = SourceRelationship.UNKNOWN
    authority_reason: str


class EvidenceMatch(BaseModel):
    relevance: ClaimSourceRelevance = ClaimSourceRelevance.NOT_ASSESSED
    entity_match: MatchStatus = MatchStatus.UNKNOWN
    event_match: MatchStatus = MatchStatus.UNKNOWN
    location_match: MatchStatus = MatchStatus.NOT_APPLICABLE
    time_match: MatchStatus = MatchStatus.UNKNOWN
    reason: str


class NumericalCheck(BaseModel):
    status: NumericalStatus = NumericalStatus.NOT_ASSESSED
    claim_raw: str | None = None
    source_raw: str | None = None
    claim_value: float | None = None
    source_value: float | None = None
    unit: str | None = None
    absolute_difference: float | None = Field(default=None, ge=0)
    relative_difference: float | None = Field(default=None, ge=0)
    tolerance: float | None = Field(default=None, ge=0)
    reason: str


class EvidenceQuality(BaseModel):
    source_profile: SourceProfile
    match: EvidenceMatch
    numerical_check: NumericalCheck | None = None
    freshness: Freshness = Freshness.UNKNOWN
    independence: EvidenceIndependence = EvidenceIndependence.UNKNOWN
    independence_group: str | None = None
    evidence_weight: float | None = Field(default=None, ge=0, le=1)
    component_scores: dict[str, float | None] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class EvidenceSummary(BaseModel):
    claim_id: str
    support_strength: EvidenceStrength = EvidenceStrength.INSUFFICIENT
    contradiction_strength: EvidenceStrength = EvidenceStrength.INSUFFICIENT
    overall_strength: EvidenceAggregateState = EvidenceAggregateState.INSUFFICIENT
    support_weight: float = Field(default=0, ge=0, le=1)
    contradiction_weight: float = Field(default=0, ge=0, le=1)
    independent_support_groups: int = Field(default=0, ge=0)
    independent_contradiction_groups: int = Field(default=0, ge=0)
    reason: str


class Claim(BaseModel):
    claim_id: str
    text: str
    extraction_method: str = "sentence_segmentation"
    context: ClaimContext | None = None
    evidence_summary: EvidenceSummary | None = None


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
    source_snippet: str | None = None
    publisher: str | None = None
    published_at: str | None = None
    retrieved_at: datetime | None = None
    stance: EvidenceStance = EvidenceStance.INSUFFICIENT
    relevance: EvidenceRelevance = EvidenceRelevance.NOT_ASSESSED
    analysis_status: EvidenceAnalysisStatus = EvidenceAnalysisStatus.NOT_RUN
    reason: str | None = None
    nli_signal: NLIEvidenceSignal | None = None
    evidence_quality: EvidenceQuality | None = None


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
    evidence_summaries: list[EvidenceSummary] = Field(default_factory=list)
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
