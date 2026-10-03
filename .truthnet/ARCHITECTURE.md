# TruthNet V2 Current Architecture

## System shape

TruthNet V2 is a FastAPI backend with a static vanilla HTML/CSS/JavaScript frontend. The backend owns validation, request handling, lazy model loading, pipeline orchestration, and structured results. The browser performs no model inference.

```text
Browser
  -> FastAPI API routers
  -> service wrappers
  -> content pipelines
  -> model adapters / search / evidence services
  -> fusion and decision policy
  -> AnalysisResponse JSON
```

`backend/schemas.py` defines the public request/result vocabulary: verdicts, uncertainty, signals, claims, search results, evidence, evidence quality, and evidence summaries.

## Layers

### FastAPI and API layer

`backend/main.py` configures logging, request IDs, structured exception handling, static-file serving, and routers.

- `GET /api/health` reports service status, inference setting, and registry readiness.
- `POST /api/news/analyze` accepts JSON `{ "text": "..." }`.
- `POST /api/image/analyze` accepts a multipart `image`.
- `POST /api/video/analyze` is a placeholder endpoint.

### Services and pipelines

API routers call small service wrappers, which delegate to `news_pipeline.py`, `image_pipeline.py`, or `video_pipeline.py`. Pipelines are the production orchestration layer.

### Model adapters

`backend/models/registry.py` registers lazy Hugging Face adapters for the News BERT baseline, AI-image baseline, and an unused deepfake baseline. `MODEL_INFERENCE_ENABLED` defaults to false; adapters load only on demand when it is enabled.

The NLI adapter is separate from the registry. It lazy-loads `cross-encoder/nli-deberta-v3-base` within the evidence-analysis boundary.

### Fusion and decisions

`backend/fusion/decision.py` contains conservative response defaults and the final News verdict policy. `backend/services/evidence_aggregation.py` computes claim-level evidence summaries before that policy runs.

### Frontend

`frontend/` is a static vanilla interface. It submits News/Image/Video requests and renders verdict, message, uncertainty, signals, and limitations. It does not currently render detailed claims, search candidates, evidence, NLI output, source quality, or aggregation summaries.

### Tests

`tests/` contains unit and integration-style coverage for API behavior, validation, models, News search/NLI/evidence logic, source quality, aggregation, verdict policy, image handling, and the frontend static surface. Tests use disabled inference and mocks by default; they do not validate downloaded models or real Tavily calls in the normal suite.

## Current News flow

```text
input text
  -> production claim creation: sentence segmentation
  -> search using the original full text
  -> claim x retrieved-source evidence items
  -> NLI: source snippet as premise, claim as hypothesis
  -> source-quality profile
  -> deterministic matching and freshness
  -> evidence independence and aggregation
  -> final News verdict policy
```

Details:

1. The optional `jy46604790/Fake-News-Bert-Detect` result is emitted as an auxiliary signal. Its result is intentionally ignored by the final News verdict policy.
2. Production claim creation currently uses `extract_candidate_claims()` in `backend/services/evidence_analysis.py`. It splits only at terminal punctuation and labels claims as `sentence_segmentation`.
3. `backend/services/claim_understanding.py` contains richer deterministic sentence/clause extraction and `ClaimContext` parsing, but it is not currently the production source for claim segmentation. The current pipeline attaches `ClaimContext` after successful search processing. This split is a known redesign point.
4. Search is provider-independent. The default provider is `none`; Tavily is the implemented provider when explicitly configured with `NEWS_SEARCH_PROVIDER=tavily` and `TAVILY_API_KEY`.
5. Evidence analysis builds one candidate item for each claim/source pair. NLI maps entailment to support, contradiction to contradiction, and neutral to neutral. It analyzes snippets, not fetched full source pages.
6. Source quality uses a conservative catalog of reviewed configured identities, including configured domains and their subdomains. Evidence matching separately assesses text/metadata matches and freshness.
7. Aggregation keeps support and contradiction separate, applies independence grouping, and produces one summary per claim.
8. A decisive result requires every extracted claim to have exactly one strong or very-strong summary in the same direction. The safety gates also require completed search, successful evidence aggregation, evidence analysis that is neither partial nor unavailable, and no claim with unavailable NLI evidence. Failure of any gate produces `inconclusive`; unavailable processing uses `not_assessed` uncertainty.

The default runtime remains inconclusive because model inference is off and search is unconfigured by default.

## Image and Video

The Image pipeline performs TruthNet validation and RGB normalization for JPEG/PNG/WebP uploads in memory, reports input/resolution signals, and always runs the classical Haar-cascade face-quality gate. FastAPI multipart handling may use transient operating-system temporary storage. The optional `capcheck/ai-human-generated-image-detection` adapter runs only when inference is enabled. Its final result is always `inconclusive`. The deepfake adapter is registered but is not invoked by the image pipeline.

The Video pipeline is an explicit `not_implemented` placeholder. It has no upload, decoding, frame, or model-analysis implementation.

## Architectural boundaries

- `backend/schemas.py`: public API and evidence contract.
- `backend/config.py`: safety limits and environment configuration contract.
- `backend/api/`: public HTTP behavior.
- `backend/pipelines/news_pipeline.py`: production News orchestration order and failure behavior.
- `backend/services/claim_understanding.py`: deterministic claim/`ClaimContext` behavior.
- `backend/services/news_search.py`: search-provider, credential, timeout, and normalization boundary.
- `backend/services/evidence_analysis.py`: replaceable stance-analysis boundary.
- `backend/services/source_quality.py`: reviewed source-identity and authority semantics.
- `backend/services/evidence_matching.py`: matching and freshness semantics.
- `backend/services/evidence_aggregation.py`: weights, thresholds, independence, and claim summaries.
- `backend/fusion/decision.py`: final News verdict safety policy.
- `backend/models/`: lazy-loading, metadata, label mapping, and normalized model signals.
- `backend/utils/validation.py`: image-upload safety boundary.
