# TruthNet V2

TruthNet V2 is an evidence-oriented, multimodal fake-content analysis platform under phased development. It uses a vanilla HTML/CSS/JavaScript frontend and a FastAPI backend.

Phase 0 foundation work is complete. Phase 1A integrates the existing news baseline and Phase 1B integrates the existing AI-image baseline. Both are guarded by `MODEL_INFERENCE_ENABLED`, which defaults to `false`. With the default setting, no Hugging Face weights are loaded or run. When inference is explicitly enabled and the optional model runtime is available, each adapter lazy-loads only when it first receives a request and then reuses its in-memory pipeline.

The automated tests keep inference disabled by default and use mocked model adapters for inference paths, so they do not download Hugging Face model weights.

## Architecture at a glance

```text
Browser (HTML, CSS, JavaScript)
  | REST / JSON
FastAPI backend
  |-- News: claims -> search/retrieval -> NLI/matching -> aggregation/verdict
  |-- Image: independent model and quality signals
  `-- Video: future
  |
Structured result: verdict, uncertainty, signals, evidence, limitations
```

The browser never runs ML inference. The backend owns validation, inference flag checks, lazy model loading, model calls, and structured response formatting.

## What works now

- Responsive custom web interface for News, Image, and future Video analysis.
- `GET /api/health` health endpoint.
- `MODEL_INFERENCE_ENABLED` defaults to `false` for safe local development and test execution.
- A flag-gated news baseline adapter for `jy46604790/Fake-News-Bert-Detect`.
- A provider-independent News `SearchProvider` contract and a Tavily current-web search provider that returns typed source candidates and safely handles failures. Search remains disabled by default (`NEWS_SEARCH_PROVIDER=none`).
- A replaceable News evidence-analysis boundary using `cross-encoder/nli-deberta-v3-base` for source-text/claim stance when model inference is explicitly enabled. It emits structured NLI signals and never treats source identity or NLI output as proof of truth.
- A flag-gated AI-image baseline adapter for `capcheck/ai-human-generated-image-detection`.
- Server-side image decoding and validation for JPEG, PNG, and WebP files up to 10 MB; corrupt files, MIME mismatches, oversized dimensions, and excessive pixel counts are rejected. Images are normalized to RGB in memory only.
- A classical face-quality gate that reports usable faces, absent faces, low-quality faces, or detector failure as an independent image signal. It uses a minimum detected face width and height of 80 pixels; this is a processing threshold, not identity verification or a deepfake result.
- Stable response schema with verdict, confidence, uncertainty, signals, evidence, limitations, and model versions.
- Lazy model adapters and a registry that report model readiness without loading weights at application startup.
- Safe structured API errors with a request ID; diagnostic details remain in server logs.
- A connected News evidence pipeline with claim-driven search, bounded source retrieval, NLI stance analysis, source/claim matching, numerical checks, claim-level aggregation, and a fail-closed verdict policy.
- News verdict confidence remains null because calibrated probabilities of real-world truth are not available; weak, mixed, incomplete, or unavailable evidence remains inconclusive.

## API contract

| Endpoint | Method | Current behavior |
| --- | --- | --- |
| `/api/health` | GET | Reports application health, whether inference is enabled, and lazy-model readiness. |
| `/api/news/analyze` | POST | Accepts `{ "text": "..." }`; returns deterministic claims and text-derived context, claim-driven search candidates, optional bounded retrieval, per-source NLI and matching details, claim-level evidence summaries, uncertainty, and a fail-closed News verdict. |
| `/api/image/analyze` | POST | Accepts multipart field `image`; validates it, reports an independent face-quality signal, and reports the AI-image baseline signal when enabled (otherwise `not_run`). Final verdict remains `inconclusive`. |
| `/api/video/analyze` | POST | Returns `not_implemented`; no video ML runs. |

The frontend renders the response contract and contains no ML inference logic. It displays each signal's model ID, prediction, status, raw score when available, inference time when available, and API-provided limitations. `backend/schemas.py` is the single source of truth for result responses.

## Current baseline behavior

### News

The news baseline preserves the V1 model mapping: `LABEL_0` is reported as `fake_news` and `LABEL_1` as `real_news`. This is a text-classifier signal only; it is **not** live fact verification. The search layer defines an injectable `SearchProvider` contract and includes a Tavily adapter for current-web source candidates. The default is `NEWS_SEARCH_PROVIDER=none`, which returns `search.status = "not_configured"`; set it to `tavily` to enable requests.

The production News pipeline uses `ClaimUnderstandingService` to create deterministic sentence and clearly independent clause candidates with text-derived context. Search queries are derived from those claims and bounded by a three-query cap and `NEWS_SEARCH_MAX_RESULTS` (default `5`). When `NEWS_SOURCE_RETRIEVAL_ENABLED=true`, the pipeline attempts bounded retrieval of public source pages with SSRF, redirect, response-size, timeout, and text-size controls; otherwise, or when retrieval fails, it uses the search snippet. Evidence records retain source and retrieval provenance. When `MODEL_INFERENCE_ENABLED=true`, the lazy NLI adapter evaluates prepared source text as the **premise** and each claim as the **hypothesis** using `cross-encoder/nli-deberta-v3-base`. Its model-label mapping is `0 (contradiction) = contradicts`, `1 (entailment) = supports`, and `2 (neutral) = neutral`. Evidence retains the predicted relation, all three softmax scores, model ID/version, status, inference time, and source truncation state.

NLI describes the textual relationship between source text and a claim. It is not fact verification, a source credibility judgment, or proof either statement is true. Claim/source matching and source quality are recorded separately; source authority does not determine NLI stance. Before tokenization, source text is capped by `NEWS_NLI_MAX_SOURCE_CHARS` (default 6000 characters), then the pair is tokenized with truncation to `NEWS_NLI_MAX_TOKENS` (default 512). Claim-level aggregation keeps support and contradiction separate and groups duplicate or copied sources. A numerical mismatch cannot add support. The fail-closed policy returns `likely_authentic` only when every claim has strong support, `likely_fake_news` only when every claim has strong contradiction, and `inconclusive` otherwise or when required processing is unavailable. Confidence remains null because calibrated probabilities of real-world truth are not available.

Search configuration is explicit: `NEWS_SEARCH_PROVIDER` defaults to `none`; `NEWS_SEARCH_TIMEOUT_SECONDS` defaults to `5` seconds (accepted range greater than 0 and at most 60); `NEWS_SEARCH_MAX_RESULTS` defaults to `5` (range 1 to 20); and Tavily reads its credential only from `TAVILY_API_KEY`. If Tavily is selected without that key, search returns an unavailable status. Source-page retrieval is separately gated by `NEWS_SOURCE_RETRIEVAL_ENABLED` and defaults to disabled. Retrieval is bounded to configured source, time, response-byte, redirect, and extracted-text limits. No key is required for tests.

NLI configuration is also explicit: `NEWS_NLI_MODEL_ID` defaults to `cross-encoder/nli-deberta-v3-base`, `NEWS_NLI_MAX_TOKENS` defaults to `512`, and `NEWS_NLI_MAX_SOURCE_CHARS` defaults to `6000`. The NLI tokenizer and model are not imported or loaded at application startup and are not loaded while `MODEL_INFERENCE_ENABLED=false`. The adapter uses CUDA when PyTorch reports availability and otherwise runs on CPU.

### AI images

The AI-image baseline preserves the V1 model mapping: `AI-generated` is reported as `ai_generated` and `human` as `human_created`. This is an individual AI-image classification signal only. Its raw score is not a calibrated probability and is not proof that an image is authentic or synthetic.

The image pipeline currently returns `inconclusive`; News can return an evidence-based likely verdict only when all extracted claims pass the existing strong-evidence gates. Neither verdict is proof of real-world truth or intent.

## Setup and run

Use Python 3.11–3.13 where possible. Python 3.14 support depends on the availability of compatible FastAPI ecosystem packages in your environment.

```powershell
cd C:\Users\hppav\Desktop\TruthNet-V2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

To allow the implemented baseline adapters to attempt inference, set the flag only for the server process:

```powershell
$env:MODEL_INFERENCE_ENABLED = "true"
python -m uvicorn backend.main:app --reload
```

If the optional Transformer runtime is unavailable, enabled model calls return a safe `failed` signal rather than a fabricated result.

## Controlled local model smoke tests

Phase 1D adds the minimum pinned news-model runtime dependencies: `transformers==4.57.6` and `torch==2.14.0`. The selected versions have installable CPython 3.14 Windows wheels and retain the stable Transformers 4.x API used by the existing adapter.

The standalone smoke tests are deliberately outside the normal pytest suite. They may download model weights on first use and verify that the resulting response remains `inconclusive` while the individual model signal completes.

The news smoke test uses the existing `jy46604790/Fake-News-Bert-Detect` adapter. Run it only with non-sensitive text and explicit inference enablement:

```powershell
$env:MODEL_INFERENCE_ENABLED = "true"
python scripts\smoke_news.py
```

To manually run the NLI stance smoke test (it may download the NLI model on first use):

```powershell
$env:MODEL_INFERENCE_ENABLED = "true"
python scripts\smoke_nli.py
```

It exercises entailment, contradiction, and neutral examples and prints the device, fixed label mapping, prediction, probabilities, and inference time.

The default smoke-test input is a harmless public-style sentence. You may supply another non-sensitive test sentence with `--text`.

Phase 1E adds a controlled real local AI-image smoke test for the existing `capcheck/ai-human-generated-image-detection` adapter. It requires an explicit path to a local JPEG, PNG, or WebP image and submits it to `/api/image/analyze`, so the normal upload validation, in-memory RGB normalization, and image pipeline all run. Do not use sensitive images.

```powershell
$env:MODEL_INFERENCE_ENABLED = "true"
python scripts\smoke_image.py --image C:\path\to\local-image.jpg
```

The image smoke test prints the structured API response, including the original model label, mapped prediction, raw score, signal status, inference time, and final `inconclusive` verdict.

## Tests

```powershell
python -m pytest
```

Tests keep `MODEL_INFERENCE_ENABLED=false` and use mocks for model-output paths, so they do not download Hugging Face model weights.

## Not implemented yet

- Deepfake detection is not integrated into the image pipeline.
- Broader metadata forensics and source credibility judgments are not implemented; source authority metadata is contextual and does not establish truth.
- The News evidence policy has not been calibrated or validated as a real-world truth probability.
- Video ML analysis is not implemented; the video endpoint is a clear placeholder only.
- No accuracy-improvement or benchmark-performance claim has been made.

## Development rule

NLI model scores and evidence weights are uncalibrated components of a claim-level evidence assessment, not probabilities that a claim is true. News verdicts require strong support or contradiction across all extracted claims and remain inconclusive when evidence is weak, mixed, incomplete, or unavailable. `likely_fake_news` means available evidence materially contradicts the claims; it does not imply deliberate manipulation. `likely_authentic` is not proof of authenticity.

Uploaded files are not saved by TruthNet application code. Web-server upload handling may use short-lived operating-system temporary storage for multipart requests; production deployment must document retention and enforce request-size, authentication, and rate-limit policies.
