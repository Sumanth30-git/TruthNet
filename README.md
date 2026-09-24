# TruthNet V2

TruthNet V2 is an evidence-oriented, multimodal fake-content analysis platform under phased development. It uses a vanilla HTML/CSS/JavaScript frontend and a FastAPI backend.

Phase 0 foundation work is complete. Phase 1A integrates the existing news baseline and Phase 1B integrates the existing AI-image baseline. Both are guarded by `MODEL_INFERENCE_ENABLED`, which defaults to `false`. With the default setting, no Hugging Face weights are loaded or run. When inference is explicitly enabled and the optional model runtime is available, each adapter lazy-loads only when it first receives a request and then reuses its in-memory pipeline.

The automated tests keep inference disabled by default and use mocked model adapters for inference paths, so they do not download Hugging Face model weights.

## Architecture at a glance

```text
Browser (HTML, CSS, JavaScript)
        ↓ REST / JSON
FastAPI backend
        ↓
Orchestration layer
   ├── News pipeline
   ├── Image pipeline
   └── Video pipeline (future)
        ↓
Independent baseline model signals
        ↓
Evidence fusion (future, only after evaluation)
        ↓
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
- An explicit `inconclusive` final verdict for news and image responses: signals are shown independently and are not fused.

## API contract

| Endpoint | Method | Current behavior |
| --- | --- | --- |
| `/api/health` | GET | Reports application health, whether inference is enabled, and lazy-model readiness. |
| `/api/news/analyze` | POST | Accepts `{ "text": "..." }`; reports the news baseline signal when enabled, sentence-level candidate claims, search status/candidates, and structured evidence analysis status. Final verdict remains `inconclusive`. |
| `/api/image/analyze` | POST | Accepts multipart field `image`; validates it, reports an independent face-quality signal, and reports the AI-image baseline signal when enabled (otherwise `not_run`). Final verdict remains `inconclusive`. |
| `/api/video/analyze` | POST | Returns `not_implemented`; no video ML runs. |

The frontend renders the response contract and contains no ML inference logic. It displays each signal's model ID, prediction, status, raw score when available, inference time when available, and API-provided limitations. `backend/schemas.py` is the single source of truth for result responses.

## Current baseline behavior

### News

The news baseline preserves the V1 model mapping: `LABEL_0` is reported as `fake_news` and `LABEL_1` as `real_news`. This is a text-classifier signal only; it is **not** live fact verification. The search layer defines an injectable `SearchProvider` contract and includes a Tavily adapter for current-web source candidates. The default is `NEWS_SEARCH_PROVIDER=none`, which returns `search.status = "not_configured"`; set it to `tavily` to enable requests.

The evidence layer treats a single input sentence as one candidate claim and splits multiple sentences at terminal punctuation. This is simple deterministic segmentation, not semantic claim extraction. It creates evidence records tied to each claim and source, preserving source title, URL, publisher, publication date, and retrieval time. When `MODEL_INFERENCE_ENABLED=true`, the lazy NLI adapter evaluates retrieved source text as the **premise** and the submitted claim as the **hypothesis** using `cross-encoder/nli-deberta-v3-base`. Its model-label mapping is `0 → contradiction → contradicts`, `1 → entailment → supports`, and `2 → neutral → neutral`. Every evidence record preserves the predicted label, all three softmax probabilities, selected confidence, model ID, status, inference time, and whether source text was truncated.

NLI describes the textual relationship between a source excerpt and a claim. It is not independent fact verification, source credibility assessment, or proof that either statement is true. Relevance remains separately `not_assessed`; no publisher or domain rule influences stance. Before tokenization, source text is deterministically capped by `NEWS_NLI_MAX_SOURCE_CHARS` (default 6000 characters), then the premise/hypothesis pair is tokenized with truncation to `NEWS_NLI_MAX_TOKENS` (default 512, the model maximum). This bounds web-content inputs while retaining each evidence record's source relationship. The final verdict remains `inconclusive` regardless of search or NLI output because evidence fusion is not implemented.

Search configuration is explicit: `NEWS_SEARCH_PROVIDER` defaults to `none`; `NEWS_SEARCH_TIMEOUT_SECONDS` defaults to `5` seconds (accepted range greater than 0 and at most 60); `NEWS_SEARCH_MAX_RESULTS` defaults to `5` (range 1–20); and Tavily reads its credential only from `TAVILY_API_KEY`. If Tavily is selected without that key, search returns an unavailable status. No key is required for tests.

NLI configuration is also explicit: `NEWS_NLI_MODEL_ID` defaults to `cross-encoder/nli-deberta-v3-base`, `NEWS_NLI_MAX_TOKENS` defaults to `512`, and `NEWS_NLI_MAX_SOURCE_CHARS` defaults to `6000`. The NLI tokenizer and model are not imported or loaded at application startup and are not loaded while `MODEL_INFERENCE_ENABLED=false`. The adapter uses CUDA when PyTorch reports availability and otherwise runs on CPU.

### AI images

The AI-image baseline preserves the V1 model mapping: `AI-generated` is reported as `ai_generated` and `human` as `human_created`. This is an individual AI-image classification signal only. Its raw score is not a calibrated probability and is not proof that an image is authentic or synthetic.

For both news and image analysis, the final verdict remains `inconclusive` until a future calibrated evidence-fusion approach is evaluated.

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
- Evidence fusion or evidence aggregation is not implemented.
- Metadata/provenance and forensic analysis are not implemented.
- Advanced multi-claim extraction and source relevance assessment are not implemented.
- Source relevance assessment and source credibility reasoning are not implemented.
- Evidence fusion, current-news final verification, a final Real/Fake decision from web evidence, and final explanation generation are not implemented.
- Frontend evidence presentation is not implemented.
- Video ML analysis is not implemented; the video endpoint is a clear placeholder only.
- No accuracy-improvement or benchmark-performance claim has been made.

## Development rule

Model outputs must be evaluated independently and calibrated before they participate in a future evidence-fusion policy. An “authentic” result must mean only that the examined signals found insufficient evidence of manipulation—not proof of authenticity. No model fusion or evidence aggregation runs in the current implementation.

Uploaded files are not saved by TruthNet application code. Web-server upload handling may use short-lived operating-system temporary storage for multipart requests; production deployment must document retention and enforce request-size, authentication, and rate-limit policies.
