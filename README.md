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
- A flag-gated AI-image baseline adapter for `capcheck/ai-human-generated-image-detection`.
- Server-side image decoding and validation for JPEG, PNG, and WebP files up to 10 MB; corrupt files, MIME mismatches, oversized dimensions, and excessive pixel counts are rejected. Images are normalized to RGB in memory only.
- Stable response schema with verdict, confidence, uncertainty, signals, evidence, limitations, and model versions.
- Lazy model adapters and a registry that report model readiness without loading weights at application startup.
- Safe structured API errors with a request ID; diagnostic details remain in server logs.
- An explicit `inconclusive` final verdict for news and image responses: signals are shown independently and are not fused.

## API contract

| Endpoint | Method | Current behavior |
| --- | --- | --- |
| `/api/health` | GET | Reports application health, whether inference is enabled, and lazy-model readiness. |
| `/api/news/analyze` | POST | Accepts `{ "text": "..." }`; reports the news baseline signal when enabled, otherwise a `not_run` signal. Final verdict remains `inconclusive`. |
| `/api/image/analyze` | POST | Accepts multipart field `image`; validates it and reports the AI-image baseline signal when enabled, otherwise a `not_run` signal. Final verdict remains `inconclusive`. |
| `/api/video/analyze` | POST | Returns `not_implemented`; no video ML runs. |

The frontend renders the response contract and contains no ML inference logic. It displays each signal's model ID, prediction, status, raw score when available, inference time when available, and API-provided limitations. `backend/schemas.py` is the single source of truth for result responses.

## Current baseline behavior

### News

The news baseline preserves the V1 model mapping: `LABEL_0` is reported as `fake_news` and `LABEL_1` as `real_news`. This is a text-classifier signal only; it is **not** live fact verification. No claim extraction, trusted-source retrieval, or supporting/contradicting evidence analysis runs today.

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
- `FaceQualityGate` is not integrated into the image pipeline.
- Evidence fusion or evidence aggregation is not implemented.
- Metadata/provenance and forensic analysis are not implemented.
- Live news verification, source retrieval, claim extraction, and claim-evidence analysis are not implemented.
- Video ML analysis is not implemented; the video endpoint is a clear placeholder only.
- No accuracy-improvement or benchmark-performance claim has been made.

## Development rule

Model outputs must be evaluated independently and calibrated before they participate in a future evidence-fusion policy. An “authentic” result must mean only that the examined signals found insufficient evidence of manipulation—not proof of authenticity. No model fusion or evidence aggregation runs in the current implementation.

Uploaded files are not saved by TruthNet application code. Web-server upload handling may use short-lived operating-system temporary storage for multipart requests; production deployment must document retention and enforce request-size, authentication, and rate-limit policies.
