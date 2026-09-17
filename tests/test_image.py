from io import BytesIO
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.config import MAX_IMAGE_WIDTH
from backend.main import app
from backend.models.ai_image_model import AIImageBaselineAdapter
from backend.models.base import BaseModelAdapter
from backend.models.registry import registry
from backend.schemas import ContentType, ModelInfo, SignalResult, SignalStatus

client = TestClient(app)

AI_IMAGE_MODEL_ID = "capcheck/ai-human-generated-image-detection"


class ScriptedImageAdapter(BaseModelAdapter):
    metadata = ModelInfo(
        model_id=AI_IMAGE_MODEL_ID,
        model_version="main",
        content_type=ContentType.IMAGE,
    )

    def __init__(self, signal: SignalResult | None = None, *, fail: bool = False) -> None:
        self.signal = signal
        self.fail = fail
        self.predict_calls: list[Image.Image] = []
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def predict(self, payload: Image.Image) -> SignalResult:
        self.predict_calls.append(payload)
        if self.fail:
            raise RuntimeError("synthetic adapter failure")
        assert self.signal is not None
        return self.signal


def image_bytes(image_format: str = "PNG", size: tuple[int, int] = (24, 16)) -> bytes:
    image = Image.new("RGBA", size, color=(24, 80, 120, 128))
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _complete_signal(prediction: str, raw_score: float = 0.91) -> SignalResult:
    return SignalResult(
        model_id=AI_IMAGE_MODEL_ID,
        model_version="main",
        prediction=prediction,
        raw_score=raw_score,
        calibrated_score=None,
        content_type=ContentType.IMAGE,
        status=SignalStatus.COMPLETE,
        inference_time_ms=12.5,
        details={"original_label": "AI-generated" if prediction == "ai_generated" else "human"},
    )


def _enable_scripted_adapter(monkeypatch: pytest.MonkeyPatch, adapter: ScriptedImageAdapter) -> None:
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "true")
    original_get = registry.get

    def get(key: str) -> BaseModelAdapter:
        if key == "ai_image_baseline":
            return adapter
        return original_get(key)

    monkeypatch.setattr(registry, "get", get)


def test_valid_image_returns_structured_inconclusive_response() -> None:
    response = client.post("/api/image/analyze", files={"image": ("sample.png", image_bytes(), "image/png")})
    body = response.json()
    assert response.status_code == 200
    assert body["verdict"] == "inconclusive"
    assert body["signals"][0]["model_id"] == "system/input-validation"
    assert body["signals"][0]["details"]["normalized_mode"] == "RGB"


def test_image_flag_off_returns_not_run_without_loading_model(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = registry.get("ai_image_baseline")

    def should_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("AI-image model must not load or predict when inference is disabled.")

    monkeypatch.setattr(adapter, "predict", should_not_run)
    monkeypatch.setattr(adapter, "load", should_not_run)
    response = client.post("/api/image/analyze", files={"image": ("sample.png", image_bytes(), "image/png")})

    body = response.json()
    signals = {item["model_id"]: item for item in body["signals"]}
    ai_signal = signals[AI_IMAGE_MODEL_ID]
    assert response.status_code == 200
    assert body["verdict"] == "inconclusive"
    assert body["confidence"] is None
    assert signals["system/input-validation"]["status"] == "complete"
    assert signals["system/image-quality"]["status"] == "complete"
    assert ai_signal["status"] == "not_run"
    assert ai_signal["prediction"] == "not_assessed"
    assert ai_signal["raw_score"] is None
    assert "traceback" not in str(body).lower()
    assert adapter.is_loaded is False
    assert all(item["model_id"] != "dima806/deepfake_vs_real_image_detection" for item in body["signals"])


def test_image_flag_on_returns_mocked_complete_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ScriptedImageAdapter(_complete_signal("human_created", 0.87))
    _enable_scripted_adapter(monkeypatch, adapter)

    response = client.post("/api/image/analyze", files={"image": ("sample.png", image_bytes(), "image/png")})
    body = response.json()
    ai_signal = next(item for item in body["signals"] if item["model_id"] == AI_IMAGE_MODEL_ID)
    assert response.status_code == 200
    assert body["verdict"] == "inconclusive"
    assert len(adapter.predict_calls) == 1
    assert adapter.predict_calls[0].mode == "RGB"
    assert adapter.load_calls == 0
    assert ai_signal["model_version"] == "main"
    assert ai_signal["prediction"] == "human_created"
    assert ai_signal["raw_score"] == 0.87
    assert ai_signal["status"] == "complete"
    assert ai_signal["inference_time_ms"] == 12.5
    assert ai_signal["calibrated_score"] is None
    assert all(item["model_id"] != "dima806/deepfake_vs_real_image_detection" for item in body["signals"])


def test_image_inference_failure_returns_failed_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ScriptedImageAdapter(fail=True)
    _enable_scripted_adapter(monkeypatch, adapter)

    response = client.post("/api/image/analyze", files={"image": ("sample.png", image_bytes(), "image/png")})
    body = response.json()
    ai_signal = next(item for item in body["signals"] if item["model_id"] == AI_IMAGE_MODEL_ID)
    assert response.status_code == 200
    assert body["verdict"] == "inconclusive"
    assert ai_signal["status"] == "failed"
    assert ai_signal["prediction"] == "unavailable"
    assert "synthetic adapter failure" not in str(body)
    assert "traceback" not in str(body).lower()


@pytest.mark.parametrize(
    ("original_label", "expected"),
    [("AI-generated", "ai_generated"), ("human", "human_created")],
)
def test_ai_image_label_mapping(monkeypatch: pytest.MonkeyPatch, original_label: str, expected: str) -> None:
    adapter = AIImageBaselineAdapter()

    def fake_run(_payload: Image.Image) -> tuple[dict[str, Any], float, None]:
        return {"label": original_label, "score": 0.64}, 8.0, None

    monkeypatch.setattr(adapter, "_run_pipeline", fake_run)
    result = adapter.predict(Image.new("RGB", (8, 8)))
    assert result.prediction == expected
    assert result.raw_score == 0.64
    assert result.calibrated_score is None
    assert result.status == SignalStatus.COMPLETE
    assert result.details["original_label"] == original_label
    assert result.inference_time_ms == 8.0


def test_image_rejects_corrupt_content() -> None:
    response = client.post("/api/image/analyze", files={"image": ("broken.png", b"not an image", "image/png")})
    assert response.status_code == 400
    assert response.json()["message"] == "Uploaded file is corrupt or is not a valid image."


def test_image_rejects_mime_content_mismatch() -> None:
    response = client.post("/api/image/analyze", files={"image": ("image.jpg", image_bytes(), "image/jpeg")})
    assert response.status_code == 400
    assert "does not match" in response.json()["message"]


def test_image_rejects_unsupported_type() -> None:
    response = client.post("/api/image/analyze", files={"image": ("sample.gif", b"gif", "image/gif")})
    assert response.status_code == 400


def test_image_rejects_oversized_dimensions() -> None:
    response = client.post(
        "/api/image/analyze",
        files={"image": ("wide.png", image_bytes(size=(MAX_IMAGE_WIDTH + 1, 1)), "image/png")},
    )
    assert response.status_code == 400
    assert "dimensions exceed" in response.json()["message"]


def test_image_rejects_excessive_pixel_count() -> None:
    image = Image.new("1", (4_000, 4_000))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    response = client.post("/api/image/analyze", files={"image": ("large.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 400
    assert "pixel safety limit" in response.json()["message"]
