from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.models.base import BaseModelAdapter
from backend.models.fake_news_model import FakeNewsBaselineAdapter
from backend.models.registry import registry
from backend.schemas import ContentType, ModelInfo, SignalResult, SignalStatus

client = TestClient(app)

NEWS_MODEL_ID = "jy46604790/Fake-News-Bert-Detect"


class ScriptedNewsAdapter(BaseModelAdapter):
    metadata = ModelInfo(
        model_id=NEWS_MODEL_ID,
        model_version="main",
        content_type=ContentType.TEXT,
    )

    def __init__(self, signal: SignalResult | None = None, *, fail: bool = False) -> None:
        self.signal = signal
        self.fail = fail
        self.predict_calls: list[str] = []
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def predict(self, payload: str) -> SignalResult:
        self.predict_calls.append(payload)
        if self.fail:
            raise RuntimeError("synthetic adapter failure")
        assert self.signal is not None
        return self.signal


def _complete_signal(prediction: str, raw_score: float = 0.91) -> SignalResult:
    return SignalResult(
        model_id=NEWS_MODEL_ID,
        model_version="main",
        prediction=prediction,
        raw_score=raw_score,
        calibrated_score=None,
        content_type=ContentType.TEXT,
        status=SignalStatus.COMPLETE,
        inference_time_ms=12.5,
        details={"original_label": "LABEL_1" if prediction == "real_news" else "LABEL_0"},
    )


def _enable_scripted_adapter(monkeypatch: pytest.MonkeyPatch, adapter: ScriptedNewsAdapter) -> None:
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "true")
    original_get = registry.get

    def get(key: str) -> BaseModelAdapter:
        if key == "fake_news_baseline":
            return adapter
        return original_get(key)

    monkeypatch.setattr(registry, "get", get)


def test_news_flag_off_returns_not_run_without_loading_model(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = registry.get("fake_news_baseline")

    def should_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("News model must not load or predict when inference is disabled.")

    monkeypatch.setattr(adapter, "predict", should_not_run)
    monkeypatch.setattr(adapter, "load", should_not_run)

    response = client.post("/api/news/analyze", json={"text": "A sample news claim."})
    body = response.json()
    signal = body["signals"][0]
    assert response.status_code == 200
    assert body["verdict"] == "inconclusive"
    assert body["confidence"] is None
    assert signal["model_id"] == NEWS_MODEL_ID
    assert signal["status"] == "not_run"
    assert signal["prediction"] == "not_assessed"
    assert signal["raw_score"] is None
    assert "traceback" not in str(body).lower()
    assert adapter.is_loaded is False


def test_news_flag_on_returns_mocked_complete_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ScriptedNewsAdapter(_complete_signal("real_news", 0.87))
    _enable_scripted_adapter(monkeypatch, adapter)

    response = client.post("/api/news/analyze", json={"text": "A sample news claim."})
    body = response.json()
    signal = body["signals"][0]
    assert response.status_code == 200
    assert body["verdict"] == "inconclusive"
    assert adapter.predict_calls == ["A sample news claim."]
    assert adapter.load_calls == 0
    assert signal["model_id"] == NEWS_MODEL_ID
    assert signal["model_version"] == "main"
    assert signal["prediction"] == "real_news"
    assert signal["raw_score"] == 0.87
    assert signal["status"] == "complete"
    assert signal["inference_time_ms"] == 12.5
    assert signal["calibrated_score"] is None


def test_news_inference_failure_returns_failed_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ScriptedNewsAdapter(fail=True)
    _enable_scripted_adapter(monkeypatch, adapter)

    response = client.post("/api/news/analyze", json={"text": "A sample news claim."})
    body = response.json()
    signal = body["signals"][0]
    assert response.status_code == 200
    assert body["verdict"] == "inconclusive"
    assert signal["status"] == "failed"
    assert signal["prediction"] == "unavailable"
    assert "synthetic adapter failure" not in str(body)
    assert "traceback" not in str(body).lower()


def test_news_rejects_blank_text() -> None:
    response = client.post("/api/news/analyze", json={"text": ""})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    assert response.json()["request_id"]


@pytest.mark.parametrize(
    ("original_label", "expected"),
    [("LABEL_0", "fake_news"), ("LABEL_1", "real_news")],
)
def test_news_label_mapping(monkeypatch: pytest.MonkeyPatch, original_label: str, expected: str) -> None:
    adapter = FakeNewsBaselineAdapter()

    def fake_run(_payload: str) -> tuple[dict[str, Any], float, None]:
        return {"label": original_label, "score": 0.64}, 8.0, None

    monkeypatch.setattr(adapter, "_run_pipeline", fake_run)
    result = adapter.predict("example claim")
    assert result.prediction == expected
    assert result.raw_score == 0.64
    assert result.calibrated_score is None
    assert result.status == SignalStatus.COMPLETE
    assert result.details["original_label"] == original_label
    assert result.inference_time_ms == 8.0
