import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint_reports_inference_disabled_by_default() -> None:
    response = client.get("/api/health")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["model_inference_enabled"] is False
    news_models = [item for item in body["models"] if item["model_id"] == "jy46604790/Fake-News-Bert-Detect"]
    assert news_models
    assert news_models[0]["status"] == "not_loaded"


def test_health_endpoint_reports_inference_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "true")
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["model_inference_enabled"] is True
