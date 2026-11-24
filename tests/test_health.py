import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.models.registry import registry
from backend.services.evidence_analysis import evidence_analysis

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
    nli_models = [item for item in body["models"] if item["model_id"] == "cross-encoder/nli-deberta-v3-base"]
    assert nli_models
    assert nli_models[0]["status"] == "not_loaded"
    assert registry.get("news_nli") is evidence_analysis.analyzer.adapter


def test_health_endpoint_reports_inference_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "true")
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["model_inference_enabled"] is True
