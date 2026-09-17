from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_video_endpoint_reports_not_implemented() -> None:
    response = client.post("/api/video/analyze")
    body = response.json()
    assert response.status_code == 200
    assert body["verdict"] == "not_implemented"
    assert body["uncertainty"] == "not_assessed"
