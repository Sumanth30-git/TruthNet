from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_unknown_route_returns_safe_structured_error_with_request_id() -> None:
    response = client.get("/api/does-not-exist")
    body = response.json()
    assert response.status_code == 404
    assert body["error"] == "request_error"
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert "traceback" not in str(body).lower()
