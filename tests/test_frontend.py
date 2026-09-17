from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_frontend_and_static_javascript_are_served() -> None:
    page = client.get("/")
    script = client.get("/static/js/app.js")
    assert page.status_code == 200
    assert "TRUTH" in page.text and "NET" in page.text
    assert script.status_code == 200
    assert "innerHTML" not in script.text
    assert "data.limitations?.length" in script.text
    assert "item.textContent = limitation" in script.text
    assert "signal.inference_time_ms" in script.text
