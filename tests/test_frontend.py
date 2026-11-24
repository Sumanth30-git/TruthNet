from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_frontend_and_static_javascript_are_served() -> None:
    page = client.get("/")
    script = client.get("/static/js/app.js")
    assert page.status_code == 200
    assert "TRUTH" in page.text and "NET" in page.text
    assert "Evidence-based analysis with uncertainty" in page.text
    assert "Classifier signal only" not in page.text
    assert script.status_code == 200
    assert "innerHTML" not in script.text
    assert "safeHttpUrl" in script.text
    assert '["http:", "https:"].includes(parsed.protocol)' in script.text
    assert 'parsed.username || parsed.password' in script.text
    assert 'link.rel = "noopener noreferrer"' in script.text
    assert 'link.textContent = title || rawUrl' in script.text
    assert 'Evidence contradicts the claim text' in script.text
    assert 'Evidence unavailable' in script.text
    assert 'Evidence insufficient' in script.text
    assert 'NLI unavailable' in script.text
    assert 'Failed (${retrieval})' in script.text
    assert 'NLI model scores (not truth probabilities)' in script.text
    assert 'data.evidence_summaries.forEach' in script.text
    assert "data.limitations?.length" in script.text
    assert "item.textContent = limitation" in script.text
    assert "signal.inference_time_ms" in script.text
