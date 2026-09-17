import pytest


@pytest.fixture(autouse=True)
def inference_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "false")
