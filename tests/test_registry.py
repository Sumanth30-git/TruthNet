from typing import Any

import pytest

from backend.models.base import BaseModelAdapter
from backend.models.registry import ModelRegistry
from backend.schemas import ContentType, ModelInfo, SignalResult, SignalStatus


class DummyAdapter(BaseModelAdapter):
    metadata = ModelInfo(
        model_id="test/dummy",
        model_version="1.0.0",
        content_type=ContentType.IMAGE,
    )

    def predict(self, payload: Any) -> SignalResult:
        return SignalResult(
            model_id=self.metadata.model_id,
            model_version=self.metadata.model_version,
            prediction="not_assessed",
            content_type=ContentType.IMAGE,
            status=SignalStatus.NOT_RUN,
        )


def test_registry_registers_retrieves_and_reports_metadata() -> None:
    registry = ModelRegistry()
    adapter = DummyAdapter()
    registry.register("dummy", adapter)
    assert registry.get("dummy") is adapter
    assert registry.available() == ["dummy"]
    assert registry.metadata() == [adapter.metadata]


def test_registry_rejects_duplicate_key() -> None:
    registry = ModelRegistry()
    registry.register("dummy", DummyAdapter())
    with pytest.raises(ValueError, match="already registered"):
        registry.register("dummy", DummyAdapter())
