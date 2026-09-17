from backend.models.base import BaseModelAdapter
from backend.schemas import ModelInfo, ModelReadiness


class ModelRegistry:
    """Registry keeps model selection separate from API and pipeline code."""

    def __init__(self) -> None:
        self._models: dict[str, BaseModelAdapter] = {}

    def register(self, key: str, model: BaseModelAdapter) -> None:
        if key in self._models:
            raise ValueError(f"Model key already registered: {key}")
        self._models[key] = model

    def get(self, key: str) -> BaseModelAdapter:
        return self._models[key]

    def available(self) -> list[str]:
        return sorted(self._models)

    def metadata(self) -> list[ModelInfo]:
        return [self._models[key].metadata for key in self.available()]

    def readiness(self) -> list[ModelReadiness]:
        return [self._models[key].readiness() for key in self.available()]


def create_default_registry() -> ModelRegistry:
    """Register adapters without loading their Hugging Face weights."""
    from backend.models.ai_image_model import AIImageBaselineAdapter
    from backend.models.deepfake_model import DeepfakeBaselineAdapter
    from backend.models.fake_news_model import FakeNewsBaselineAdapter

    default_registry = ModelRegistry()
    default_registry.register("fake_news_baseline", FakeNewsBaselineAdapter())
    default_registry.register("ai_image_baseline", AIImageBaselineAdapter())
    default_registry.register("deepfake_baseline", DeepfakeBaselineAdapter())
    return default_registry


registry = create_default_registry()
