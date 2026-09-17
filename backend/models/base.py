from abc import ABC, abstractmethod
from typing import Any

from backend.schemas import ModelInfo, ModelLoadStatus, ModelReadiness, SignalResult


class BaseModelAdapter(ABC):
    """Common interface for future, independently evaluated model adapters."""

    metadata: ModelInfo

    @property
    def is_loaded(self) -> bool:
        """Adapters override this when they support lazy model loading."""
        return False

    @property
    def load_status(self) -> ModelLoadStatus:
        return ModelLoadStatus.LOADED if self.is_loaded else ModelLoadStatus.NOT_LOADED

    @property
    def load_message(self) -> str | None:
        return None

    def readiness(self) -> ModelReadiness:
        return ModelReadiness(
            **self.metadata.model_dump(),
            status=self.load_status,
            message=self.load_message,
        )

    def load(self) -> None:
        """Load model resources only when a future pipeline needs them."""
        raise NotImplementedError("This adapter does not implement model loading yet.")

    @abstractmethod
    def predict(self, payload: Any) -> SignalResult:
        """Return a typed, normalized signal result."""
