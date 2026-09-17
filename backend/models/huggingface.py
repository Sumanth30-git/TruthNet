from threading import Lock
from time import perf_counter
from typing import Any

from backend.config import model_inference_enabled
from backend.models.base import BaseModelAdapter
from backend.schemas import ModelLoadStatus, SignalResult, SignalStatus


class LazyHuggingFaceAdapter(BaseModelAdapter):
    """Caches one Transformers pipeline per adapter and loads it only on demand."""

    pipeline_task: str

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._load_error: str | None = None
        self._lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def load_status(self) -> ModelLoadStatus:
        if self.is_loaded:
            return ModelLoadStatus.LOADED
        if self._load_error:
            return ModelLoadStatus.FAILED
        return ModelLoadStatus.NOT_LOADED

    @property
    def load_message(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        if not model_inference_enabled():
            raise RuntimeError("Model inference is disabled.")
        if self._pipeline is not None:
            return
        with self._lock:
            if self._pipeline is not None:
                return
            if self._load_error:
                raise RuntimeError(self._load_error)
            try:
                from transformers import pipeline

                self._pipeline = pipeline(
                    self.pipeline_task,
                    model=self.metadata.model_id,
                    device=self._pipeline_device(),
                )
            except Exception:
                self._load_error = "Model could not be loaded. Check model access, dependencies, and available memory."
                raise RuntimeError(self._load_error) from None

    def _pipeline_device(self) -> int:
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1

    def _run_pipeline(self, payload: Any) -> tuple[dict[str, Any] | None, float | None, str | None]:
        started = perf_counter()
        try:
            self.load()
            result = self._pipeline(payload)[0]
            return result, round((perf_counter() - started) * 1000, 2), None
        except Exception:
            elapsed = round((perf_counter() - started) * 1000, 2)
            return None, elapsed, "Model inference is currently unavailable."

    def _failed_signal(self, inference_time_ms: float | None, message: str) -> SignalResult:
        return SignalResult(
            model_id=self.metadata.model_id,
            model_version=self.metadata.model_version,
            prediction="unavailable",
            content_type=self.metadata.content_type,
            status=SignalStatus.FAILED,
            inference_time_ms=inference_time_ms,
            details={"message": message},
        )
