"""Lazy NLI adapter for source-text premise and claim hypothesis comparisons."""

from __future__ import annotations

from threading import Lock
from time import perf_counter
from typing import Any

from backend.config import (
    model_inference_enabled,
    news_nli_max_source_chars,
    news_nli_max_tokens,
    news_nli_model_id,
)
from backend.schemas import NLIEvidenceSignal, NLIProbabilities, SignalStatus


class NLIModelAdapter:
    """Caches a DeBERTa NLI model and produces typed three-way NLI signals.

    The model's documented test mapping is fixed: 0 contradiction, 1 entailment,
    and 2 neutral. It evaluates source text as premise and a submitted claim as
    hypothesis. This assesses textual relation only, never source credibility.
    """

    model_version = "main"
    _LABELS = ("contradiction", "entailment", "neutral")

    def __init__(
        self,
        *,
        model_id: str | None = None,
        max_tokens: int | None = None,
        max_source_chars: int | None = None,
    ) -> None:
        self.model_id = model_id or news_nli_model_id()
        self.max_tokens = max_tokens or news_nli_max_tokens()
        self.max_source_chars = max_source_chars or news_nli_max_source_chars()
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"
        self._load_error = False
        self._lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._tokenizer is not None and self._model is not None

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        if not model_inference_enabled():
            raise RuntimeError("Model inference is disabled.")
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            if self._load_error:
                raise RuntimeError("NLI model is unavailable.")
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
                self._torch = torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model.to(self._device)
                self._model.eval()
            except Exception:
                self._load_error = True
                self._tokenizer = None
                self._model = None
                self._torch = None
                raise RuntimeError("NLI model is unavailable.") from None

    def predict(self, premise: object, hypothesis: object) -> NLIEvidenceSignal:
        if not isinstance(premise, str) or not premise.strip() or not isinstance(hypothesis, str) or not hypothesis.strip():
            return self._signal(
                prediction="not_assessed",
                status=SignalStatus.FAILED,
                message="Claim and source text are required for NLI analysis.",
            )
        if not model_inference_enabled():
            return self._signal(
                prediction="not_assessed",
                status=SignalStatus.NOT_RUN,
                message="Model inference is disabled.",
            )

        source_text = premise.strip()
        source_text_truncated = len(source_text) > self.max_source_chars
        source_text = source_text[: self.max_source_chars]
        started = perf_counter()
        try:
            self.load()
            inputs = self._tokenizer(
                source_text,
                hypothesis.strip(),
                return_tensors="pt",
                truncation=True,
                max_length=self.max_tokens,
            )
            inputs = {name: value.to(self._device) for name, value in inputs.items()}
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            logits = outputs.logits
            if getattr(logits, "shape", None) is None or logits.shape[-1] != len(self._LABELS):
                raise ValueError("Malformed NLI model output")
            values = [float(value) for value in self._torch.softmax(logits, dim=-1)[0].tolist()]
            probabilities = NLIProbabilities(
                contradiction=values[0], entailment=values[1], neutral=values[2]
            )
            index = max(range(len(values)), key=values.__getitem__)
            return NLIEvidenceSignal(
                model_id=self.model_id,
                model_version=self.model_version,
                prediction=self._LABELS[index],
                status=SignalStatus.COMPLETE,
                probabilities=probabilities,
                selected_confidence=values[index],
                inference_time_ms=round((perf_counter() - started) * 1000, 2),
                source_text_truncated=source_text_truncated,
            )
        except Exception:
            return self._signal(
                prediction="unavailable",
                status=SignalStatus.FAILED,
                message="NLI model inference is currently unavailable.",
                inference_time_ms=round((perf_counter() - started) * 1000, 2),
                source_text_truncated=source_text_truncated,
            )

    def _signal(
        self,
        *,
        prediction: str,
        status: SignalStatus,
        message: str,
        inference_time_ms: float | None = None,
        source_text_truncated: bool = False,
    ) -> NLIEvidenceSignal:
        return NLIEvidenceSignal(
            model_id=self.model_id,
            model_version=self.model_version,
            prediction=prediction,
            status=status,
            inference_time_ms=inference_time_ms,
            source_text_truncated=source_text_truncated,
            details={"message": message},
        )
