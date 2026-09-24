from types import SimpleNamespace

import pytest
import torch

from backend.config import news_nli_max_source_chars, news_nli_max_tokens, news_nli_model_id
from backend.models.nli_model import NLIModelAdapter
from backend.schemas import (
    EvidenceAnalysisStatus,
    EvidenceRelevance,
    EvidenceStance,
    NLIEvidenceSignal,
    NLIProbabilities,
    SignalStatus,
)
from backend.services.evidence_analysis import EvidenceAnalysisService, NLIEvidenceAnalyzer


class StubNLIAdapter:
    def __init__(self, signal: NLIEvidenceSignal):
        self.signal = signal
        self.calls: list[tuple[object, object]] = []

    def predict(self, premise: object, hypothesis: object) -> NLIEvidenceSignal:
        self.calls.append((premise, hypothesis))
        return self.signal


def test_nli_configuration_uses_environment_with_safe_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEWS_NLI_MODEL_ID", "example/nli-model")
    monkeypatch.setenv("NEWS_NLI_MAX_TOKENS", "256")
    monkeypatch.setenv("NEWS_NLI_MAX_SOURCE_CHARS", "4000")
    assert news_nli_model_id() == "example/nli-model"
    assert news_nli_max_tokens() == 256
    assert news_nli_max_source_chars() == 4000
    monkeypatch.setenv("NEWS_NLI_MAX_TOKENS", "999")
    monkeypatch.setenv("NEWS_NLI_MAX_SOURCE_CHARS", "bad")
    assert news_nli_max_tokens() == 512
    assert news_nli_max_source_chars() == 6000


def _signal(label: str, *, status: SignalStatus = SignalStatus.COMPLETE) -> NLIEvidenceSignal:
    return NLIEvidenceSignal(
        model_id="cross-encoder/nli-deberta-v3-base",
        model_version="main",
        prediction=label,
        status=status,
        probabilities=NLIProbabilities(contradiction=0.1, entailment=0.8, neutral=0.1)
        if status == SignalStatus.COMPLETE else None,
        selected_confidence=0.8 if status == SignalStatus.COMPLETE else None,
        inference_time_ms=12.34 if status == SignalStatus.COMPLETE else None,
        details={"message": "Model inference is disabled."} if status != SignalStatus.COMPLETE else {},
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("entailment", EvidenceStance.SUPPORTS),
        ("contradiction", EvidenceStance.CONTRADICTS),
        ("neutral", EvidenceStance.NEUTRAL),
    ],
)
def test_nli_analyzer_maps_labels_and_preserves_signal(label, expected):
    adapter = StubNLIAdapter(_signal(label))
    assessment = NLIEvidenceAnalyzer(adapter).analyze("The claim.", "Retrieved evidence.")
    assert assessment.stance == expected
    assert assessment.relevance == EvidenceRelevance.NOT_ASSESSED
    assert assessment.status == EvidenceAnalysisStatus.COMPLETE
    assert assessment.nli_signal is adapter.signal
    assert assessment.nli_signal.probabilities.entailment == 0.8
    assert assessment.nli_signal.selected_confidence == 0.8
    assert assessment.nli_signal.inference_time_ms == 12.34
    assert adapter.calls == [("Retrieved evidence.", "The claim.")]


def test_nli_analyzer_handles_disabled_failed_and_unrecognized_signals():
    disabled = NLIEvidenceAnalyzer(StubNLIAdapter(_signal("not_assessed", status=SignalStatus.NOT_RUN)))
    assessment = disabled.analyze("Claim", "Source")
    assert assessment.stance == EvidenceStance.INSUFFICIENT
    assert assessment.status == EvidenceAnalysisStatus.INSUFFICIENT

    failed = NLIEvidenceAnalyzer(StubNLIAdapter(_signal("unavailable", status=SignalStatus.FAILED)))
    assert failed.analyze("Claim", "Source").status == EvidenceAnalysisStatus.UNAVAILABLE

    unknown = NLIEvidenceAnalyzer(StubNLIAdapter(_signal("unexpected")))
    assert unknown.analyze("Claim", "Source").status == EvidenceAnalysisStatus.UNAVAILABLE


def test_nli_model_disabled_does_not_load_or_download(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "false")
    adapter = NLIModelAdapter()
    monkeypatch.setattr(adapter, "load", lambda: pytest.fail("model loading must not run"))
    result = adapter.predict("Source text", "Claim text")
    assert result.status == SignalStatus.NOT_RUN
    assert result.prediction == "not_assessed"
    assert adapter.is_loaded is False


def test_nli_model_loading_failure_is_safe(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "true")
    adapter = NLIModelAdapter()
    adapter._load_error = True
    result = adapter.predict("Source text", "Claim text")
    assert result.status == SignalStatus.FAILED
    assert result.prediction == "unavailable"
    assert "path" not in str(result).lower()


def test_nli_model_handles_empty_inputs_without_loading(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "true")
    adapter = NLIModelAdapter()
    monkeypatch.setattr(adapter, "load", lambda: pytest.fail("empty input must not load the model"))
    assert adapter.predict("", "Claim").status == SignalStatus.FAILED
    assert adapter.predict("Source", " ").status == SignalStatus.FAILED


class _Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, premise, hypothesis, **kwargs):
        self.calls.append((premise, hypothesis, kwargs))
        return {"input_ids": torch.tensor([[1, 2]])}


class _Model:
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, **_inputs):
        return SimpleNamespace(logits=self.logits)


def _loaded_adapter(logits, *, max_source_chars=10):
    adapter = NLIModelAdapter(max_tokens=32, max_source_chars=max_source_chars)
    adapter._tokenizer = _Tokenizer()
    adapter._model = _Model(logits)
    adapter._torch = torch
    return adapter


def test_nli_model_preserves_probabilities_confidence_and_truncation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "true")
    adapter = _loaded_adapter(torch.tensor([[0.0, 3.0, 1.0]]))
    result = adapter.predict("0123456789 beyond", "Claim")
    assert result.status == SignalStatus.COMPLETE
    assert result.prediction == "entailment"
    assert result.probabilities is not None
    assert result.selected_confidence == result.probabilities.entailment
    assert result.source_text_truncated is True
    premise, hypothesis, kwargs = adapter._tokenizer.calls[0]
    assert premise == "0123456789" and hypothesis == "Claim"
    assert kwargs["max_length"] == 32 and kwargs["truncation"] is True


def test_nli_model_malformed_output_and_inference_failure_are_safe(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_INFERENCE_ENABLED", "true")
    malformed = _loaded_adapter(torch.tensor([[1.0, 2.0]]))
    malformed_result = malformed.predict("Source", "Claim")
    assert malformed_result.status == SignalStatus.FAILED
    assert "Malformed" not in str(malformed_result)

    broken = _loaded_adapter(torch.tensor([[1.0, 2.0, 3.0]]))
    broken._tokenizer = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("private path"))
    failed_result = broken.predict("Source", "Claim")
    assert failed_result.status == SignalStatus.FAILED
    assert "private path" not in str(failed_result)


def test_nli_analyzer_remains_replaceable_inside_service():
    adapter = StubNLIAdapter(_signal("neutral"))
    service = EvidenceAnalysisService(NLIEvidenceAnalyzer(adapter))
    assert service.analyzer.adapter is adapter
