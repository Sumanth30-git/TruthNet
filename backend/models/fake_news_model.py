from backend.models.huggingface import LazyHuggingFaceAdapter
from backend.schemas import ContentType, ModelInfo, SignalResult, SignalStatus


class FakeNewsBaselineAdapter(LazyHuggingFaceAdapter):
    pipeline_task = "text-classification"
    metadata = ModelInfo(
        model_id="jy46604790/Fake-News-Bert-Detect",
        model_version="main",
        content_type=ContentType.TEXT,
    )

    def predict(self, payload: str) -> SignalResult:
        result, inference_time_ms, error = self._run_pipeline(payload)
        if error or result is None:
            return self._failed_signal(inference_time_ms, error or "Model inference failed.")

        original_label = str(result["label"])
        prediction = {"LABEL_0": "fake_news", "LABEL_1": "real_news"}.get(
            original_label, "unrecognized_label"
        )
        return SignalResult(
            model_id=self.metadata.model_id,
            model_version=self.metadata.model_version,
            prediction=prediction,
            raw_score=float(result["score"]),
            calibrated_score=None,
            content_type=ContentType.TEXT,
            status=SignalStatus.COMPLETE,
            inference_time_ms=inference_time_ms,
            details={"original_label": original_label},
        )
