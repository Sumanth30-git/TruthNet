from PIL import Image

from backend.models.huggingface import LazyHuggingFaceAdapter
from backend.schemas import ContentType, ModelInfo, SignalResult, SignalStatus


class AIImageBaselineAdapter(LazyHuggingFaceAdapter):
    pipeline_task = "image-classification"
    metadata = ModelInfo(
        model_id="capcheck/ai-human-generated-image-detection",
        model_version="main",
        content_type=ContentType.IMAGE,
    )

    def predict(self, payload: Image.Image) -> SignalResult:
        result, inference_time_ms, error = self._run_pipeline(payload)
        if error or result is None:
            return self._failed_signal(inference_time_ms, error or "Model inference failed.")

        original_label = str(result["label"])
        prediction = {"AI-generated": "ai_generated", "human": "human_created"}.get(
            original_label, "unrecognized_label"
        )
        return SignalResult(
            model_id=self.metadata.model_id,
            model_version=self.metadata.model_version,
            prediction=prediction,
            raw_score=float(result["score"]),
            calibrated_score=None,
            content_type=ContentType.IMAGE,
            status=SignalStatus.COMPLETE,
            inference_time_ms=inference_time_ms,
            details={"original_label": original_label},
        )
