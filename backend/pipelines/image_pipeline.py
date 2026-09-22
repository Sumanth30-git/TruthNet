import logging
from typing import Any

from backend.config import FACE_QUALITY_MIN_DIMENSION_PX, model_inference_enabled
from backend.fusion.decision import inconclusive_result
from backend.models.registry import ModelRegistry, registry
from backend.schemas import AnalysisResponse, ContentType, SignalResult, SignalStatus
from backend.utils.face_detection import FaceGateResult, FaceQualityGate
from backend.utils.validation import ValidatedImage

logger = logging.getLogger(__name__)

IMAGE_LIMITATIONS = [
    "This is an individual AI-image classifier signal, not proof of authenticity.",
    "Face quality is a classical detection gate, not identity verification or a deepfake result.",
    "Deepfake analysis is not implemented in this phase.",
    "Metadata/provenance and forensic analysis are not implemented yet.",
    "Raw model scores are not calibrated probabilities.",
]

face_quality_gate = FaceQualityGate()


def _quality_signal(image: ValidatedImage) -> SignalResult:
    prediction = "acceptable" if min(image.width, image.height) >= 128 else "low_resolution"
    return SignalResult(
        model_id="system/image-quality",
        model_version="0.1.0",
        prediction=prediction,
        content_type=ContentType.IMAGE,
        status=SignalStatus.COMPLETE,
        details={"minimum_recommended_dimension": 128},
    )


def _face_quality_signal(result: FaceGateResult) -> SignalResult:
    if result.status == SignalStatus.FAILED:
        prediction = "detection_failed"
    elif result.faces_detected == 0:
        prediction = "no_face_detected"
    elif result.usable_face:
        prediction = "usable_face_detected"
    else:
        prediction = "face_low_quality"

    return SignalResult(
        model_id="system/face-quality-gate",
        model_version="0.1.0",
        prediction=prediction,
        content_type=ContentType.IMAGE,
        status=result.status,
        inference_time_ms=result.inference_time_ms,
        details={
            "faces_detected": result.faces_detected,
            "usable_faces": result.usable_faces,
            "minimum_usable_face_dimension_px": FACE_QUALITY_MIN_DIMENSION_PX,
            "faces": [
                {
                    "x": face.x,
                    "y": face.y,
                    "width": face.width,
                    "height": face.height,
                    "usable": face.usable,
                }
                for face in result.faces
            ],
            **({"message": result.message} if result.message else {}),
        },
    )


def _ai_image_signal(
    *,
    adapter,
    prediction: str,
    status: SignalStatus,
    raw_score: float | None = None,
    inference_time_ms: float | None = None,
    details: dict[str, Any] | None = None,
) -> SignalResult:
    return SignalResult(
        model_id=adapter.metadata.model_id,
        model_version=adapter.metadata.model_version,
        prediction=prediction,
        raw_score=raw_score,
        calibrated_score=None,
        content_type=ContentType.IMAGE,
        status=status,
        inference_time_ms=inference_time_ms,
        details=details or {},
    )


def analyze_image(
    image: ValidatedImage,
    model_registry: ModelRegistry = registry,
) -> AnalysisResponse:
    adapter = model_registry.get("ai_image_baseline")
    face_signal = _face_quality_signal(face_quality_gate.analyze(image.image))
    if not model_inference_enabled():
        ai_signal = _ai_image_signal(
            adapter=adapter,
            prediction="not_assessed",
            status=SignalStatus.NOT_RUN,
            details={"message": "Model inference is disabled."},
        )
    else:
        try:
            ai_signal = adapter.predict(image.image)
        except Exception:
            logger.exception("AI-image baseline inference failed")
            ai_signal = _ai_image_signal(
                adapter=adapter,
                prediction="unavailable",
                status=SignalStatus.FAILED,
                details={"message": "Model inference is currently unavailable."},
            )

    signals = [
        SignalResult(
            model_id="system/input-validation",
            model_version="0.1.0",
            prediction="accepted",
            content_type=ContentType.IMAGE,
            status=SignalStatus.COMPLETE,
            details={
                "mime_type": image.content_type,
                "width": image.width,
                "height": image.height,
                "normalized_mode": image.image.mode,
            },
        ),
        _quality_signal(image),
        face_signal,
        ai_signal,
    ]
    return inconclusive_result(
        "This is an individual AI-image classifier signal, not a fused authenticity verdict.",
        signals,
        model_versions=[adapter.metadata],
        limitations=IMAGE_LIMITATIONS,
    )
