from dataclasses import dataclass
from threading import Lock
from time import perf_counter

from PIL import Image

from backend.schemas import SignalStatus


@dataclass(frozen=True)
class FaceGateResult:
    usable_face: bool
    faces_detected: int
    usable_faces: int
    status: SignalStatus
    inference_time_ms: float | None
    message: str | None = None


class FaceQualityGate:
    """Lazy classical face detector used only to gate face-specific analysis."""

    def __init__(self) -> None:
        self._cascade: object | None = None
        self._lock = Lock()

    def _load_cascade(self) -> object:
        if self._cascade is not None:
            return self._cascade
        with self._lock:
            if self._cascade is not None:
                return self._cascade
            import cv2

            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                raise RuntimeError("Face detector resources are unavailable.")
            self._cascade = cascade
            return cascade

    def analyze(self, image: Image.Image) -> FaceGateResult:
        started = perf_counter()
        try:
            import cv2
            import numpy as np

            grayscale = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            faces = self._load_cascade().detectMultiScale(
                grayscale,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(32, 32),
            )
            usable_faces = sum(width >= 80 and height >= 80 for _, _, width, height in faces)
            return FaceGateResult(
                usable_face=usable_faces > 0,
                faces_detected=len(faces),
                usable_faces=usable_faces,
                status=SignalStatus.COMPLETE,
                inference_time_ms=round((perf_counter() - started) * 1000, 2),
            )
        except Exception:
            return FaceGateResult(
                usable_face=False,
                faces_detected=0,
                usable_faces=0,
                status=SignalStatus.FAILED,
                inference_time_ms=round((perf_counter() - started) * 1000, 2),
                message="Face detection is currently unavailable.",
            )
