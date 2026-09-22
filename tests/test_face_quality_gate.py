from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.pipelines.image_pipeline import face_quality_gate
from backend.schemas import SignalStatus
from backend.utils.face_detection import FaceBoundingBox, FaceGateResult

client = TestClient(app)


def image_bytes() -> bytes:
    image = Image.new("RGB", (256, 192), color=(24, 80, 120))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _response_for(monkeypatch: pytest.MonkeyPatch, result: FaceGateResult) -> dict:
    monkeypatch.setattr(face_quality_gate, "analyze", lambda _image: result)
    response = client.post(
        "/api/image/analyze",
        files={"image": ("sample.png", image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "inconclusive"
    return next(signal for signal in body["signals"] if signal["model_id"] == "system/face-quality-gate")


def test_face_gate_reports_no_detected_face(monkeypatch: pytest.MonkeyPatch) -> None:
    signal = _response_for(
        monkeypatch,
        FaceGateResult(False, 0, 0, SignalStatus.COMPLETE, 4.2),
    )
    assert signal["prediction"] == "no_face_detected"
    assert signal["status"] == "complete"
    assert signal["details"]["faces"] == []


def test_face_gate_reports_usable_face(monkeypatch: pytest.MonkeyPatch) -> None:
    signal = _response_for(
        monkeypatch,
        FaceGateResult(
            True,
            1,
            1,
            SignalStatus.COMPLETE,
            5.1,
            faces=(FaceBoundingBox(12, 18, 128, 130, True),),
        ),
    )
    assert signal["prediction"] == "usable_face_detected"
    assert signal["details"]["faces_detected"] == 1
    assert signal["details"]["faces"][0] == {
        "x": 12,
        "y": 18,
        "width": 128,
        "height": 130,
        "usable": True,
    }


def test_face_gate_reports_low_quality_face(monkeypatch: pytest.MonkeyPatch) -> None:
    signal = _response_for(
        monkeypatch,
        FaceGateResult(
            False,
            1,
            0,
            SignalStatus.COMPLETE,
            3.7,
            faces=(FaceBoundingBox(4, 5, 48, 52, False),),
        ),
    )
    assert signal["prediction"] == "face_low_quality"
    assert signal["details"]["usable_faces"] == 0
    assert signal["details"]["minimum_usable_face_dimension_px"] == 80


def test_face_gate_reports_detector_failure_without_internal_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal = _response_for(
        monkeypatch,
        FaceGateResult(
            False,
            0,
            0,
            SignalStatus.FAILED,
            1.3,
            message="Face detection is currently unavailable.",
        ),
    )
    assert signal["prediction"] == "detection_failed"
    assert signal["status"] == "failed"
    assert signal["details"]["message"] == "Face detection is currently unavailable."
    assert "traceback" not in str(signal).lower()
