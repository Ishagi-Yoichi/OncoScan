import base64
from io import BytesIO

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
pytest.importorskip("PIL")

from fastapi.testclient import TestClient
from PIL import Image

from oncoscan.api import create_app


def _sample_png() -> bytes:
    image = Image.new("RGB", (64, 64), color=(80, 110, 140))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_gradcam_payload() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/predict",
            data={"scan_type": "brain"},
            files={"image": ("scan.png", _sample_png(), "image/png")},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["scan_type"] == "brain"
    assert 0 <= payload["confidence"] <= 1
    assert set(payload["probabilities"]) == {"no_tumor", "tumor"}
    assert base64.b64decode(payload["heatmap_png_base64"])
    assert base64.b64decode(payload["overlay_png_base64"])
