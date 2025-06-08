# tests/test_text.py
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_scribe_with_valid_image():
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    image_bytes = io.BytesIO(buffer.tobytes())

    response = client.post(
        "/asl/scribe",
        files={"image": ("dummy.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert "translation" in response.json()


def test_scribe_with_invalid_image():
    response = client.post(
        "/asl/scribe",
        files={"image": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    assert "detail" in response.json()
