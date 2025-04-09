# tests/test_speech.py
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_speech_interpret_with_valid_image():
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    image_bytes = io.BytesIO(buffer.tobytes())

    response = client.post(
        "/asl/interpret",
        files={"image": ("dummy.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert "translation" in response.json()


def test_speech_interpret_with_invalid_image():
    response = client.post(
        "/asl/interpret",
        files={"image": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    assert "detail" in response.json()
