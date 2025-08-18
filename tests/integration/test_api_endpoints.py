import pytest
from fastapi.testclient import TestClient
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to Comharthai API - American Sign Language Recognition"
    }


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_recognition_languages(client):
    """Test the recognition languages endpoint."""
    response = client.get("/recognition/languages")
    assert response.status_code == 200
    data = response.json()
    assert "languages" in data
    assert "default" in data
    assert "ASL" in data["languages"]
    assert data["default"] == "ASL"


# Note: The following tests are marked as skipped as they require actual model implementations
# They are included to demonstrate the testing structure but will be skipped


@pytest.mark.skip(reason="Requires implemented models")
def test_recognition_image(client):
    """Test the image recognition endpoint."""
    # This would test uploading an image for recognition
    pass


@pytest.mark.skip(reason="Requires implemented models")
def test_recording_endpoints(client):
    """Test the recording endpoints."""
    # This would test starting and stopping recordings
    pass


@pytest.mark.skip(reason="Requires implemented models")
def test_transcription_endpoints(client):
    """Test the transcription endpoints."""
    # This would test transcribing sign language videos
    pass
