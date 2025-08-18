#!/usr/bin/env python3
"""
Direct API test to identify ASL recognition issues.
"""

import requests
import cv2
import numpy as np
import tempfile
import os
import json


def create_test_image():
    """Create a simple test image with a hand-like shape."""
    # Create a 640x480 image
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw a simple hand-like shape (rectangle for simplicity)
    cv2.rectangle(img, (200, 150), (400, 350), (255, 255, 255), -1)
    cv2.circle(img, (300, 100), 30, (255, 255, 255), -1)  # Thumb
    cv2.circle(img, (250, 120), 25, (255, 255, 255), -1)  # Index
    cv2.circle(img, (300, 110), 25, (255, 255, 255), -1)  # Middle
    cv2.circle(img, (350, 120), 25, (255, 255, 255), -1)  # Ring
    cv2.circle(img, (380, 140), 20, (255, 255, 255), -1)  # Pinky

    return img


def test_api_with_image(image, filename="test_image.jpg"):
    """Test the API with a given image."""
    print(f"Testing API with {filename}")

    # Save image to temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, image)

        try:
            # Test ASL endpoint
            url = "http://localhost:8000/recognition/image?language=ASL"

            with open(tmp_file.name, "rb") as f:
                files = {"file": f}
                response = requests.post(url, files=files, timeout=30)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")

            if response.status_code == 200:
                result = response.json()
                print("✓ API call successful")
                print(f"Result: {json.dumps(result, indent=2)}")
            else:
                print(f"✗ API call failed: {response.status_code}")
                print(f"Error: {response.text}")

        except requests.exceptions.Timeout:
            print("✗ Request timed out")
        except requests.exceptions.ConnectionError:
            print("✗ Connection error - is the server running?")
        except Exception as e:
            print(f"✗ Error: {e}")
        finally:
            # Clean up temp file
            os.unlink(tmp_file.name)


def test_with_video_frame():
    """Test with a frame from the ASL alphabet video."""
    video_path = "tests/Learn ASL Alphabet Video.mp4"

    if os.path.exists(video_path):
        print(f"Testing with frame from {video_path}")

        cap = cv2.VideoCapture(video_path)

        # Skip to middle of video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

        ret, frame = cap.read()
        cap.release()

        if ret:
            test_api_with_image(frame, "video_frame.jpg")
        else:
            print("✗ Could not extract frame from video")
    else:
        print(f"Video file not found: {video_path}")


def test_languages_endpoint():
    """Test the languages endpoint."""
    print("Testing languages endpoint")

    try:
        url = "http://localhost:8000/recognition/languages"
        response = requests.get(url, timeout=10)

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✓ Languages endpoint successful")
            print(f"Available languages: {result}")
        else:
            print(f"✗ Languages endpoint failed: {response.text}")

    except Exception as e:
        print(f"✗ Error testing languages endpoint: {e}")


if __name__ == "__main__":
    print("Direct API Test Script")
    print("=" * 50)

    # Test languages endpoint first
    test_languages_endpoint()
    print()

    # Test with simple synthetic image
    print("Creating synthetic test image...")
    test_image = create_test_image()
    test_api_with_image(test_image, "synthetic_hand.jpg")
    print()

    # Test with video frame
    test_with_video_frame()

    print("\n" + "=" * 50)
    print("API test complete")
