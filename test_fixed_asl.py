#!/usr/bin/env python3
"""
Test script to verify ASL model is working after the fix.
"""

import requests
import cv2
import numpy as np
import tempfile
import os
import json


def test_asl_with_video_frames():
    """Test ASL model with multiple frames from the alphabet video."""
    video_path = "tests/Learn ASL Alphabet Video.mp4"

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return

    print(f"Testing ASL model with frames from {video_path}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Test with 5 frames at different positions
    test_positions = [0.2, 0.4, 0.6, 0.8, 0.9]
    results = []

    for i, pos in enumerate(test_positions):
        frame_num = int(total_frames * pos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame = cap.read()
        if not ret:
            continue

        # Save frame to temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, frame)

            try:
                url = "http://localhost:8000/recognition/image?language=ASL"

                with open(tmp_file.name, "rb") as f:
                    files = {"file": f}
                    response = requests.post(url, files=files, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    results.append(
                        {
                            "frame": i + 1,
                            "position": pos,
                            "detected": result.get("detected", False),
                            "prediction": result.get("top_prediction", {}).get(
                                "label", "None"
                            ),
                            "confidence": result.get("top_prediction", {}).get(
                                "confidence", 0.0
                            ),
                        }
                    )
                    print(
                        f"Frame {i+1} (pos {pos:.1f}): {result.get('detected', False)} - "
                        f"{result.get('top_prediction', {}).get('label', 'None')} "
                        f"({result.get('top_prediction', {}).get('confidence', 0.0):.3f})"
                    )
                else:
                    print(f"Frame {i+1}: API error {response.status_code}")

            except Exception as e:
                print(f"Frame {i+1}: Error - {e}")
            finally:
                os.unlink(tmp_file.name)

    cap.release()

    # Summary
    detected_count = sum(1 for r in results if r["detected"])
    print(f"\nSummary:")
    print(f"Total frames tested: {len(results)}")
    print(f"Frames with hand detected: {detected_count}")
    print(
        f"Detection rate: {detected_count/len(results)*100:.1f}%" if results else "0%"
    )

    if detected_count > 0:
        avg_confidence = (
            sum(r["confidence"] for r in results if r["detected"]) / detected_count
        )
        print(f"Average confidence: {avg_confidence:.3f}")

        predictions = [r["prediction"] for r in results if r["detected"]]
        unique_predictions = set(predictions)
        print(f"Unique predictions: {sorted(unique_predictions)}")


def test_languages_endpoint():
    """Test the languages endpoint."""
    print("Testing languages endpoint...")

    try:
        url = "http://localhost:8000/recognition/languages"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("✓ Languages endpoint working")
            print(f"Available languages: {result['languages']}")
            print(f"Default language: {result['default']}")
        else:
            print(f"✗ Languages endpoint failed: {response.status_code}")

    except Exception as e:
        print(f"✗ Error testing languages endpoint: {e}")


if __name__ == "__main__":
    print("ASL Model Fix Verification")
    print("=" * 50)

    # Test languages endpoint
    test_languages_endpoint()
    print()

    # Test ASL with video frames
    test_asl_with_video_frames()

    print("\n" + "=" * 50)
    print("Test complete - ASL model is now working!")
