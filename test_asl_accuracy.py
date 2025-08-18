#!/usr/bin/env python3
"""
Test ASL model accuracy by extracting more frames and analyzing predictions.
"""

import requests
import cv2
import numpy as np
import tempfile
import os
import json
from collections import Counter


def extract_frames_systematically(video_path, num_frames=10):
    """Extract frames systematically from the video."""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    print(f"Video info: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s duration")

    # Extract frames at regular intervals
    frame_positions = np.linspace(0.1, 0.9, num_frames)
    frames_data = []

    for i, pos in enumerate(frame_positions):
        frame_num = int(total_frames * pos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame = cap.read()
        if ret:
            timestamp = frame_num / fps
            frames_data.append(
                {
                    "index": i,
                    "frame_num": frame_num,
                    "position": pos,
                    "timestamp": timestamp,
                    "frame": frame,
                }
            )

    cap.release()
    return frames_data


def test_frame_with_asl(frame_data):
    """Test a single frame with the ASL model."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, frame_data["frame"])

        try:
            url = "http://localhost:8000/recognition/image?language=ASL"

            with open(tmp_file.name, "rb") as f:
                files = {"file": f}
                response = requests.post(url, files=files, timeout=30)

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "detected": result.get("detected", False),
                    "top_prediction": result.get("top_prediction", {}),
                    "top_3_predictions": result.get("top_3_predictions", []),
                    "raw_result": result,
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "response": response.text,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            os.unlink(tmp_file.name)


def analyze_predictions(results):
    """Analyze the prediction results."""
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS")
    print("=" * 60)

    successful_tests = [r for r in results if r["test_result"]["success"]]
    detected_hands = [r for r in successful_tests if r["test_result"]["detected"]]

    print(f"Total frames tested: {len(results)}")
    print(f"Successful API calls: {len(successful_tests)}")
    print(f"Frames with hands detected: {len(detected_hands)}")

    if len(results) > 0:
        print(f"API success rate: {len(successful_tests)/len(results)*100:.1f}%")
    if len(successful_tests) > 0:
        print(
            f"Hand detection rate: {len(detected_hands)/len(successful_tests)*100:.1f}%"
        )

    if detected_hands:
        print(f"\nPREDICTIONS:")
        print("-" * 40)

        all_predictions = []
        confidence_scores = []

        for r in detected_hands:
            frame_info = r["frame_data"]
            pred = r["test_result"]["top_prediction"]
            top_3 = r["test_result"]["top_3_predictions"]

            letter = pred.get("label", "Unknown")
            confidence = pred.get("confidence", 0.0)

            all_predictions.append(letter)
            confidence_scores.append(confidence)

            print(
                f"Frame {frame_info['index']+1} (t={frame_info['timestamp']:.1f}s): "
                f"{letter} ({confidence:.3f})"
            )

            # Show top 3 predictions
            if len(top_3) > 1:
                top_3_str = ", ".join(
                    [f"{p['label']}({p['confidence']:.3f})" for p in top_3[:3]]
                )
                print(f"    Top 3: {top_3_str}")

        # Summary statistics
        print(f"\nSUMMARY STATISTICS:")
        print("-" * 40)

        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            min_confidence = np.min(confidence_scores)
            max_confidence = np.max(confidence_scores)

            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Confidence range: {min_confidence:.3f} - {max_confidence:.3f}")

        # Letter frequency
        letter_counts = Counter(all_predictions)
        print(f"Letter frequency: {dict(letter_counts)}")
        print(f"Unique letters detected: {len(letter_counts)}")

        # Check if predictions make sense for an alphabet video
        expected_letters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        detected_letters = set(all_predictions)

        print(f"Expected alphabet letters: {len(expected_letters)}")
        print(f"Actually detected letters: {sorted(detected_letters)}")

        if len(detected_letters) > 5:
            print("✓ Good diversity - detecting multiple different letters")
        elif len(detected_letters) > 2:
            print("⚠ Moderate diversity - detecting some different letters")
        else:
            print("✗ Poor diversity - very few different letters detected")


def main():
    video_path = "tests/Learn ASL Alphabet Video.mp4"

    print("ASL Model Accuracy Test")
    print("=" * 60)

    # Extract frames
    print("Extracting frames from video...")
    frames_data = extract_frames_systematically(video_path, num_frames=15)

    if not frames_data:
        print("No frames extracted. Exiting.")
        return

    print(f"Extracted {len(frames_data)} frames")

    # Test each frame
    print("\nTesting frames with ASL model...")
    results = []

    for frame_data in frames_data:
        print(f"Testing frame {frame_data['index']+1}/{len(frames_data)}...", end=" ")

        test_result = test_frame_with_asl(frame_data)
        results.append({"frame_data": frame_data, "test_result": test_result})

        if test_result["success"] and test_result["detected"]:
            pred = test_result["top_prediction"]
            print(f"✓ {pred.get('label', 'Unknown')} ({pred.get('confidence', 0):.3f})")
        elif test_result["success"]:
            print("✓ No hand detected")
        else:
            print(f"✗ {test_result.get('error', 'Unknown error')}")

    # Analyze results
    analyze_predictions(results)


if __name__ == "__main__":
    main()
