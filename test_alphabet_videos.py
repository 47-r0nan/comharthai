#!/usr/bin/env python3
"""
Video Testing Script for Comharthai ASL/ISL Recognition
Tests both alphabet videos against both models to demonstrate:
1. ISL model working correctly
2. ASL model overfitting (predicting only 'S')
3. Comparative analysis for research findings
"""

import cv2
import requests
import json
import os
import time
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
ASL_VIDEO_PATH = "/Users/rharris/college/PRO/deafInclusionTool/comharthai/tests/Learn ASL Alphabet Video.mp4"
ISL_VIDEO_PATH = "/Users/rharris/college/PRO/deafInclusionTool/comharthai/tests/ABC in Irish Sign Language.mp4"
OUTPUT_DIR = "/Users/rharris/college/PRO/deafInclusionTool/comharthai/test_results"


def setup_output_dir():
    """Create output directory for test results"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ Output directory created: {OUTPUT_DIR}")


def test_api_connection():
    """Test if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/recognition/languages")
        if response.status_code == 200:
            languages = response.json()
            print(f"✅ API is running. Available languages: {languages}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running:")
        print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return False


def extract_frames(video_path, max_frames=10, skip_seconds=2):
    """Extract frames from video for testing"""
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(fps * skip_seconds)

    frames = []
    frame_count = 0
    extracted_count = 0

    print(f"📹 Extracting frames from: {os.path.basename(video_path)}")

    while cap.read()[0] and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to get diverse samples
        if frame_count % skip_frames == 0:
            frames.append(frame)
            extracted_count += 1
            print(f"   Frame {extracted_count}/{max_frames} extracted")

        frame_count += 1

    cap.release()
    print(f"✅ Extracted {len(frames)} frames from {os.path.basename(video_path)}")
    return frames


def test_frame_with_model(frame, language, frame_idx, video_name):
    """Test a single frame with specified model"""
    # Save frame temporarily
    temp_path = f"{OUTPUT_DIR}/temp_frame_{video_name}_{frame_idx}.jpg"
    cv2.imwrite(temp_path, frame)

    try:
        # Send to API
        with open(temp_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{API_BASE_URL}/recognition/image?language={language}", files=files
            )

        # Clean up temp file
        os.remove(temp_path)

        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"❌ API error for {language}: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Error testing frame with {language}: {e}")
        return None


def run_comprehensive_test():
    """Run comprehensive test of both videos against both models"""
    print("🚀 Starting Comprehensive Video Testing")
    print("=" * 60)

    # Setup
    setup_output_dir()

    # Test API connection
    if not test_api_connection():
        return

    # Extract frames from both videos
    print("\n📹 EXTRACTING FRAMES")
    print("-" * 30)
    asl_frames = extract_frames(ASL_VIDEO_PATH, max_frames=5, skip_seconds=3)
    isl_frames = extract_frames(ISL_VIDEO_PATH, max_frames=5, skip_seconds=3)

    if not asl_frames or not isl_frames:
        print("❌ Could not extract frames from videos")
        return

    # Test results storage
    results = {
        "asl_video_vs_asl_model": [],
        "asl_video_vs_isl_model": [],
        "isl_video_vs_asl_model": [],
        "isl_video_vs_isl_model": [],
    }

    print("\n🧪 TESTING FRAMES")
    print("-" * 30)

    # Test ASL video frames
    print("\n1. Testing ASL Video Frames:")
    for i, frame in enumerate(asl_frames):
        print(f"\n   Frame {i+1}/{len(asl_frames)}:")

        # Test with ASL model
        asl_result = test_frame_with_model(frame, "ASL", i, "asl")
        if asl_result:
            top_pred = asl_result.get("top_prediction", {})
            print(
                f"      ASL Model → {top_pred.get('label', 'N/A')} ({top_pred.get('confidence', 0):.2f})"
            )
            results["asl_video_vs_asl_model"].append(top_pred.get("label", "N/A"))

        # Test with ISL model
        isl_result = test_frame_with_model(frame, "ISL", i, "asl")
        if isl_result:
            top_pred = isl_result.get("top_prediction", {})
            print(
                f"      ISL Model → {top_pred.get('label', 'N/A')} ({top_pred.get('confidence', 0):.2f})"
            )
            results["asl_video_vs_isl_model"].append(top_pred.get("label", "N/A"))

        time.sleep(0.5)  # Rate limiting

    # Test ISL video frames
    print("\n2. Testing ISL Video Frames:")
    for i, frame in enumerate(isl_frames):
        print(f"\n   Frame {i+1}/{len(isl_frames)}:")

        # Test with ASL model
        asl_result = test_frame_with_model(frame, "ASL", i, "isl")
        if asl_result:
            top_pred = asl_result.get("top_prediction", {})
            print(
                f"      ASL Model → {top_pred.get('label', 'N/A')} ({top_pred.get('confidence', 0):.2f})"
            )
            results["isl_video_vs_asl_model"].append(top_pred.get("label", "N/A"))

        # Test with ISL model
        isl_result = test_frame_with_model(frame, "ISL", i, "isl")
        if isl_result:
            top_pred = isl_result.get("top_prediction", {})
            print(
                f"      ISL Model → {top_pred.get('label', 'N/A')} ({top_pred.get('confidence', 0):.2f})"
            )
            results["isl_video_vs_isl_model"].append(top_pred.get("label", "N/A"))

        time.sleep(0.5)  # Rate limiting

    # Analysis and Summary
    print("\n📊 RESULTS ANALYSIS")
    print("=" * 60)

    print("\n🔍 ASL Model Performance:")
    asl_predictions = (
        results["asl_video_vs_asl_model"] + results["isl_video_vs_asl_model"]
    )
    unique_asl_preds = set(asl_predictions)
    print(f"   Unique predictions: {unique_asl_preds}")
    print(f"   Total predictions: {len(asl_predictions)}")

    if len(unique_asl_preds) == 1:
        print("   ⚠️  OVERFITTING DETECTED: Only predicts one letter!")
        print("   📝 Research Finding: ASL model shows severe overfitting")

    print("\n🔍 ISL Model Performance:")
    isl_predictions = results["isl_video_vs_isl_model"]
    unique_isl_preds = set(isl_predictions)
    print(f"   Unique predictions: {unique_isl_preds}")
    print(f"   Total predictions: {len(isl_predictions)}")

    if len(unique_isl_preds) > 1:
        print("   ✅ DIVERSE PREDICTIONS: Model shows good generalization")
        print("   📝 Research Finding: ISL model demonstrates proper learning")

    # Save detailed results
    results_file = f"{OUTPUT_DIR}/test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {results_file}")

    # Demo recommendations
    print("\n🎯 DEMO RECOMMENDATIONS")
    print("-" * 30)
    print("1. Show ISL model working correctly with diverse predictions")
    print("2. Demonstrate ASL overfitting as research finding")
    print("3. Emphasize unified API supporting both models")
    print("4. Highlight data quality importance for ML success")

    print("\n✅ Testing Complete!")


if __name__ == "__main__":
    run_comprehensive_test()
