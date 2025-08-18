#!/usr/bin/env python3
"""
ASL Alphabet Video Testing Script
Tests the ASL alphabet video against the ASL model from GitHub
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
OUTPUT_DIR = "/Users/rharris/college/PRO/deafInclusionTool/comharthai/asl_test_results"


def setup_output_dir():
    """Create output directory for test results"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ Output directory: {OUTPUT_DIR}")


def test_api_connection():
    """Test if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/recognition/languages")
        if response.status_code == 200:
            languages = response.json()
            print(f"✅ API running. Languages: {languages['languages']}")
            return True
        else:
            print(f"❌ API error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Start server with:")
        print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return False


def extract_asl_frames(max_frames=10, skip_seconds=2):
    """Extract frames from ASL alphabet video"""
    if not os.path.exists(ASL_VIDEO_PATH):
        print(f"❌ ASL video not found: {ASL_VIDEO_PATH}")
        return []

    cap = cv2.VideoCapture(ASL_VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"📹 ASL Video: {duration:.1f}s, {fps:.1f} FPS, {total_frames} frames")

    skip_frames = int(fps * skip_seconds)
    frames = []
    frame_count = 0
    extracted_count = 0

    while cap.read()[0] and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to get diverse samples across the alphabet
        if frame_count % skip_frames == 0:
            frames.append(frame)
            extracted_count += 1
            timestamp = frame_count / fps
            print(f"   Frame {extracted_count}: {timestamp:.1f}s")

        frame_count += 1

    cap.release()
    print(f"✅ Extracted {len(frames)} frames from ASL alphabet video")
    return frames


def test_frame_with_asl(frame, frame_idx):
    """Test a single frame with ASL model"""
    # Save frame temporarily
    temp_path = f"{OUTPUT_DIR}/asl_frame_{frame_idx}.jpg"
    cv2.imwrite(temp_path, frame)

    try:
        # Send to ASL API
        with open(temp_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{API_BASE_URL}/recognition/image?language=ASL",
                files=files,
                timeout=10,  # Add timeout to handle potential hangs
            )

        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"❌ API error: {response.status_code}")
            if response.text:
                print(f"   Error details: {response.text[:200]}...")
            return {"error": f"HTTP {response.status_code}"}

    except requests.exceptions.Timeout:
        print(f"❌ Request timeout")
        return {"error": "Timeout"}
    except Exception as e:
        print(f"❌ Error testing frame: {e}")
        return {"error": str(e)}


def run_asl_alphabet_test():
    """Run comprehensive ASL alphabet video test"""
    print("🚀 ASL ALPHABET VIDEO TESTING")
    print("=" * 50)

    # Setup
    setup_output_dir()

    # Test API
    if not test_api_connection():
        return

    # Extract frames
    print(f"\n📹 EXTRACTING FRAMES FROM ASL VIDEO")
    print("-" * 30)
    frames = extract_asl_frames(max_frames=10, skip_seconds=2)

    if not frames:
        print("❌ No frames extracted")
        return

    # Test each frame
    print(f"\n🧪 TESTING {len(frames)} FRAMES WITH ASL MODEL")
    print("-" * 30)

    results = []
    predictions = []
    errors = []

    for i, frame in enumerate(frames):
        print(f"\nFrame {i+1}/{len(frames)}:")

        result = test_frame_with_asl(frame, i + 1)
        if result and "error" not in result:
            top_pred = result.get("top_prediction", {})
            label = top_pred.get("label", "N/A")
            confidence = top_pred.get("confidence", 0)

            print(f"   🎯 Prediction: {label} (confidence: {confidence:.3f})")

            results.append(result)
            predictions.append(label)

            # Show top 3 predictions if available
            all_preds = result.get("all_predictions", [])
            if len(all_preds) > 1:
                print(f"   📊 Top 3: ", end="")
                for j, pred in enumerate(all_preds[:3]):
                    print(f"{pred['label']}({pred['confidence']:.2f})", end="")
                    if j < min(2, len(all_preds) - 1):
                        print(", ", end="")
                print()
        else:
            error_msg = (
                result.get("error", "Unknown error") if result else "No response"
            )
            print(f"   ❌ Failed: {error_msg}")
            predictions.append("ERROR")
            errors.append(error_msg)

        time.sleep(0.5)  # Rate limiting

    # Analysis
    print(f"\n📊 RESULTS ANALYSIS")
    print("=" * 50)

    successful_predictions = [p for p in predictions if p != "ERROR" and p != "N/A"]
    unique_predictions = set(successful_predictions)
    error_count = len([p for p in predictions if p == "ERROR"])

    print(f"✅ Successful predictions: {len(successful_predictions)}/{len(frames)}")
    print(f"❌ Errors/failures: {error_count}/{len(frames)}")
    print(f"🎯 Unique letters recognized: {len(unique_predictions)}")
    print(f"📝 Letters detected: {sorted(unique_predictions)}")

    if error_count > len(frames) // 2:
        print(f"⚠️  HIGH ERROR RATE: ASL model appears unstable")

    if len(unique_predictions) <= 1:
        print(f"⚠️  OVERFITTING DETECTED: Limited letter diversity")
        print(f"   This suggests the ASL model has poor generalization")
    elif len(unique_predictions) > 5:
        print(f"🎉 SUCCESS: ASL model shows good diversity!")
    else:
        print(f"📊 MODERATE: Some diversity but room for improvement")

    # Error analysis
    if errors:
        print(f"\n❌ ERROR BREAKDOWN:")
        error_types = {}
        for error in errors:
            error_types[error] = error_types.get(error, 0) + 1
        for error_type, count in error_types.items():
            print(f"   {error_type}: {count} times")

    # Save results
    results_file = f"{OUTPUT_DIR}/asl_test_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "video_path": ASL_VIDEO_PATH,
                "frames_tested": len(frames),
                "successful_predictions": len(successful_predictions),
                "error_count": error_count,
                "unique_letters": list(unique_predictions),
                "all_predictions": predictions,
                "errors": errors,
                "detailed_results": results,
            },
            f,
            indent=2,
        )

    print(f"\n💾 Results saved to: {results_file}")

    # Comparison with ISL
    print(f"\n🔄 COMPARISON INSIGHTS")
    print("-" * 30)
    print("Now we can compare ASL vs ISL model performance:")
    print("- ASL model (GitHub): [Results above]")
    print("- ISL model (Custom): 8/8 success, 2 unique letters (M, W)")
    print("This gives us real comparative data for our research!")

    print(f"\n🎬 READY FOR ANALYSIS!")


if __name__ == "__main__":
    run_asl_alphabet_test()
