#!/usr/bin/env python3
"""
Debug script to identify ASL model issues.
"""

import sys
import os
import traceback
import numpy as np
import cv2
import torch

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.models.asl_model import ASLModel


def test_model_loading():
    """Test if the ASL model can be loaded properly."""
    print("=== Testing ASL Model Loading ===")

    try:
        model = ASLModel()
        print("✓ ASL Model instance created")

        model.load_model()
        print("✓ Model loaded successfully")

        # Test if PyTorch model is loaded
        if model.model is not None:
            print("✓ PyTorch model loaded")
            print(f"  Device: {model.device}")
            print(f"  Model type: {type(model.model)}")
        else:
            print("✗ PyTorch model is None")

        # Test if MediaPipe is loaded
        if model.hands is not None:
            print("✓ MediaPipe hands loaded")
        else:
            print("✗ MediaPipe hands is None")

        return model

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        traceback.print_exc()
        return None


def test_model_inference(model):
    """Test model inference with a simple image."""
    print("\n=== Testing Model Inference ===")

    if model is None:
        print("✗ No model to test")
        return

    try:
        # Create a dummy image (224x224x3)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("✓ Created dummy image")

        # Test preprocessing
        processed = model.preprocess(dummy_image)
        print(f"✓ Preprocessing completed: {type(processed)}")

        # Test prediction
        prediction = model.predict(processed)
        print(f"✓ Prediction completed: {prediction}")

        # Test postprocessing
        result = model.postprocess(prediction)
        print(f"✓ Postprocessing completed: {result}")

    except Exception as e:
        print(f"✗ Error in inference: {e}")
        traceback.print_exc()


def test_with_real_image():
    """Test with a real image if available."""
    print("\n=== Testing with Real Image ===")

    # Look for test images
    test_image_paths = [
        "tests/asl_alphabet.mp4",  # We'll extract a frame
        "data/test_image.jpg",
        "test.jpg",
    ]

    for path in test_image_paths:
        if os.path.exists(path):
            print(f"Found test file: {path}")

            if path.endswith(".mp4"):
                # Extract a frame from video
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    print("✓ Extracted frame from video")

                    # Test with the frame
                    model = ASLModel()
                    try:
                        model.load_model()
                        processed = model.preprocess(frame)
                        prediction = model.predict(processed)
                        result = model.postprocess(prediction)
                        print(f"✓ Real image test result: {result}")
                    except Exception as e:
                        print(f"✗ Error with real image: {e}")
                        traceback.print_exc()
                else:
                    print("✗ Could not extract frame from video")
            break
    else:
        print("No test images found")


if __name__ == "__main__":
    print("ASL Model Debug Script")
    print("=" * 50)

    # Test model loading
    model = test_model_loading()

    # Test inference
    test_model_inference(model)

    # Test with real image
    test_with_real_image()

    print("\n" + "=" * 50)
    print("Debug complete")
