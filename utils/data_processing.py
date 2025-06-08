import os
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for hand landmark detection.

    Args:
        image: Input image as numpy array (BGR format)

    Returns:
        Preprocessed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    return thresh


def extract_hand_roi(
    image: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Extract the region of interest (ROI) containing the hand.

    Args:
        image: Input image as numpy array

    Returns:
        Tuple containing:
        - ROI image or None if no hand detected
        - ROI coordinates (x, y, w, h) or None if no hand detected
    """
    # Preprocess the image
    preprocessed = preprocess_image(image)

    # Find contours
    contours, _ = cv2.findContours(
        preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    # Find the largest contour (assuming it's the hand)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add some padding
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    # Extract ROI
    roi = image[y : y + h, x : x + w]

    return roi, (x, y, w, h)


def organize_dataset(source_dir: str, target_dir: str) -> None:
    """
    Organize the ISL-HS dataset into a structured format.

    Args:
        source_dir: Path to the source directory containing the dataset
        target_dir: Path to the target directory where the organized dataset will be stored
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Create directories for static and dynamic gestures
    static_dir = os.path.join(target_dir, "static")
    dynamic_dir = os.path.join(target_dir, "dynamic")
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(dynamic_dir, exist_ok=True)

    # Static letters (A-Y excluding J, X, Z)
    static_letters = "ABCDEFGHIKLMNOPQRSTUVWY"

    # Dynamic letters
    dynamic_letters = "JXZ"

    # Process static letters
    for letter in static_letters:
        letter_dir = os.path.join(static_dir, letter)
        os.makedirs(letter_dir, exist_ok=True)

        # Look for frames in the source directory
        for person_dir in os.listdir(source_dir):
            if person_dir.startswith("Person"):
                person_path = os.path.join(source_dir, person_dir)

                # Check if it's a directory
                if os.path.isdir(person_path):
                    # Look for frames matching this letter
                    for filename in os.listdir(person_path):
                        if filename.startswith(f"{person_dir}-{letter}-"):
                            src_path = os.path.join(person_path, filename)
                            dst_path = os.path.join(letter_dir, filename)

                            # Copy the file
                            try:
                                with open(src_path, "rb") as src_file:
                                    with open(dst_path, "wb") as dst_file:
                                        dst_file.write(src_file.read())
                                logger.info(f"Copied {src_path} to {dst_path}")
                            except Exception as e:
                                logger.error(f"Error copying {src_path}: {str(e)}")

    # Process dynamic letters
    for letter in dynamic_letters:
        letter_dir = os.path.join(dynamic_dir, letter)
        os.makedirs(letter_dir, exist_ok=True)

        # Look for videos in the source directory
        for person_dir in os.listdir(source_dir):
            if person_dir.startswith("Person"):
                person_path = os.path.join(source_dir, person_dir)

                # Check if it's a directory
                if os.path.isdir(person_path):
                    # Look for videos matching this letter
                    for filename in os.listdir(person_path):
                        if filename.lower().startswith(
                            letter.lower()
                        ) and filename.endswith((".mov", ".mp4")):
                            src_path = os.path.join(person_path, filename)
                            dst_path = os.path.join(
                                letter_dir, f"{person_dir}_{filename}"
                            )

                            # Copy the file
                            try:
                                with open(src_path, "rb") as src_file:
                                    with open(dst_path, "wb") as dst_file:
                                        dst_file.write(src_file.read())
                                logger.info(f"Copied {src_path} to {dst_path}")
                            except Exception as e:
                                logger.error(f"Error copying {src_path}: {str(e)}")


def create_train_val_test_split(
    dataset_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Dict[str, List[str]]:
    """
    Create train, validation, and test splits for the dataset.

    Args:
        dataset_dir: Path to the dataset directory
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation

    Returns:
        Dictionary containing file paths for each split
    """
    # Dictionary to store file paths for each split
    splits = {"train": [], "val": [], "test": []}

    # Process static gestures
    static_dir = os.path.join(dataset_dir, "static")
    if os.path.exists(static_dir):
        for letter_dir in os.listdir(static_dir):
            letter_path = os.path.join(static_dir, letter_dir)

            if os.path.isdir(letter_path):
                # Get all files for this letter
                files = [
                    os.path.join(letter_path, f)
                    for f in os.listdir(letter_path)
                    if f.endswith((".jpg", ".jpeg", ".png"))
                ]

                # Shuffle files
                np.random.shuffle(files)

                # Calculate split indices
                n_files = len(files)
                n_train = int(n_files * train_ratio)
                n_val = int(n_files * val_ratio)

                # Split files
                train_files = files[:n_train]
                val_files = files[n_train : n_train + n_val]
                test_files = files[n_train + n_val :]

                # Add to splits
                splits["train"].extend(train_files)
                splits["val"].extend(val_files)
                splits["test"].extend(test_files)

    # Process dynamic gestures
    dynamic_dir = os.path.join(dataset_dir, "dynamic")
    if os.path.exists(dynamic_dir):
        for letter_dir in os.listdir(dynamic_dir):
            letter_path = os.path.join(dynamic_dir, letter_dir)

            if os.path.isdir(letter_path):
                # Get all files for this letter
                files = [
                    os.path.join(letter_path, f)
                    for f in os.listdir(letter_path)
                    if f.endswith((".mov", ".mp4", ".avi"))
                ]

                # Shuffle files
                np.random.shuffle(files)

                # Calculate split indices
                n_files = len(files)
                n_train = int(n_files * train_ratio)
                n_val = int(n_files * val_ratio)

                # Split files
                train_files = files[:n_train]
                val_files = files[n_train : n_train + n_val]
                test_files = files[n_train + n_val :]

                # Add to splits
                splits["train"].extend(train_files)
                splits["val"].extend(val_files)
                splits["test"].extend(test_files)

    # Log split statistics
    logger.info(
        f"Dataset split: {len(splits['train'])} training, {len(splits['val'])} validation, {len(splits['test'])} test"
    )

    return splits


if __name__ == "__main__":
    # Example usage
    source_dir = "../data/raw/ISL-HS"
    target_dir = "../data/processed/ISL-HS"

    # Organize the dataset
    organize_dataset(source_dir, target_dir)

    # Create train/val/test splits
    splits = create_train_val_test_split(target_dir)

    # Save splits to files
    for split_name, file_list in splits.items():
        with open(f"../data/splits/{split_name}.txt", "w") as f:
            for file_path in file_list:
                f.write(f"{file_path}\n")
