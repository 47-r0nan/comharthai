#!/usr/bin/env python3
"""
Script to organize the ISL-HS dataset into the recommended structure.
"""

import os
import shutil
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_directory_structure(base_dir):
    """
    Create the recommended directory structure.

    Args:
        base_dir: Base directory for the data
    """
    # Define directories to create
    directories = [
        "raw/Frames",
        "raw/Videos",
        "processed/ISL-HS/static",
        "processed/ISL-HS/dynamic",
        "recordings",
        "transcriptions",
        "splits",
    ]

    # Create directories
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def organize_data(base_dir):
    """
    Organize the data into the recommended structure.

    Args:
        base_dir: Base directory for the data
    """
    # Check if the data is already in the raw directory
    raw_frames_dir = os.path.join(base_dir, "raw", "Frames")
    raw_videos_dir = os.path.join(base_dir, "raw", "Videos")

    # Check if the data is in the base directory
    frames_dir = os.path.join(base_dir, "Frames")
    videos_dir = os.path.join(base_dir, "Videos")

    # Create the new directory structure
    create_directory_structure(base_dir)

    # Check if the data is already in the raw directory
    if os.path.exists(raw_frames_dir) and os.path.exists(raw_videos_dir):
        logger.info("Data is already in the raw directory structure.")
        return True

    # Check if the data is in the base directory
    elif os.path.exists(frames_dir) and os.path.exists(videos_dir):
        # Move Frames to raw/Frames
        logger.info(f"Moving files from {frames_dir} to {raw_frames_dir}")

        # List all items in the source directory
        for item in os.listdir(frames_dir):
            src_path = os.path.join(frames_dir, item)
            dst_path = os.path.join(raw_frames_dir, item)

            # Move the item
            if os.path.isdir(src_path):
                if not os.path.exists(dst_path):
                    shutil.copytree(src_path, dst_path)
                    logger.info(f"Copied directory: {src_path} to {dst_path}")
            else:
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied file: {src_path} to {dst_path}")

        # Move Videos to raw/Videos
        logger.info(f"Moving files from {videos_dir} to {raw_videos_dir}")

        # List all items in the source directory
        for item in os.listdir(videos_dir):
            src_path = os.path.join(videos_dir, item)
            dst_path = os.path.join(raw_videos_dir, item)

            # Move the item
            if os.path.isdir(src_path):
                if not os.path.exists(dst_path):
                    shutil.copytree(src_path, dst_path)
                    logger.info(f"Copied directory: {src_path} to {dst_path}")
            else:
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied file: {src_path} to {dst_path}")

        logger.info("Data organization complete!")
        return True

    else:
        logger.error(
            f"Expected directories not found: neither {frames_dir}/{videos_dir} nor {raw_frames_dir}/{raw_videos_dir}"
        )
        return False


def process_static_dynamic_split(base_dir):
    """
    Process the raw data to separate static and dynamic gestures.

    Args:
        base_dir: Base directory for the data
    """
    # Define paths
    raw_frames_dir = os.path.join(base_dir, "raw", "Frames")
    raw_videos_dir = os.path.join(base_dir, "raw", "Videos")
    static_dir = os.path.join(base_dir, "processed", "ISL-HS", "static")
    dynamic_dir = os.path.join(base_dir, "processed", "ISL-HS", "dynamic")

    # Static letters (A-Y excluding J, X, Z)
    static_letters = "ABCDEFGHIKLMNOPQRSTUVWY"

    # Dynamic letters
    dynamic_letters = "JXZ"

    # Process frames for static letters
    if os.path.exists(raw_frames_dir):
        logger.info("Processing frames for static letters...")

        # Create a directory for each static letter
        for letter in static_letters:
            letter_dir = os.path.join(static_dir, letter)
            os.makedirs(letter_dir, exist_ok=True)

        # Process each person's folder
        for person_dir in os.listdir(raw_frames_dir):
            person_path = os.path.join(raw_frames_dir, person_dir)
            if not os.path.isdir(person_path):
                continue

            # Process each frame file
            for filename in os.listdir(person_path):
                # Extract letter from filename (e.g., "Person1-A-1-1.jpg" -> "A")
                parts = filename.split("-")
                if len(parts) >= 2 and len(parts[1]) >= 1:
                    letter = parts[1][0].upper()  # Take just the first character

                    # Check if it's a static letter
                    if letter in static_letters:
                        src_path = os.path.join(person_path, filename)
                        dst_path = os.path.join(static_dir, letter, filename)

                        # Copy the file
                        try:
                            shutil.copy2(src_path, dst_path)
                            # Uncomment for verbose logging
                            # logger.info(f"Copied {src_path} to {dst_path}")
                        except Exception as e:
                            logger.error(f"Error copying {src_path}: {str(e)}")

    # Process videos for dynamic letters
    if os.path.exists(raw_videos_dir):
        logger.info("Processing videos for dynamic letters...")

        # Create a directory for each dynamic letter
        for letter in dynamic_letters:
            letter_dir = os.path.join(dynamic_dir, letter)
            os.makedirs(letter_dir, exist_ok=True)

        # Process each person's folder
        for person_dir in os.listdir(raw_videos_dir):
            person_path = os.path.join(raw_videos_dir, person_dir)
            if not os.path.isdir(person_path):
                continue

            # Process each video file
            for filename in os.listdir(person_path):
                # Check if it's a dynamic letter video
                for letter in dynamic_letters:
                    if filename.lower().startswith(
                        letter.lower()
                    ) and filename.endswith((".mov", ".mp4")):
                        src_path = os.path.join(person_path, filename)
                        dst_path = os.path.join(
                            dynamic_dir, letter, f"{person_dir}_{filename}"
                        )

                        # Copy the file
                        try:
                            shutil.copy2(src_path, dst_path)
                            logger.info(f"Copied {src_path} to {dst_path}")
                        except Exception as e:
                            logger.error(f"Error copying {src_path}: {str(e)}")

    logger.info("Static/dynamic split processing complete!")


def main():
    # Get the base directory (assuming this script is in utils/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(os.path.dirname(current_dir), "data")

    logger.info(f"Organizing data in: {base_dir}")

    # Organize the data
    if organize_data(base_dir):
        # Process static/dynamic split
        process_static_dynamic_split(base_dir)

        logger.info(
            f"""
Data organization complete! The new structure is:

data/
├── raw/                # Your original dataset
│   ├── Frames/         # Your original Frames folder
│   └── Videos/         # Your original Videos folder
├── processed/          # Processed dataset files
│   └── ISL-HS/
│       ├── static/     # Static gesture images (A-Y excluding J, X, Z)
│       └── dynamic/    # Dynamic gesture videos (J, X, Z)
├── recordings/         # Stored video recordings from the API
├── transcriptions/     # Generated transcriptions from the API
└── splits/             # Train/validation/test splits
"""
        )
    else:
        logger.error("Data organization failed!")


if __name__ == "__main__":
    main()
