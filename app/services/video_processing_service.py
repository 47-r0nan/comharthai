"""
Video Processing Service - Core business logic for video processing
"""

import cv2
import os
from typing import List, Dict, Any, Generator, Optional
import logging

from app.services.asl_recognition_service import asl_recognition_service

logger = logging.getLogger(__name__)


class FramePrediction:
    """Data class for frame prediction results"""

    def __init__(self, timestamp: float, prediction: Dict[str, Any]):
        self.timestamp = timestamp
        self.prediction = prediction
        self.detected = prediction.get("detected", False)
        self.letter = (
            prediction.get("top_prediction", {}).get("label", "")
            if self.detected
            else ""
        )
        self.confidence = (
            prediction.get("top_prediction", {}).get("confidence", 0.0)
            if self.detected
            else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "detected": self.detected,
            "letter": self.letter,
            "confidence": self.confidence,
            "full_prediction": self.prediction,
        }


class VideoProcessingService:
    """Service for video processing operations"""

    def __init__(self):
        self.frame_skip = 5  # Process every 5th frame for performance
        self.confidence_threshold = 0.7  # Minimum confidence for valid predictions

    def extract_frames(
        self, video_path: str, max_frames: Optional[int] = None
    ) -> Generator[tuple, None, None]:
        """
        Extract frames from video with timestamps

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (None for all)

        Yields:
            Tuple of (frame, timestamp_seconds)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            processed_count = 0

            logger.info(f"Processing video: {video_path} (FPS: {fps})")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames for performance
                if frame_count % self.frame_skip == 0:
                    timestamp = frame_count / fps
                    yield frame, timestamp
                    processed_count += 1

                    if max_frames and processed_count >= max_frames:
                        break

                frame_count += 1

        finally:
            cap.release()
            logger.info(
                f"Processed {processed_count} frames from {frame_count} total frames"
            )

    def process_video(
        self, video_path: str, language: str = "ASL", max_frames: Optional[int] = None
    ) -> List[FramePrediction]:
        """
        Process entire video and return frame predictions

        Args:
            video_path: Path to video file
            language: Language model to use
            max_frames: Maximum frames to process

        Returns:
            List of frame predictions with timestamps
        """
        predictions = []

        try:
            for frame, timestamp in self.extract_frames(video_path, max_frames):
                # Get prediction for this frame
                result = asl_recognition_service.recognize_from_frame_with_timestamp(
                    frame, timestamp, language
                )

                frame_pred = FramePrediction(timestamp, result)
                predictions.append(frame_pred)

                # Log progress every 10 frames
                if len(predictions) % 10 == 0:
                    logger.info(f"Processed {len(predictions)} frames...")

            logger.info(
                f"Video processing complete: {len(predictions)} frames processed"
            )
            return predictions

        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get basic information about a video file

        Args:
            video_path: Path to video file

        Returns:
            Video metadata
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            info = {
                "path": video_path,
                "filename": os.path.basename(video_path),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT)
                / cap.get(cv2.CAP_PROP_FPS),
                "file_size_bytes": os.path.getsize(video_path),
            }

            return info

        finally:
            cap.release()


# Singleton instance
video_processing_service = VideoProcessingService()
