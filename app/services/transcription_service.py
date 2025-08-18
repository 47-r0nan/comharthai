"""
Transcription Service - Convert video predictions to timestamped text
"""

import os
import uuid
import time
from typing import List, Dict, Any, Optional
import logging

from app.services.video_processing_service import (
    video_processing_service,
    FramePrediction,
)

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Service for converting video predictions to text transcriptions"""

    def __init__(self, transcriptions_dir: str = "data/transcriptions"):
        self.transcriptions_dir = transcriptions_dir
        os.makedirs(transcriptions_dir, exist_ok=True)

        # Transcription parameters
        self.min_confidence = 0.7  # Minimum confidence for including predictions
        self.gap_threshold = 2.0  # Seconds of gap to consider word boundary
        self.min_word_length = 1  # Minimum letters per word

    def create_transcription_from_video(
        self, video_path: str, language: str = "ASL"
    ) -> Dict[str, Any]:
        """
        Create a complete transcription from a video file

        Args:
            video_path: Path to video file
            language: Language model to use

        Returns:
            Transcription metadata and content
        """
        try:
            # Process video to get frame predictions
            logger.info(f"Starting transcription for video: {video_path}")
            frame_predictions = video_processing_service.process_video(
                video_path, language
            )

            # Convert predictions to text
            transcription_content = self._predictions_to_text(frame_predictions)

            # Generate transcription metadata
            transcription_id = f"{uuid.uuid4()}.txt"
            transcription_path = os.path.join(self.transcriptions_dir, transcription_id)

            # Save transcription to file
            self._save_transcription(
                transcription_path, transcription_content, video_path, frame_predictions
            )

            # Get video info for metadata
            video_info = video_processing_service.get_video_info(video_path)

            result = {
                "transcription_id": transcription_id,
                "video_path": video_path,
                "language": language,
                "status": "completed",
                "timestamp": time.time(),
                "stats": {
                    "total_frames_processed": len(frame_predictions),
                    "frames_with_detection": sum(
                        1 for p in frame_predictions if p.detected
                    ),
                    "total_letters_detected": len(
                        [
                            p
                            for p in frame_predictions
                            if p.detected and p.confidence >= self.min_confidence
                        ]
                    ),
                    "video_duration": video_info.get("duration_seconds", 0),
                    "processing_fps": len(frame_predictions)
                    / video_info.get("duration_seconds", 1),
                },
                "content_preview": transcription_content["text"][:200] + "..."
                if len(transcription_content["text"]) > 200
                else transcription_content["text"],
            }

            logger.info(f"Transcription completed: {transcription_id}")
            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def _predictions_to_text(
        self, predictions: List[FramePrediction]
    ) -> Dict[str, Any]:
        """
        Convert frame predictions to structured text with timestamps

        Args:
            predictions: List of frame predictions

        Returns:
            Structured transcription content
        """
        # Filter predictions by confidence
        valid_predictions = [
            p for p in predictions if p.detected and p.confidence >= self.min_confidence
        ]

        if not valid_predictions:
            return {
                "text": "[No sign language detected]",
                "words": [],
                "letters": [],
                "timestamps": [],
            }

        # Group predictions into words based on time gaps
        words = []
        current_word = []
        last_timestamp = 0

        for pred in valid_predictions:
            # Check if there's a significant gap (new word)
            if pred.timestamp - last_timestamp > self.gap_threshold and current_word:
                # Finish current word
                word_text = "".join([p.letter for p in current_word])
                if len(word_text) >= self.min_word_length:
                    words.append(
                        {
                            "text": word_text,
                            "start_time": current_word[0].timestamp,
                            "end_time": current_word[-1].timestamp,
                            "letters": [p.letter for p in current_word],
                            "confidences": [p.confidence for p in current_word],
                        }
                    )
                current_word = []

            current_word.append(pred)
            last_timestamp = pred.timestamp

        # Don't forget the last word
        if current_word:
            word_text = "".join([p.letter for p in current_word])
            if len(word_text) >= self.min_word_length:
                words.append(
                    {
                        "text": word_text,
                        "start_time": current_word[0].timestamp,
                        "end_time": current_word[-1].timestamp,
                        "letters": [p.letter for p in current_word],
                        "confidences": [p.confidence for p in current_word],
                    }
                )

        # Generate final text
        text = " ".join([word["text"] for word in words])

        return {
            "text": text,
            "words": words,
            "letters": [
                {
                    "letter": p.letter,
                    "timestamp": p.timestamp,
                    "confidence": p.confidence,
                }
                for p in valid_predictions
            ],
            "timestamps": [p.timestamp for p in valid_predictions],
        }

    def _save_transcription(
        self,
        file_path: str,
        content: Dict[str, Any],
        video_path: str,
        predictions: List[FramePrediction],
    ):
        """Save transcription to file with metadata"""

        with open(file_path, "w") as f:
            f.write("# ASL Video Transcription\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Video: {os.path.basename(video_path)}\n")
            f.write(f"# Total frames processed: {len(predictions)}\n")
            f.write(
                f"# Frames with detection: {sum(1 for p in predictions if p.detected)}\n"
            )
            f.write("\n## Transcribed Text:\n")
            f.write(content["text"])
            f.write("\n\n## Word-by-Word Breakdown:\n")

            for i, word in enumerate(content["words"], 1):
                f.write(
                    f"{i}. '{word['text']}' ({word['start_time']:.1f}s - {word['end_time']:.1f}s)\n"
                )
                f.write(f"   Letters: {' '.join(word['letters'])}\n")
                f.write(
                    f"   Avg Confidence: {sum(word['confidences'])/len(word['confidences']):.2%}\n\n"
                )

            f.write("\n## Letter-by-Letter Timeline:\n")
            for letter_info in content["letters"]:
                f.write(
                    f"{letter_info['timestamp']:.1f}s: {letter_info['letter']} ({letter_info['confidence']:.2%})\n"
                )

    def get_transcription(self, transcription_id: str) -> Optional[str]:
        """Get transcription content by ID"""
        file_path = os.path.join(self.transcriptions_dir, transcription_id)

        if not os.path.exists(file_path):
            return None

        with open(file_path, "r") as f:
            return f.read()

    def list_transcriptions(self) -> List[Dict[str, Any]]:
        """List all available transcriptions"""
        transcriptions = []

        for filename in os.listdir(self.transcriptions_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.transcriptions_dir, filename)
                transcriptions.append(
                    {
                        "transcription_id": filename,
                        "size_bytes": os.path.getsize(file_path),
                        "created_at": os.path.getctime(file_path),
                        "modified_at": os.path.getmtime(file_path),
                    }
                )

        return transcriptions

    def delete_transcription(self, transcription_id: str) -> bool:
        """Delete a transcription file"""
        file_path = os.path.join(self.transcriptions_dir, transcription_id)

        if os.path.exists(file_path):
            os.remove(file_path)
            return True

        return False


# Singleton instance
transcription_service = TranscriptionService()
