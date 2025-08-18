"""
Transcription Controller - Orchestrates transcription operations
"""

import os
import logging
from typing import Dict, Any, Optional

from app.services.transcription_service import transcription_service
from app.config import settings

logger = logging.getLogger(__name__)


class TranscriptionController:
    """Controller for transcription operations"""

    def __init__(self):
        self.recordings_dir = settings.RECORDING_DIR

    async def create_transcription(
        self, recording_id: str, language: str = "ASL"
    ) -> Dict[str, Any]:
        """
        Create a transcription from a recorded video

        Args:
            recording_id: ID of the recorded video
            language: Language model to use

        Returns:
            Transcription creation result
        """
        try:
            # Validate recording exists
            recording_path = os.path.join(self.recordings_dir, recording_id)

            if not os.path.exists(recording_path):
                logger.warning(f"Recording not found: {recording_id}")
                return {
                    "success": False,
                    "error": "Recording not found",
                    "recording_id": recording_id,
                }

            # Validate file is a video
            if not self._is_video_file(recording_path):
                logger.warning(f"File is not a video: {recording_id}")
                return {
                    "success": False,
                    "error": "File is not a valid video format",
                    "recording_id": recording_id,
                }

            logger.info(f"Creating transcription for recording: {recording_id}")

            # Create transcription using service
            result = transcription_service.create_transcription_from_video(
                recording_path, language
            )

            # Add success flag and recording info
            result["success"] = True
            result["recording_id"] = recording_id

            logger.info(
                f"Transcription created successfully: {result['transcription_id']}"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to create transcription for {recording_id}: {e}")
            return {
                "success": False,
                "error": f"Transcription failed: {str(e)}",
                "recording_id": recording_id,
            }

    async def get_transcription(self, transcription_id: str) -> Dict[str, Any]:
        """
        Get transcription content by ID

        Args:
            transcription_id: ID of the transcription

        Returns:
            Transcription content or error
        """
        try:
            content = transcription_service.get_transcription(transcription_id)

            if content is None:
                return {
                    "success": False,
                    "error": "Transcription not found",
                    "transcription_id": transcription_id,
                }

            return {
                "success": True,
                "transcription_id": transcription_id,
                "content": content,
            }

        except Exception as e:
            logger.error(f"Failed to get transcription {transcription_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to retrieve transcription: {str(e)}",
                "transcription_id": transcription_id,
            }

    async def list_transcriptions(self) -> Dict[str, Any]:
        """
        List all available transcriptions

        Returns:
            List of transcriptions with metadata
        """
        try:
            transcriptions = transcription_service.list_transcriptions()

            return {
                "success": True,
                "transcriptions": transcriptions,
                "count": len(transcriptions),
            }

        except Exception as e:
            logger.error(f"Failed to list transcriptions: {e}")
            return {
                "success": False,
                "error": f"Failed to list transcriptions: {str(e)}",
                "transcriptions": [],
            }

    async def delete_transcription(self, transcription_id: str) -> Dict[str, Any]:
        """
        Delete a transcription

        Args:
            transcription_id: ID of the transcription to delete

        Returns:
            Deletion result
        """
        try:
            success = transcription_service.delete_transcription(transcription_id)

            if success:
                logger.info(f"Transcription deleted: {transcription_id}")
                return {
                    "success": True,
                    "message": f"Transcription {transcription_id} deleted successfully",
                    "transcription_id": transcription_id,
                }
            else:
                return {
                    "success": False,
                    "error": "Transcription not found",
                    "transcription_id": transcription_id,
                }

        except Exception as e:
            logger.error(f"Failed to delete transcription {transcription_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to delete transcription: {str(e)}",
                "transcription_id": transcription_id,
            }

    def _is_video_file(self, file_path: str) -> bool:
        """Check if file is a valid video format"""
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
        _, ext = os.path.splitext(file_path.lower())
        return ext in video_extensions


# Singleton instance
transcription_controller = TranscriptionController()
