"""
ASL Recognition Service - Core business logic for sign language recognition
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional
import logging

from app.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class ASLRecognitionService:
    """Service for ASL recognition operations"""

    def __init__(self):
        self._models = {}  # Cache for loaded models

    def get_model(self, language: str = "ASL"):
        """Get or create a model for the specified language"""
        if language not in self._models:
            try:
                self._models[language] = ModelFactory.get_model(language)
                logger.info(f"Loaded {language} model")
            except Exception as e:
                logger.error(f"Failed to load {language} model: {e}")
                raise

        return self._models[language]

    def recognize_from_image(
        self, image: np.ndarray, language: str = "ASL"
    ) -> Dict[str, Any]:
        """
        Recognize ASL from a single image

        Args:
            image: Input image as numpy array
            language: Language model to use

        Returns:
            Recognition result with predictions and metadata
        """
        try:
            model = self.get_model(language)
            result = model.recognize(image)

            # Add service-level metadata
            result["service"] = "asl_recognition"
            result["language"] = language
            result["image_shape"] = image.shape

            return result

        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return {
                "detected": False,
                "message": f"Recognition error: {str(e)}",
                "service": "asl_recognition",
                "language": language,
            }

    def recognize_from_frame_with_timestamp(
        self, frame: np.ndarray, timestamp: float, language: str = "ASL"
    ) -> Dict[str, Any]:
        """
        Recognize ASL from a video frame with timestamp

        Args:
            frame: Video frame as numpy array
            timestamp: Timestamp in seconds
            language: Language model to use

        Returns:
            Recognition result with timestamp
        """
        result = self.recognize_from_image(frame, language)
        result["timestamp"] = timestamp
        result["frame_time"] = f"{timestamp:.2f}s"

        return result

    def get_available_languages(self) -> list:
        """Get list of available recognition languages"""
        return ModelFactory.get_available_languages()

    def get_model_info(self, language: str = "ASL") -> Dict[str, Any]:
        """Get information about a specific model"""
        try:
            model = self.get_model(language)
            return model.get_info()
        except Exception as e:
            logger.error(f"Failed to get model info for {language}: {e}")
            return {"error": str(e)}


# Singleton instance
asl_recognition_service = ASLRecognitionService()
