import os
import pickle
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional
import logging

# Setup logging
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Class for loading and managing trained models.
    """

    def __init__(self, models_dir: str = "../models"):
        """
        Initialize the model loader.

        Args:
            models_dir: Directory containing the trained models
        """
        self.models_dir = models_dir
        self.static_model = None
        self.dynamic_model = None
        self.static_label_encoder = None
        self.dynamic_label_encoder = None

    def load_static_model(
        self,
        model_filename: str = "isl_recognition_model.h5",
        encoder_filename: str = "label_encoder.pkl",
    ) -> bool:
        """
        Load the static gesture recognition model.

        Args:
            model_filename: Filename of the model
            encoder_filename: Filename of the label encoder

        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = os.path.join(self.models_dir, model_filename)
            encoder_path = os.path.join(self.models_dir, encoder_filename)

            # Check if files exist
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False

            if not os.path.exists(encoder_path):
                logger.error(f"Label encoder file not found: {encoder_path}")
                return False

            # Load model
            self.static_model = tf.keras.models.load_model(model_path)

            # Load label encoder
            with open(encoder_path, "rb") as f:
                self.static_label_encoder = pickle.load(f)

            logger.info(f"Static model loaded successfully from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading static model: {str(e)}")
            return False

    def load_dynamic_model(
        self,
        model_filename: str = "isl_dynamic_gesture_model.h5",
        encoder_filename: str = "dynamic_label_encoder.pkl",
    ) -> bool:
        """
        Load the dynamic gesture recognition model.

        Args:
            model_filename: Filename of the model
            encoder_filename: Filename of the label encoder

        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = os.path.join(self.models_dir, model_filename)
            encoder_path = os.path.join(self.models_dir, encoder_filename)

            # Check if files exist
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False

            if not os.path.exists(encoder_path):
                logger.error(f"Label encoder file not found: {encoder_path}")
                return False

            # Load model
            self.dynamic_model = tf.keras.models.load_model(model_path)

            # Load label encoder
            with open(encoder_path, "rb") as f:
                self.dynamic_label_encoder = pickle.load(f)

            logger.info(f"Dynamic model loaded successfully from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading dynamic model: {str(e)}")
            return False

    def predict_static_gesture(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict a static gesture from features.

        Args:
            features: Hand landmark features

        Returns:
            Tuple containing:
            - Predicted letter
            - Confidence score
        """
        if self.static_model is None or self.static_label_encoder is None:
            logger.error("Static model not loaded")
            return "?", 0.0

        # Reshape features if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Make prediction
        prediction = self.static_model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]

        # Get the letter
        predicted_letter = self.static_label_encoder.inverse_transform(
            [predicted_class]
        )[0]

        return predicted_letter, float(confidence)

    def predict_dynamic_gesture(
        self, features_sequence: np.ndarray
    ) -> Tuple[str, float]:
        """
        Predict a dynamic gesture from a sequence of features.

        Args:
            features_sequence: Sequence of hand landmark features

        Returns:
            Tuple containing:
            - Predicted letter
            - Confidence score
        """
        if self.dynamic_model is None or self.dynamic_label_encoder is None:
            logger.error("Dynamic model not loaded")
            return "?", 0.0

        # Reshape features if needed
        if features_sequence.ndim == 2:
            features_sequence = features_sequence.reshape(
                1, features_sequence.shape[0], features_sequence.shape[1]
            )

        # Make prediction
        prediction = self.dynamic_model.predict(features_sequence)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]

        # Get the letter
        predicted_letter = self.dynamic_label_encoder.inverse_transform(
            [predicted_class]
        )[0]

        return predicted_letter, float(confidence)
