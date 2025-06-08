import pytest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from app.models.model_factory import get_model, list_available_models


class TestModelFactory:
    """Test cases for the model factory module."""

    @patch("app.models.model_factory.ISLModel")
    @patch("app.models.model_factory.ASLModel")
    def test_get_model_isl(self, mock_asl_model, mock_isl_model):
        """Test getting ISL model."""
        # Setup mock
        mock_isl_instance = MagicMock()
        mock_isl_model.return_value = mock_isl_instance

        # Call function
        model = get_model("ISL")

        # Assertions
        assert model == mock_isl_instance
        mock_isl_model.assert_called_once()
        mock_asl_model.assert_not_called()

    @patch("app.models.model_factory.ISLModel")
    @patch("app.models.model_factory.ASLModel")
    def test_get_model_asl(self, mock_asl_model, mock_isl_model):
        """Test getting ASL model."""
        # Setup mock
        mock_asl_instance = MagicMock()
        mock_asl_model.return_value = mock_asl_instance

        # Call function
        model = get_model("ASL")

        # Assertions
        assert model == mock_asl_instance
        mock_asl_model.assert_called_once()
        mock_isl_model.assert_not_called()

    def test_get_model_invalid(self):
        """Test getting an invalid model."""
        with pytest.raises(ValueError):
            get_model("INVALID")

    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models()
        assert isinstance(models, list)
        assert "ISL" in models
        assert "ASL" in models
