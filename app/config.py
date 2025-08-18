"""
Configuration settings for the Comharthai application.
"""

import os
from pydantic import BaseSettings
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # API settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Comharthai"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Sign language model settings
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "ASL")
    MODEL_PATHS: Dict[str, str] = {
        "ASL": os.getenv(
            "ASL_MODEL_PATH", "models/weights/asl_crop_v4_1_mobilenet_weights.pth"
        ),
        "ISL": os.getenv("ISL_MODEL_PATH", "outputs/models/isl_mnv2_finetuned.keras"),
    }

    # Azure settings (if using Azure services)
    AZURE_SPEECH_KEY: Optional[str] = os.getenv("AZURE_SPEECH_KEY")
    AZURE_SPEECH_REGION: Optional[str] = os.getenv("AZURE_SPEECH_REGION")
    AZURE_VISION_KEY: Optional[str] = os.getenv("AZURE_VISION_KEY")
    AZURE_VISION_ENDPOINT: Optional[str] = os.getenv("AZURE_VISION_ENDPOINT")

    # Video recording settings
    RECORDING_DIR: str = os.getenv("RECORDING_DIR", "data/recordings")
    MAX_RECORDING_LENGTH_SECONDS: int = int(
        os.getenv("MAX_RECORDING_LENGTH_SECONDS", "300")
    )

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Ensure recording directory exists
os.makedirs(settings.RECORDING_DIR, exist_ok=True)
