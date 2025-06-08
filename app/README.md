# Comharthai API

This directory contains the FastAPI application for the Comharthai sign language recognition system.

## Directory Structure

```
app/
├── models/             # Sign language recognition models
│   ├── __init__.py
│   ├── base_model.py   # Abstract base class for all models
│   ├── asl_model.py    # American Sign Language model
│   ├── isl_model.py    # Irish Sign Language model
│   └── model_factory.py # Factory for creating models
├── routers/            # API endpoints
│   ├── __init__.py
│   ├── recognition.py  # Sign language recognition endpoints
│   ├── recording.py    # Video recording endpoints
│   └── transcription.py # Transcription endpoints
├── __init__.py
├── config.py           # Application configuration
└── main.py             # FastAPI application entry point
```

## How It Works

### Model System

The application uses a flexible model system that supports multiple sign language models:

1. **Base Model**: `SignLanguageModel` defines the interface for all sign language models
2. **Language-Specific Models**:
   - `ASLModel`: American Sign Language recognition
   - `ISLModel`: Irish Sign Language recognition
3. **Model Factory**: `ModelFactory` creates and manages models based on the requested language

### API Endpoints

#### Recognition

- `GET /recognition/languages`: Lists all available sign language models
- `POST /recognition/image?language=ISL`: Recognizes signs from an uploaded image
- `WebSocket /recognition/stream/{language}`: Provides real-time sign recognition from video stream

#### Recording

- `POST /recording/start`: Starts recording a video session
- `POST /recording/stop`: Stops recording and saves the video
- `GET /recording/{session_id}`: Gets information about a recorded session

#### Transcription

- `POST /transcription/video`: Generates text transcription from a sign language video
- `GET /transcription/{transcription_id}`: Gets a transcription by ID

## Adding a New Sign Language Model

To add support for a new sign language:

1. Create a new model class that inherits from `SignLanguageModel`:

```python
from app.models.base_model import SignLanguageModel

class BSLModel(SignLanguageModel):
    """British Sign Language model."""

    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        self.language = "BSL"
        # Initialize model-specific components

    def load_model(self) -> None:
        # Load model implementation
        pass

    def preprocess(self, frame):
        # Preprocess implementation
        pass

    def predict(self, input_data):
        # Prediction implementation
        pass

    def postprocess(self, prediction):
        # Postprocess implementation
        pass
```

2. Register the model in `model_factory.py`:

```python
from app.models.bsl_model import BSLModel

# Add to the _models dictionary in ModelFactory
ModelFactory._models["BSL"] = BSLModel
```

3. Add the model path to your `.env` file:

```
BSL_MODEL_PATH=/path/to/bsl/model
```

## Configuration

The application uses environment variables for configuration. See `.env.example` for available options.

Key configuration options:

- `DEFAULT_LANGUAGE`: Default sign language to use (e.g., "ISL")
- `MODEL_PATHS`: Paths to trained models for each language
- `AZURE_*`: Azure Cognitive Services credentials (if using Azure)
- `RECORDING_DIR`: Directory for storing recorded videos

## Development

To run the application in development mode:

```bash
uvicorn app.main:app --reload
```

This will start the API server with auto-reload enabled.
