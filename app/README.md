# Comharthai API Application

## Overview
This directory contains the FastAPI application for the Comharthai ASL recognition system. The application provides RESTful endpoints and WebSocket connections for real-time sign language recognition.

## Architecture

### Core Components

#### Models (`models/`)
- **`base_model.py`**: Abstract base class defining the interface for all sign language models
- **`asl_model.py`**: American Sign Language recognition implementation using MediaPipe + PyTorch
- **`mobilenet_asl.py`**: MobileNetV2 architecture with channel attention layers
- **`model_factory.py`**: Factory pattern for creating and managing different language models
- **`isl_model.py`**: Irish Sign Language model (placeholder implementation)

#### Routers (`routers/`)
- **`recognition.py`**: Image upload and real-time WebSocket recognition endpoints
- **`recording.py`**: Video recording management (save, list, download, delete)
- **`transcription.py`**: Video-to-text transcription services

#### Configuration (`config.py`)
- Environment-based settings using Pydantic
- Model paths and API configuration
- Azure services configuration (optional)

## API Endpoints

### Recognition Endpoints
```
GET  /recognition/languages           # List available models
POST /recognition/image?language=ASL  # Upload image for recognition
WS   /recognition/stream/{language}   # Real-time recognition
```

### Recording Endpoints
```
POST   /recording/save                # Save uploaded video
GET    /recording/list                # List all recordings
GET    /recording/download/{id}       # Download specific recording
DELETE /recording/{id}                # Delete recording
```

### Transcription Endpoints
```
POST   /transcription/create          # Create transcription from video
GET    /transcription/list            # List all transcriptions
GET    /transcription/{id}            # Get transcription content
DELETE /transcription/{id}            # Delete transcription
```

## Model Implementation

### ASL Model Pipeline
1. **Hand Detection**: MediaPipe detects hand landmarks (21 points per hand)
2. **Preprocessing**: Crop hand region, resize to 224x224, normalize
3. **Classification**: MobileNetV2 with attention layers predicts letter (A-Z)
4. **Postprocessing**: Apply confidence thresholding and temporal smoothing

### Key Features
- **Prediction Smoothing**: Averages predictions over 5 frames to reduce jitter
- **Confidence Thresholding**: Only outputs predictions above 70% confidence
- **Real-time Processing**: Optimized for ~30 FPS performance
- **Single Hand Focus**: Optimized for single-hand detection for better accuracy

## Usage Examples

### Loading a Model
```python
from app.models.model_factory import ModelFactory

# Create ASL model
model = ModelFactory.get_model("ASL", "models/weights/asl_crop_v4_1_mobilenet_weights.pth")

# Recognize from image
import cv2
image = cv2.imread("hand_gesture.jpg")
result = model.recognize(image)
print(result)
```

### API Client Example
```python
import requests

# Image recognition
url = "http://localhost:8000/recognition/image?language=ASL"
files = {"file": open("gesture.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

if result["detected"]:
    print(f"Letter: {result['top_prediction']['label']}")
    print(f"Confidence: {result['top_prediction']['confidence']:.2f}")
```

### WebSocket Client Example
```python
import asyncio
import websockets
import json
import base64
import cv2

async def recognize_realtime():
    uri = "ws://localhost:8000/recognition/stream/ASL"
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if ret:
                # Encode frame as base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                # Send to server
                await websocket.send(frame_b64)

                # Receive result
                result = await websocket.recv()
                data = json.loads(result)

                if data.get("detected"):
                    print(f"Recognized: {data['top_prediction']['label']}")

asyncio.run(recognize_realtime())
```

## Configuration

### Environment Variables
```bash
# Application settings
DEBUG=False
DEFAULT_LANGUAGE=ASL

# Model paths
ASL_MODEL_PATH=models/weights/asl_crop_v4_1_mobilenet_weights.pth
ISL_MODEL_PATH=

# Recording settings
RECORDING_DIR=data/recordings
MAX_RECORDING_LENGTH_SECONDS=300

# Azure services (optional)
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
AZURE_VISION_KEY=
AZURE_VISION_ENDPOINT=
```

### Model Requirements
- PyTorch model weights must be compatible with the CustomMobileNetV2 architecture
- Models should output 26 classes (A-Z letters)
- Input images should be 224x224 RGB

## Error Handling

### Common Error Responses
```json
{
  "detected": false,
  "message": "No hand detected",
  "predictions": {}
}
```

```json
{
  "detected": false,
  "message": "Error in prediction",
  "predictions": {}
}
```

### HTTP Error Codes
- `400`: Invalid image file or malformed request
- `404`: Recording/transcription not found
- `500`: Internal server error (model loading, prediction errors)

## Performance Considerations

### Optimization Tips
- Use GPU when available (CUDA support)
- Batch process multiple images when possible
- Implement caching for frequently accessed models
- Use appropriate image compression for WebSocket streams

### Resource Usage
- **Memory**: ~500MB for loaded ASL model
- **CPU**: ~10-20% for real-time recognition
- **Storage**: ~12MB per model file

## Testing

### Unit Tests
```bash
python -m pytest tests/unit/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### Manual Testing
```bash
# Test model loading
python -c "from app.models.asl_model import ASLModel; m = ASLModel(); m.load_model(); print('✅ Model loaded')"

# Test API endpoints
curl -X GET http://localhost:8000/recognition/languages
```

## Extending the System

### Adding New Languages
1. Create new model class inheriting from `SignLanguageModel`
2. Implement required methods: `load_model()`, `preprocess()`, `predict()`, `postprocess()`
3. Register in `ModelFactory`
4. Add model path to configuration

### Custom Model Integration
1. Ensure model outputs 26 classes (A-Z)
2. Implement preprocessing to match expected input format
3. Add confidence scoring and smoothing logic
4. Test with various hand positions and lighting conditions
