# Comharthai 🤟👌 - ASL Recognition API

## Overview
Comharthai is a FastAPI-based web service for American Sign Language (ASL) alphabet recognition. The system uses MediaPipe for hand detection and a pre-trained MobileNetV2 model for real-time ASL letter classification (A-Z).

> **Current Status**: Fully functional ASL alphabet recognition system with 98.6% accuracy on test data. Supports real-time recognition via WebSocket and batch processing of uploaded images/videos.

## Features
- **Real-time ASL Recognition**: WebSocket-based live video processing
- **Image Upload Recognition**: Process individual images for ASL letter detection
- **Video Transcription**: Convert recorded sign language videos to text
- **High Accuracy**: 98.6% accuracy on ASL alphabet dataset using MobileNetV2
- **Prediction Smoothing**: Temporal averaging to reduce prediction jitter
- **Confidence Thresholding**: Only outputs high-confidence predictions (>70%)
- **RESTful API**: Complete FastAPI implementation with automatic documentation

## Technology Stack
- **Backend**: Python 3.8+, FastAPI, Uvicorn
- **Computer Vision**: MediaPipe (hand detection), OpenCV
- **Machine Learning**: PyTorch, MobileNetV2 with attention layers
- **API**: RESTful endpoints + WebSocket for real-time processing

## Model Details
- **Architecture**: Custom MobileNetV2 with channel attention layers
- **Training Data**: ASL Alphabet Dataset
- **Classes**: 26 letters (A-Z)
- **Accuracy**: 98.6% on test set
- **Inference Speed**: Real-time capable (~30 FPS)
- **Model Size**: ~12MB (lightweight for deployment)

## Project Structure
```
comharthai/
├── app/                    # FastAPI application
│   ├── models/            # ASL recognition models
│   │   ├── asl_model.py   # Main ASL model implementation
│   │   ├── mobilenet_asl.py # MobileNetV2 architecture
│   │   └── base_model.py  # Abstract base class
│   ├── routers/           # API endpoints
│   │   ├── recognition.py # Image/video recognition endpoints
│   │   ├── recording.py   # Video recording management
│   │   └── transcription.py # Video-to-text transcription
│   ├── main.py           # FastAPI application entry point
│   └── config.py         # Application configuration
├── models/               # Pre-trained model weights
│   └── weights/         # PyTorch model files
├── data/                # Data storage
│   ├── recordings/      # Uploaded video files
│   └── transcriptions/  # Generated text transcriptions
├── tests/               # Unit and integration tests
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip
- Webcam (for real-time recognition)

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd comharthai
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Running the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Usage

### Available Endpoints

#### Recognition
- `GET /recognition/languages` - List available sign language models
- `POST /recognition/image?language=ASL` - Recognize ASL letters from uploaded image
- `WebSocket /recognition/stream/ASL` - Real-time ASL recognition from video stream

#### Recording Management
- `POST /recording/save` - Save a video recording
- `GET /recording/list` - List all recordings
- `GET /recording/download/{recording_id}` - Download a specific recording
- `DELETE /recording/{recording_id}` - Delete a recording

#### Transcription
- `POST /transcription/create` - Generate text transcription from recorded video
- `GET /transcription/list` - List all transcriptions
- `GET /transcription/{transcription_id}` - Get transcription content
- `DELETE /transcription/{transcription_id}` - Delete a transcription

### Example Usage

#### Recognize ASL from Image
```python
import requests

url = "http://localhost:8000/recognition/image?language=ASL"
files = {"file": open("asl_gesture.jpg", "rb")}
response = requests.post(url, files=files)

result = response.json()
print(f"Detected letter: {result['top_prediction']['label']}")
print(f"Confidence: {result['top_prediction']['confidence']:.2f}")
```

#### Real-time Recognition with WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/recognition/stream/ASL');

ws.onopen = () => {
    console.log('Connected to ASL recognition service');
};

ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    if (result.detected) {
        console.log(`Letter: ${result.top_prediction.label} (${result.top_prediction.confidence})`);
    }
};

// Send video frames as base64-encoded images
function sendFrame(base64Image) {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(base64Image);
    }
}
```

## Model Performance

### Recognition Accuracy
- **Overall Accuracy**: 98.6% on test dataset
- **Real-time Performance**: ~30 FPS on CPU
- **Confidence Threshold**: 0.7 (70%)
- **Prediction Smoothing**: 5-frame temporal averaging

### Supported Gestures
- All 26 ASL alphabet letters (A-Z)
- Static hand gestures only (no motion-based letters)
- Single hand detection (optimized for accuracy)

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- CPU: Any modern processor
- Camera: Any USB webcam (for real-time recognition)

### Recommended Requirements
- Python 3.10+
- 8GB RAM
- GPU: CUDA-compatible (for faster inference)
- Camera: HD webcam for better recognition accuracy

## Limitations

1. **Static Gestures Only**: Currently supports static hand positions (no motion-based letters like J, Z)
2. **Single Hand**: Optimized for single-hand detection
3. **Lighting Sensitivity**: Works best in well-lit environments
4. **Distance Sensitivity**: Optimal performance when hand is 1-2 feet from camera
5. **ASL Only**: Currently trained only on American Sign Language alphabet

## Future Enhancements

- **Irish Sign Language (ISL)** support with additional training data
- **Dynamic gesture recognition** for motion-based letters
- **Word-level recognition** beyond individual letters
- **Multi-hand support** for more complex signs
- **Mobile app integration** via API
- **Real-time video call integration**

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **MediaPipe**: Google's hand detection framework
- **MobileNetV2**: Efficient neural network architecture
- **ASL Dataset**: Training data for alphabet recognition
- **FastAPI**: Modern web framework for building APIs

## Contact

For questions or support, please open an issue in the repository.
