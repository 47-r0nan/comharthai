# Comharthai 🤟👌 - Irish Sign Language Inclusion Tool

## Overview
Comharthai is a tool designed to help deaf people in corporate and educational environments by translating, transcribing, and recording video calls with a focus on Irish Sign Language (ISL). The name "Comharthai" means "Signs" in Irish Gaelic, reflecting the project's Irish roots.

> **Development Status**: The API infrastructure and endpoints are fully implemented. The sign language recognition models are currently in development, with the architecture in place to easily integrate them once completed.

## Features
- Real-time recognition of sign language alphabets (ISL and ASL supported)
- Translation of sign language to text
- Video recording and storage for later reference
- API for integration with video conferencing tools
- Support for multiple sign languages with easy switching

## Technology Stack
- **Backend**: Python, FastAPI
- **Computer Vision**: MediaPipe, OpenCV
- **Cloud Services**: Azure Cognitive Services
- **Development**: Google Colab (for model training)
- **Deployment**: Docker

## Dataset
The project uses the Irish Sign Language - Hand shape dataset (ISL-HS), which contains:
- 26 hand gestures (23 static, 3 dynamic)
- Data from 6 participants (3 males, 3 females)
- 468 videos total
- 58,114 frames (52,688 for static shapes, 5,426 for dynamic gestures)

## Project Structure
```
comharthai/
├── app/            # FastAPI application
│   ├── models/     # Sign language recognition models
│   ├── routers/    # API endpoints
│   └── config.py   # Application configuration
├── data/           # Dataset and processed data
├── docs/           # Documentation and screenshots
├── models/         # Trained models
├── notebooks/      # Jupyter notebooks for experimentation
├── tests/          # Unit and integration tests
├── utils/          # Utility functions
└── requirements.txt # Python dependencies
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip
- Docker (optional)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/comharthai.git
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

4. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running the API

#### Using Python
```bash
cd comharthai
uvicorn app.main:app --reload
```

#### Using Docker
```bash
docker-compose up
```

### API Documentation
Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Running Tests
To run the test suite:
```bash
python -m unittest discover -s tests
```

## API Usage

### Available Endpoints

#### Recognition
- `GET /recognition/languages` - List available sign language models
- `POST /recognition/image?language=ISL` - Recognize signs from an uploaded image
- `WebSocket /recognition/stream/{language}` - Real-time sign recognition from video stream

#### Recording
- `POST /recording/start` - Start recording a video session
- `POST /recording/stop` - Stop recording and save the video
- `GET /recording/{session_id}` - Get information about a recorded session

#### Transcription
- `POST /transcription/video` - Generate text transcription from a sign language video
- `GET /transcription/{transcription_id}` - Get a transcription by ID

### Example: Recognizing Signs from an Image
```python
import requests

url = "http://localhost:8000/recognition/image?language=ISL"
files = {"file": open("hand_gesture.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Example: Real-time Recognition with WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/recognition/stream/ISL');

ws.onopen = () => {
  console.log('Connected to sign recognition service');
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log('Recognition result:', result);
};

// Send video frames as base64-encoded images
function sendFrame(base64Image) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(base64Image);
  }
}
```

## Adding New Sign Language Models

The system is designed to be extensible. To add a new sign language model:

1. Create a new model class in `app/models/` that inherits from `SignLanguageModel`
2. Register the model in `app/models/model_factory.py`
3. Add the model path to your `.env` file

## License
[To be determined]

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
