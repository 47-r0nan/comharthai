# 🤟 Comharthai API - Deaf Inclusion Tool

This project is a backend API that supports translating sign language (starting with ASL and ISL) into text and speech. It is built using **FastAPI**, and designed to support multiple sign language models in a modular and extensible way.

---

## 🏗️ Features

- 🔤 `/asl/scribe` – Sign to Text (image upload)
- 🔊 `/asl/interpret` – Sign to Speech (image upload + TTS)
- 🧾 `/logs` – Save and retrieve signed translations
- 🧪 Fully tested with `pytest` and FastAPI’s `TestClient`
- 🧠 Built with scalability in mind for model integration, session logging, and future enhancements


## 📦 Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/deaf-inclusion-tool.git
   cd deaf-inclusion-tool
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


## 🚀 Running the API

```bash
uvicorn app.main:app --reload
```

Visit the docs at:
👉 http://localhost:8000/docs


## 🔍 Example Endpoints
`/asl/scribe` (POST)
Upload an image of a hand sign.
Returns the predicted letter (currently dummy logic).

`/asl/interpret` (POST)
Same as `/scribe`, but also speaks the predicted letter using TTS.

`/logs` (POST / GET)
Store and retrieve signed translations. Useful for reviewing interaction history.


## 🧪 Running Tests

```bash
PYTHONPATH=. pytest tests
```

Includes:
- Valid/invalid image cases
- Mocked TTS testing for /interpret
- Full endpoint coverage


## 📂 Project Structure

```bash
app/
├── api/
│   ├── text.py       # /asl/scribe
│   ├── speech.py     # /asl/interpret
│   └── logs.py       # /logs endpoints
├── services/         # Prediction + TTS logic
├── models/           # Pydantic schemas
├── main.py           # FastAPI app entrypoint
tests/                # Unit tests
```


## ⚠️ Known Limitations

- Current prediction uses dummy model logic (`predict_letter()` returns "T").
- Existing ASL models tested were inaccurate — in discussion with ML lecturer to retrain or replace.
- `/logs` uses in-memory storage and will reset on restart.


## 🧠 Future Plans

- ✅ Replace dummy logic with actual ASL/ISL models
- ✅ Use persistent database for logs
- ✅ Real-time webcam support or video upload endpoint
- ✅ Docker support for deployment


## 🙏 Acknowledgements

- [SignLanguageDetectionCNN](https://github.com/cirizzil/SignLanguageDetectionCNN/tree/main) for the initial ASL model base
- FastAPI & Pyttsx3 for powerful backend tools
