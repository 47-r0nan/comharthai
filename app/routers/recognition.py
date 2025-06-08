from fastapi import (
    APIRouter,
    File,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    Query,
    Depends,
)
from fastapi.responses import JSONResponse
import logging
import cv2
import numpy as np
import base64
from typing import List, Dict, Any, Optional

from app.models.model_factory import ModelFactory
from app.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/recognition",
    tags=["recognition"],
    responses={404: {"description": "Not found"}},
)

# Store active models to avoid reloading
active_models = {}


def get_model(language: str = Query(None)):
    """
    Dependency to get the appropriate sign language model.

    Args:
        language: Language code (e.g., "ASL", "ISL")

    Returns:
        Initialized sign language model
    """
    language = language or settings.DEFAULT_LANGUAGE

    if language not in active_models:
        model_path = settings.MODEL_PATHS.get(language, "")
        active_models[language] = ModelFactory.get_model(language, model_path)

    return active_models[language]


@router.get("/languages")
async def get_available_languages():
    """Get a list of available sign language models."""
    return {
        "languages": ModelFactory.get_available_languages(),
        "default": settings.DEFAULT_LANGUAGE,
    }


@router.post("/image")
async def recognize_from_image(file: UploadFile = File(...), model=Depends(get_model)):
    """
    Recognize sign language from an uploaded image.

    Args:
        file: Uploaded image file
        model: Sign language model (injected by dependency)

    Returns:
        Recognition results
    """
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400, content={"error": "Invalid image file"}
            )

        # Process the image
        result = model.recognize(image)

        # Add image with landmarks if hand was detected
        if result.get("detected", False):
            _, landmarks = model.preprocess(image)
            if landmarks:
                image_with_landmarks = model.draw_landmarks(image, landmarks)
                _, buffer = cv2.imencode(".jpg", image_with_landmarks)
                encoded_image = base64.b64encode(buffer).decode("utf-8")
                result[
                    "image_with_landmarks"
                ] = f"data:image/jpeg;base64,{encoded_image}"

        return result

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Error processing image: {str(e)}"}
        )


@router.websocket("/stream/{language}")
async def websocket_endpoint(websocket: WebSocket, language: str):
    """
    WebSocket endpoint for real-time sign language recognition.

    Args:
        websocket: WebSocket connection
        language: Language code (e.g., "ASL", "ISL")
    """
    await websocket.accept()

    try:
        # Get the appropriate model
        model_path = settings.MODEL_PATHS.get(language.upper(), "")
        model = ModelFactory.get_model(language.upper(), model_path)

        while True:
            # Receive frame as base64 encoded string
            data = await websocket.receive_text()

            try:
                # Decode base64 image
                encoded_data = data.split(",")[1] if "," in data else data
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({"error": "Invalid frame data"})
                    continue

                # Process the frame
                result = model.recognize(frame)

                # Add frame with landmarks if hand was detected
                if result.get("detected", False):
                    _, landmarks = model.preprocess(frame)
                    if landmarks:
                        frame_with_landmarks = model.draw_landmarks(frame, landmarks)
                        _, buffer = cv2.imencode(".jpg", frame_with_landmarks)
                        encoded_frame = base64.b64encode(buffer).decode("utf-8")
                        result[
                            "frame_with_landmarks"
                        ] = f"data:image/jpeg;base64,{encoded_frame}"

                # Send results back to client
                await websocket.send_json(result)

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                await websocket.send_json(
                    {"error": f"Error processing frame: {str(e)}"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass
