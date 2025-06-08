from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.services.inference import predict_letter, speak_letter
from app.models.schemas import ScribeResponse
import cv2
import numpy as np

router = APIRouter()


@router.post(
    "/interpret", response_model=ScribeResponse, status_code=status.HTTP_200_OK
)
async def interpret(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        letter = predict_letter(img)
        speak_letter(letter)
        return ScribeResponse(translation=letter)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
