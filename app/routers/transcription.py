from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
import logging

from app.controllers.transcription_controller import transcription_controller

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/transcription",
    tags=["transcription"],
    responses={404: {"description": "Not found"}},
)


@router.post("/create")
async def create_transcription(
    recording_id: str,
    background_tasks: BackgroundTasks,
    language: str = Query(
        default="ASL", description="Language model to use (ASL, ISL)"
    ),
):
    """
    Create a transcription from a recorded video.

    This endpoint processes a video file frame-by-frame to recognize ASL letters
    and generates a timestamped text transcription.
    """
    try:
        result = await transcription_controller.create_transcription(
            recording_id, language
        )

        if result["success"]:
            return result
        else:
            # Determine appropriate HTTP status code
            if "not found" in result.get("error", "").lower():
                raise HTTPException(status_code=404, detail=result["error"])
            else:
                raise HTTPException(status_code=400, detail=result["error"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_transcription: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/list")
async def list_transcriptions():
    """
    List all available transcriptions with metadata.
    """
    try:
        result = await transcription_controller.list_transcriptions()

        if result["success"]:
            return {
                "transcriptions": result["transcriptions"],
                "count": result["count"],
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in list_transcriptions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{transcription_id}")
async def get_transcription(transcription_id: str):
    """
    Get the content of a specific transcription.
    """
    try:
        result = await transcription_controller.get_transcription(transcription_id)

        if result["success"]:
            return {
                "transcription_id": result["transcription_id"],
                "content": result["content"],
            }
        else:
            if "not found" in result.get("error", "").lower():
                raise HTTPException(status_code=404, detail=result["error"])
            else:
                raise HTTPException(status_code=500, detail=result["error"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_transcription: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{transcription_id}")
async def delete_transcription(transcription_id: str):
    """
    Delete a specific transcription.
    """
    try:
        result = await transcription_controller.delete_transcription(transcription_id)

        if result["success"]:
            return {"status": "success", "message": result["message"]}
        else:
            if "not found" in result.get("error", "").lower():
                raise HTTPException(status_code=404, detail=result["error"])
            else:
                raise HTTPException(status_code=500, detail=result["error"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in delete_transcription: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
