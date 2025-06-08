from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import logging
import os
import uuid
import time
from typing import List, Dict, Any

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/recording",
    tags=["recording"],
    responses={404: {"description": "Not found"}},
)

# Directory to store recordings
RECORDINGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "recordings"
)
os.makedirs(RECORDINGS_DIR, exist_ok=True)


@router.post("/save")
async def save_recording(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    """
    Save a video recording.
    """
    try:
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.mp4"
        file_path = os.path.join(RECORDINGS_DIR, filename)

        # Save the file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Return the file ID for later retrieval
        return {"recording_id": filename, "timestamp": time.time()}

    except Exception as e:
        logger.error(f"Error saving recording: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving recording: {str(e)}")


@router.get("/list")
async def list_recordings():
    """
    List all available recordings.
    """
    try:
        recordings = []
        for filename in os.listdir(RECORDINGS_DIR):
            if filename.endswith(".mp4"):
                file_path = os.path.join(RECORDINGS_DIR, filename)
                recordings.append(
                    {
                        "recording_id": filename,
                        "size_bytes": os.path.getsize(file_path),
                        "created_at": os.path.getctime(file_path),
                    }
                )

        return {"recordings": recordings}

    except Exception as e:
        logger.error(f"Error listing recordings: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error listing recordings: {str(e)}"
        )


@router.get("/download/{recording_id}")
async def download_recording(recording_id: str):
    """
    Download a specific recording.
    """
    try:
        file_path = os.path.join(RECORDINGS_DIR, recording_id)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Recording not found")

        return FileResponse(file_path, media_type="video/mp4", filename=recording_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading recording: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error downloading recording: {str(e)}"
        )


@router.delete("/{recording_id}")
async def delete_recording(recording_id: str):
    """
    Delete a specific recording.
    """
    try:
        file_path = os.path.join(RECORDINGS_DIR, recording_id)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Recording not found")

        os.remove(file_path)
        return {"status": "success", "message": f"Recording {recording_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting recording: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting recording: {str(e)}"
        )
