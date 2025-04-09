from fastapi import APIRouter, HTTPException, status
from app.models.schemas import LogEntry, LogResponse

router = APIRouter()

# In-memory store (for now — replace with DB later)
LOG_STORAGE = []


@router.post("/logs", response_model=LogEntry, status_code=status.HTTP_201_CREATED)
def create_log(entry: LogEntry):
    LOG_STORAGE.append(entry)
    return entry


@router.get("/logs", response_model=LogResponse)
def get_logs():
    return LogResponse(logs=LOG_STORAGE)
