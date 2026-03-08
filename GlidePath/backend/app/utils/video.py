import tempfile
import uuid
from pathlib import Path
from typing import Optional

import cv2
from fastapi import UploadFile


UPLOAD_DIR = Path(tempfile.gettempdir()) / "glidepath_uploads"
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_valid_video(filename: str, content_type: Optional[str]) -> bool:
    """Return True if a file name or MIME type looks like a video."""
    extension = Path(filename).suffix.lower()
    if extension in ALLOWED_EXTENSIONS:
        return True
    if content_type and content_type.startswith("video/"):
        return True
    return False


async def save_upload(file: UploadFile) -> str:
    """Save uploaded file into a temp directory and return the path."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "").suffix.lower() or ".mp4"
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"

    with path.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            handle.write(chunk)

    return str(path)


def get_video_metadata(filepath: str) -> dict:
    """Read basic video metadata using OpenCV.

    Returns dict with frame_count, fps, width, height.
    Returns sensible defaults if the file cannot be opened.
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return {
            "frame_count": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
        }

    metadata = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return metadata
