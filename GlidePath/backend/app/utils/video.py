import logging
import uuid
from pathlib import Path
from typing import Iterator, Optional

import cv2
from fastapi import UploadFile
import numpy as np

logger = logging.getLogger(__name__)

# Project root is GlidePath/, so backend/app/utils -> GlidePath/backend
BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
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

    logger.info("Saved uploaded video to %s", path)
    return str(path)


def get_video_metadata(filepath: str) -> dict:
    """Read basic video metadata using OpenCV.

    Returns a dict with frame_count, fps, width, and height.
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return {"frame_count": 0, "fps": 0.0, "width": 0, "height": 0}

    metadata = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return metadata


def make_run_output_dir(run_id: str) -> Path:
    run_output_dir = OUTPUT_DIR / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    return run_output_dir


def sample_video_frames(video_path: str, sample_interval: int, max_frames: Optional[int] = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield sampled frames as (frame_index, frame) pairs."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open uploaded video file.")

    frame_index = 0
    sampled = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_index % sample_interval != 0:
                frame_index += 1
                continue

            if not isinstance(frame, np.ndarray) or frame.size == 0:
                logger.warning("Skipping unreadable sampled frame %s from %s", frame_index, video_path)
                frame_index += 1
                continue

            yield frame_index, frame
            sampled += 1

            if max_frames is not None and sampled >= max_frames:
                break

            frame_index += 1
    finally:
        cap.release()
