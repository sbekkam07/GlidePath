import tempfile
import uuid
from pathlib import Path
from typing import Optional

import cv2
from fastapi import UploadFile


UPLOAD_DIR = Path(tempfile.gettempdir()) / "glidepath_uploads"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "glidepath_outputs"
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


def _offset_series(offsets: list[float], max_count: int) -> list[float]:
    """Normalize offsets for frame preview generation."""
    if max_count <= 0:
        return []

    if not offsets:
        return [0.0] * max_count

    if len(offsets) >= max_count:
        return offsets[:max_count]

    padded = list(offsets)
    while len(padded) < max_count:
        padded.append(offsets[-1] if offsets else 0.0)
    return padded


def _draw_overlay(frame, frame_index: int, alignment: str, offset: float) -> "any":
    """Render centerline + runway mock centerline and alignment label."""
    if frame is None:
        return None

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    h, w = frame.shape[:2]
    frame_center_x = w // 2
    runway_center_x = max(0, min(w - 1, int(frame_center_x + offset)))

    overlay = frame.copy()
    cv2.line(overlay, (frame_center_x, 0), (frame_center_x, h - 1), (0, 255, 0), 2)
    cv2.line(
        overlay,
        (runway_center_x, int(h * 0.2)),
        (runway_center_x, int(h * 0.8)),
        (0, 0, 255),
        2,
    )
    label = f"Frame {frame_index} | Alignment: {alignment}"
    cv2.putText(
        overlay,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"Offset: {offset:.1f}px",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return overlay


def extract_overlay_previews(
    video_path: str,
    alignment: str,
    offsets: list[float],
    sample_interval: int = 20,
    max_frames: int = 10,
    output_dir: Optional[Path] = None,
) -> tuple[list[str], str]:
    """Extract every Nth frame, draw overlays, and save preview images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = output_dir.name if output_dir else f"run_{uuid.uuid4().hex[:12]}"
    run_output_dir = output_dir or (OUTPUT_DIR / run_id)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    preview_offsets = _offset_series(offsets, max_frames)
    saved_paths: list[str] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return saved_paths

    frame_index = 0
    preview_index = 0
    while len(saved_paths) < max_frames:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % sample_interval == 0:
            sample_offset = preview_offsets[min(preview_index, len(preview_offsets) - 1)]
            preview_frame = _draw_overlay(frame, frame_index, alignment, sample_offset)
            out_path = run_output_dir / f"frame_{preview_index + 1:03d}.jpg"
            cv2.imwrite(str(out_path), preview_frame)
            saved_paths.append(f"/outputs/{run_id}/frame_{preview_index + 1:03d}.jpg")
            preview_index += 1

        frame_index += 1

    cap.release()
    return saved_paths, run_id
