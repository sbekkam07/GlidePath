import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

import cv2
from fastapi import UploadFile

from ..services.runway_geometry import draw_runway_overlay


LOGGER = logging.getLogger(__name__)

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


def inspect_video_file(filepath: str | Path) -> dict[str, Any]:
    """Inspect whether a generated video exists and can be decoded."""
    path = Path(filepath)
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": 0,
        "readable": False,
        "frame_count": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
    }
    if not path.exists():
        return info

    try:
        info["size_bytes"] = int(path.stat().st_size)
    except OSError:
        return info

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return info

    info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info["fps"] = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    success, first_frame = cap.read()
    info["readable"] = bool(
        success and first_frame is not None and getattr(first_frame, "size", 0) > 0
    )
    cap.release()
    return info


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


def _empty_render_geometry(width: int, bbox: list[int] | None = None) -> dict:
    return {
        "bbox": list(bbox) if bbox else None,
        "left_edge": None,
        "right_edge": None,
        "centerline": None,
        "center_x_bottom": None,
        "image_center_x": width // 2,
        "signed_offset_px": None,
        "runway_polygon": None,
        "geometry_confidence": 0.0,
    }


def _prepare_overlay_geometry(
    width: int,
    detection: dict | None = None,
    fallback_offset: float | None = None,
) -> tuple[dict, float | None]:
    detector_confidence = None
    geometry_for_render = _empty_render_geometry(width)

    if detection is not None and isinstance(detection, dict):
        detector_confidence = (
            float(detection.get("confidence", 0.0) or 0.0)
            if detection.get("confidence") is not None
            else None
        )
        geometry = detection.get("geometry")
        if isinstance(geometry, dict):
            geometry_for_render = dict(geometry)
        if geometry_for_render.get("bbox") is None and isinstance(detection.get("bbox"), list):
            geometry_for_render["bbox"] = list(detection["bbox"])

    if geometry_for_render.get("signed_offset_px") is None and isinstance(fallback_offset, (int, float)):
        geometry_for_render["signed_offset_px"] = float(fallback_offset)
    if geometry_for_render.get("image_center_x") is None:
        geometry_for_render["image_center_x"] = width // 2

    return geometry_for_render, detector_confidence


def render_overlay_frame(
    frame,
    frame_index: int,
    detection: dict | None = None,
    alignment: str | None = None,
    fallback_offset: float | None = None,
) -> "any":
    """Render runway overlays in a single shared path used by video + previews."""
    if frame is None:
        return None
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    _, width = frame.shape[:2]
    geometry_for_render, detector_confidence = _prepare_overlay_geometry(
        width,
        detection=detection,
        fallback_offset=fallback_offset,
    )

    return draw_runway_overlay(
        frame,
        geometry_for_render,
        frame_index=frame_index,
        alignment=alignment,
        detector_confidence=detector_confidence,
    )


def _draw_overlay(
    frame,
    frame_index: int,
    alignment: str,
    offset: float,
    detection: dict | None = None,
) -> "any":
    """Compatibility wrapper for preview generation."""
    return render_overlay_frame(
        frame,
        frame_index=frame_index,
        detection=detection,
        alignment=alignment,
        fallback_offset=offset,
    )


def extract_overlay_previews(
    video_path: str,
    alignment: str,
    offsets: list[float],
    detections: Optional[list[dict]] = None,
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
    detections_by_frame = {
        d.get("frame", idx): d
        for idx, d in enumerate(detections or [])
        if isinstance(d, dict)
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        LOGGER.warning("Preview extraction failed: could not open video '%s'", video_path)
        return saved_paths, run_id

    frame_index = 0
    preview_index = 0
    while len(saved_paths) < max_frames:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % sample_interval == 0:
            sample_offset = preview_offsets[min(preview_index, len(preview_offsets) - 1)]
            detection = detections_by_frame.get(frame_index) if detections_by_frame else None
            preview_frame = _draw_overlay(frame, frame_index, alignment, sample_offset, detection=detection)
            out_path = run_output_dir / f"frame_{preview_index + 1:03d}.jpg"
            cv2.imwrite(str(out_path), preview_frame)
            saved_paths.append(f"/outputs/{run_id}/frame_{preview_index + 1:03d}.jpg")
            preview_index += 1

        frame_index += 1

    cap.release()
    return saved_paths, run_id
