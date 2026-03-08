import logging
import math
import tempfile
import uuid
from collections import deque
from pathlib import Path
from statistics import median
from typing import Optional

import cv2
from fastapi import UploadFile


UPLOAD_DIR = Path(tempfile.gettempdir()) / "glidepath_uploads"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "glidepath_outputs"
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
LOGGER = logging.getLogger(__name__)

# Guidance labels and tuning. Raw offset alone was too biased/noisy across long clips,
# so this uses a short recent window with a rolling baseline + hysteresis.
GUIDANCE_ALIGNED = "ALIGNED"
GUIDANCE_CORRECT_LEFT = "CORRECT LEFT"
GUIDANCE_CORRECT_RIGHT = "CORRECT RIGHT"
GUIDANCE_RECENT_WINDOW = 8
GUIDANCE_BASELINE_WINDOW = 30
GUIDANCE_MIN_BASELINE_SAMPLES = 6
GUIDANCE_ENTER_THRESHOLD = 0.04
GUIDANCE_EXIT_THRESHOLD = 0.022
GUIDANCE_SWITCH_THRESHOLD = 0.055
GUIDANCE_CONFIRM_FRAMES = 2
GUIDANCE_MIN_WIDTH_PX = 240.0
MVP_GUIDANCE_ENTER_PX = 20.0
MVP_GUIDANCE_WIDTH_RATIO = 0.04
MVP_BOX_COLOR = (0, 170, 255)
MVP_REFERENCE_LINE_COLOR = (170, 170, 170)
MVP_RUNWAY_LINE_COLOR = (0, 210, 255)


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


def _new_guidance_state() -> dict:
    return {
        "normalized_offsets": deque(maxlen=GUIDANCE_BASELINE_WINDOW),
        "label": GUIDANCE_ALIGNED,
        "pending_label": None,
        "pending_count": 0,
    }


def _normalize_offset(offset_px: float, frame_width: int) -> float:
    normalizer = max(float(frame_width), GUIDANCE_MIN_WIDTH_PX)
    return float(offset_px) / normalizer


def _compute_guidance_signal(normalized_offsets: deque[float]) -> float:
    if not normalized_offsets:
        return 0.0
    recent = list(normalized_offsets)[-GUIDANCE_RECENT_WINDOW:]
    recent_center = float(median(recent))

    # Rolling median baseline reduces persistent detector bias so guidance stays reactive.
    if len(normalized_offsets) >= GUIDANCE_MIN_BASELINE_SAMPLES:
        baseline = float(median(normalized_offsets))
    else:
        baseline = 0.0

    return recent_center - baseline


def _apply_hysteresis(signal: float, current_label: str) -> str:
    abs_signal = abs(signal)

    # Hysteresis: harder to enter correction than to stay, and easier to return to ALIGNED.
    if current_label == GUIDANCE_ALIGNED:
        if signal >= GUIDANCE_ENTER_THRESHOLD:
            return GUIDANCE_CORRECT_RIGHT
        if signal <= -GUIDANCE_ENTER_THRESHOLD:
            return GUIDANCE_CORRECT_LEFT
        return GUIDANCE_ALIGNED

    if current_label == GUIDANCE_CORRECT_RIGHT:
        if signal <= -GUIDANCE_SWITCH_THRESHOLD:
            return GUIDANCE_CORRECT_LEFT
        if abs_signal <= GUIDANCE_EXIT_THRESHOLD:
            return GUIDANCE_ALIGNED
        return GUIDANCE_CORRECT_RIGHT

    if current_label == GUIDANCE_CORRECT_LEFT:
        if signal >= GUIDANCE_SWITCH_THRESHOLD:
            return GUIDANCE_CORRECT_RIGHT
        if abs_signal <= GUIDANCE_EXIT_THRESHOLD:
            return GUIDANCE_ALIGNED
        return GUIDANCE_CORRECT_LEFT

    return GUIDANCE_ALIGNED


def get_guidance_label(
    signed_offset_px: float,
    frame_width: int,
    guidance_state: dict,
) -> str:
    normalized_offsets = guidance_state["normalized_offsets"]
    normalized_offsets.append(_normalize_offset(signed_offset_px, frame_width))
    signal = _compute_guidance_signal(normalized_offsets)
    current_label = guidance_state["label"]
    proposed_label = _apply_hysteresis(signal, current_label)

    if proposed_label == current_label:
        guidance_state["pending_label"] = None
        guidance_state["pending_count"] = 0
        return current_label

    if guidance_state["pending_label"] == proposed_label:
        guidance_state["pending_count"] += 1
    else:
        guidance_state["pending_label"] = proposed_label
        guidance_state["pending_count"] = 1

    if guidance_state["pending_count"] >= GUIDANCE_CONFIRM_FRAMES:
        guidance_state["label"] = proposed_label
        guidance_state["pending_label"] = None
        guidance_state["pending_count"] = 0

    return guidance_state["label"]


def _draw_guidance_box(frame, label: str):
    h, w = frame.shape[:2]
    bg_color = (46, 160, 67) if label == GUIDANCE_ALIGNED else (0, 120, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    text_thickness = 1
    pad_x = 10
    pad_y = 6
    margin = 14

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    box_w = text_w + (pad_x * 2)
    box_h = text_h + baseline + (pad_y * 2)
    x1 = max(0, w - box_w - margin)
    y1 = max(0, h - box_h - margin)
    x2 = min(w - 1, x1 + box_w)
    y2 = min(h - 1, y1 + box_h)

    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (x1 + pad_x, y2 - baseline - pad_y),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def _draw_overlay(
    frame,
    frame_index: int,
    alignment: str,
    offset: float,
    guidance_state: Optional[dict] = None,
) -> "any":
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
    if guidance_state is not None:
        guidance_label = get_guidance_label(float(offset), w, guidance_state)
        _draw_guidance_box(overlay, guidance_label)
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
    guidance_state = _new_guidance_state()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return saved_paths, run_id

    frame_index = 0
    preview_index = 0
    while len(saved_paths) < max_frames:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % sample_interval == 0:
            sample_offset = preview_offsets[min(preview_index, len(preview_offsets) - 1)]
            preview_frame = _draw_overlay(
                frame,
                frame_index,
                alignment,
                sample_offset,
                guidance_state=guidance_state,
            )
            out_path = run_output_dir / f"frame_{preview_index + 1:03d}.jpg"
            cv2.imwrite(str(out_path), preview_frame)
            saved_paths.append(f"/outputs/{run_id}/frame_{preview_index + 1:03d}.jpg")
            preview_index += 1

        frame_index += 1

    cap.release()
    return saved_paths, run_id


def create_run_output_dir(run_id: Optional[str] = None) -> tuple[str, Path]:
    """Create and return a run output directory under /outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
    run_output_dir = OUTPUT_DIR / final_run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    return final_run_id, run_output_dir


def clamp_bbox(bbox: Optional[list[int]], frame_width: int, frame_height: int) -> Optional[list[int]]:
    if bbox is None or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(frame_width - 1, x1))
    x2 = max(0, min(frame_width - 1, x2))
    y1 = max(0, min(frame_height - 1, y1))
    y2 = max(0, min(frame_height - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def compute_bbox_signed_offset_px(bbox: Optional[list[int]], frame_width: int) -> Optional[float]:
    if bbox is None:
        return None
    runway_center_x = (bbox[0] + bbox[2]) * 0.5
    image_center_x = frame_width * 0.5
    return float(runway_center_x - image_center_x)


def get_mvp_guidance_label(signed_offset_px: float, frame_width: int) -> str:
    """Simple, forgiving guidance from bbox midpoint offset."""
    threshold = max(MVP_GUIDANCE_ENTER_PX, MVP_GUIDANCE_WIDTH_RATIO * float(frame_width))
    if signed_offset_px > threshold:
        return GUIDANCE_CORRECT_RIGHT
    if signed_offset_px < -threshold:
        return GUIDANCE_CORRECT_LEFT
    return GUIDANCE_ALIGNED


def _draw_bbox_label(frame: "any", bbox: list[int], confidence: float) -> None:
    x1, y1, x2, _ = bbox
    label = f"runway {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    text_thickness = 1
    pad_x = 8
    pad_y = 4

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    label_h = text_h + baseline + (pad_y * 2)
    label_w = text_w + (pad_x * 2)

    lx1 = max(0, x1)
    lx2 = min(frame.shape[1] - 1, lx1 + label_w)
    ly2 = y1 - 4
    ly1 = ly2 - label_h

    # Keep placement stable just above box; if near top, place just inside the box.
    if ly1 < 0:
        ly1 = min(frame.shape[0] - 1, y1 + 4)
        ly2 = min(frame.shape[0] - 1, ly1 + label_h)

    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), MVP_BOX_COLOR, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (lx1 + pad_x, ly2 - baseline - pad_y),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_runway_overlay_frame(
    frame: "any",
    bbox: Optional[list[int]],
    confidence: float,
    runway_center_x: Optional[float],
    image_center_x: float,
    guidance_label: str,
    draw_center_lines: bool = True,
) -> "any":
    """
    Shared renderer for preview frames and annotated video frames.
    Draws bbox + runway label and compact bottom-right guidance box.
    """
    if frame is None:
        return frame

    if len(frame.shape) == 2:
        rendered = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        rendered = frame.copy()

    h, w = rendered.shape[:2]

    if bbox is not None:
        cv2.rectangle(
            rendered,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            MVP_BOX_COLOR,
            2,
            cv2.LINE_AA,
        )
        _draw_bbox_label(rendered, bbox, confidence)

    if draw_center_lines:
        ref_x = max(0, min(w - 1, int(round(image_center_x))))
        cv2.line(rendered, (ref_x, 0), (ref_x, h - 1), MVP_REFERENCE_LINE_COLOR, 1, cv2.LINE_AA)
        if runway_center_x is not None:
            runway_x = max(0, min(w - 1, int(round(runway_center_x))))
            cv2.line(rendered, (runway_x, 0), (runway_x, h - 1), MVP_RUNWAY_LINE_COLOR, 1, cv2.LINE_AA)

    _draw_guidance_box(rendered, guidance_label)
    return rendered


def open_browser_mp4_writer(
    output_path: Path,
    fps: float,
    frame_width: int,
    frame_height: int,
) -> tuple[Optional[cv2.VideoWriter], Optional[str], float]:
    """
    Create an MP4 writer using browser-friendly codecs when available.
    Returns writer, codec label, and fps actually used.
    """
    safe_fps = float(fps)
    if not math.isfinite(safe_fps) or safe_fps <= 0:
        safe_fps = 24.0

    size = (int(frame_width), int(frame_height))
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            safe_fps,
            size,
        )
        if writer.isOpened():
            return writer, codec, safe_fps
        writer.release()

    LOGGER.error("Unable to open VideoWriter for %s", output_path)
    return None, None, safe_fps
