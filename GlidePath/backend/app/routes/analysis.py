import os
import uuid
import statistics
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas.analysis import AnalysisResponse, Wind
from ..services.runway_detector import detect_runway_series
from ..services.scoring import compute_alignment_scores
from ..utils.video import (
    extract_overlay_previews,
    get_video_metadata,
    is_valid_video,
    inspect_video_file,
    save_upload,
    OUTPUT_DIR,
)

router = APIRouter()
LOGGER = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
GEOMETRY_OFFSET_THRESHOLD = 0.58


def _run_pipeline(video_path: str, frame_count: int, frame_width: int) -> tuple[dict, str | None, list[str]]:
    """Run runway inference on the whole video and prepare scoring + overlays."""
    run_id = uuid.uuid4().hex[:12]
    run_output_dir = OUTPUT_DIR / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    annotated_video_path = run_output_dir / "runway_overlay.mp4"

    detections = detect_runway_series(video_path, annotated_output_path=annotated_video_path)

    image_center_x = frame_width // 2 if frame_width > 0 else 0
    offsets: list[float] = []
    for detection in detections:
        geometry = detection.get("geometry") if isinstance(detection, dict) else None
        signed_offset = None
        if (
            isinstance(geometry, dict)
            and float(geometry.get("geometry_confidence", 0.0) or 0.0) >= GEOMETRY_OFFSET_THRESHOLD
        ):
            signed_offset = geometry.get("signed_offset_px")
        if signed_offset is None and isinstance(detection, dict) and detection.get("center_x") is not None:
            signed_offset = float(detection["center_x"] - image_center_x)
        if signed_offset is not None:
            offsets.append(float(signed_offset))

    confidences = [
        float(d.get("confidence", 0.0) or 0.0)
        for d in detections
        if isinstance(d, dict)
    ]

    signed_offset_px = statistics.mean(offsets) if offsets else 0.0
    overall_confidence = max(confidences) if confidences else 0.0

    geometry = {
        "signed_offset_px": signed_offset_px,
        "offset_per_frame": offsets,
    }

    scores = compute_alignment_scores(
        geometry,
        frame_count,
        overall_confidence,
    )

    try:
        preview_frames, _ = extract_overlay_previews(
            video_path=video_path,
            alignment=scores["alignment"],
            detections=detections,
            offsets=offsets,
            sample_interval=20,
            max_frames=10,
            output_dir=run_output_dir,
        )
    except Exception:
        LOGGER.exception("Preview frame extraction failed for video '%s'", video_path)
        preview_frames = []

    overlay_info = inspect_video_file(annotated_video_path)
    LOGGER.info(
        (
            "Overlay artifact path=%s exists=%s size_bytes=%s readable=%s "
            "frames=%s fps=%.2f width=%s height=%s"
        ),
        annotated_video_path,
        overlay_info["exists"],
        overlay_info["size_bytes"],
        overlay_info["readable"],
        overlay_info["frame_count"],
        float(overlay_info["fps"] or 0.0),
        overlay_info["width"],
        overlay_info["height"],
    )
    overlay_video = (
        f"/outputs/{run_id}/runway_overlay.mp4"
        if overlay_info["exists"] and overlay_info["size_bytes"] > 0 and overlay_info["readable"]
        else None
    )
    if overlay_video is None:
        LOGGER.warning("Overlay video disabled for run_id=%s due to invalid output artifact", run_id)

    return scores, overlay_video, preview_frames


@router.get("/analyze-sample", response_model=AnalysisResponse)
def analyze_sample():
    """Return a hardcoded mock analysis response for frontend development."""
    return AnalysisResponse(
        alignment="aligned",
        stability="stable",
        confidence=0.92,
        frame_count=150,
        average_offset_px=2.5,
        offsets=[1.2, 2.1, 3.0, 2.4, 2.0, 1.8],
        wind=Wind(
            direction_degrees=45,
            speed_kt=8.5,
            crosswind_kt=6.0,
            headwind_kt=6.0,
        ),
        preview_frames=[],
    )


@router.post("/analyze-video", response_model=AnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
    """Save the upload, generate demo overlays, and return mock runway analysis."""
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if not is_valid_video(filename, file.content_type):
        raise HTTPException(
            status_code=415,
            detail={
                "error": "Unsupported file type.",
                "allowed_extensions": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
            },
        )

    try:
        saved_path = await save_upload(file)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    file_size = os.path.getsize(saved_path)
    if file_size == 0:
        os.remove(saved_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if file_size > MAX_FILE_SIZE_BYTES:
        os.remove(saved_path)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB.",
        )

    metadata = get_video_metadata(saved_path)
    frame_count = metadata["frame_count"] if metadata["frame_count"] > 0 else 243
    frame_width = metadata.get("width", 0) or 0

    try:
        scores, overlay_video, preview_frames = _run_pipeline(saved_path, frame_count, frame_width)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Analysis pipeline failed.")

    return AnalysisResponse(
        alignment=scores["alignment"],
        stability=scores["stability"],
        confidence=scores["confidence"],
        frame_count=scores["frame_count"],
        average_offset_px=scores["average_offset_px"],
        offsets=scores["offsets"],
        wind=None,
        overlay_video=overlay_video,
        preview_frames=preview_frames,
    )
