import os
import statistics
import logging

import cv2

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas.analysis import AnalysisResponse, Wind
from ..services.runway_detector import detect_runway
from ..services.scoring import compute_alignment_scores
from ..utils.video import (
    clamp_bbox,
    create_run_output_dir,
    draw_runway_overlay_frame,
    get_video_metadata,
    get_mvp_guidance_label,
    is_valid_video,
    open_browser_mp4_writer,
    save_upload,
)


router = APIRouter()
LOGGER = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def _run_pipeline(video_path: str) -> dict:
    """Run bbox-based analysis and generate previews + annotated video."""
    sample_interval = 20
    max_preview_frames = 10
    max_bbox_reuse_frames = 2

    run_id, run_output_dir = create_run_output_dir()
    overlay_filename = "runway_overlay.mp4"
    overlay_path = run_output_dir / overlay_filename

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video for analysis.")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    codec = None
    fps_used = fps
    writer_attempted = False
    writer_failed = False
    if frame_width > 0 and frame_height > 0:
        writer_attempted = True
        writer, codec, fps_used = open_browser_mp4_writer(
            output_path=overlay_path,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    LOGGER.info(
        "Annotated video init path=%s size=%sx%s fps=%.2f codec=%s writer_opened=%s",
        overlay_path,
        frame_width,
        frame_height,
        fps_used,
        codec or "none",
        bool(writer),
    )

    preview_frames: list[str] = []
    offsets: list[float] = []
    confidences: list[float] = []
    frame_count = 0

    last_bbox = None
    last_confidence = 0.0
    last_offset = 0.0
    missed_detection_frames = 0
    last_guidance_label = "ALIGNED"

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            if frame is None or frame.size == 0:
                frame_count += 1
                continue

            frame_count += 1
            h, w = frame.shape[:2]
            image_center_x = w * 0.5
            if writer is None and not writer_attempted:
                writer_attempted = True
                writer, codec, fps_used = open_browser_mp4_writer(
                    output_path=overlay_path,
                    fps=fps,
                    frame_width=w,
                    frame_height=h,
                )

            detection = detect_runway(frame)
            confidence = float(detection.get("confidence", 0.0) or 0.0)
            bbox = clamp_bbox(detection.get("bbox"), w, h)

            runway_center_x = None
            signed_offset_px = None

            if bbox is not None:
                runway_center_x = (bbox[0] + bbox[2]) * 0.5
                signed_offset_px = float(runway_center_x - image_center_x)
                offsets.append(signed_offset_px)
                confidences.append(confidence)

                last_bbox = list(bbox)
                last_confidence = confidence
                last_offset = signed_offset_px
                missed_detection_frames = 0
            elif last_bbox is not None and missed_detection_frames < max_bbox_reuse_frames:
                bbox = list(last_bbox)
                runway_center_x = (bbox[0] + bbox[2]) * 0.5
                signed_offset_px = float(last_offset)
                confidence = max(0.0, last_confidence * 0.95)
                offsets.append(signed_offset_px)
                missed_detection_frames += 1
            else:
                missed_detection_frames += 1

            if signed_offset_px is not None:
                last_guidance_label = get_mvp_guidance_label(signed_offset_px, w)

            rendered = draw_runway_overlay_frame(
                frame=frame,
                bbox=bbox,
                confidence=confidence,
                runway_center_x=runway_center_x,
                image_center_x=image_center_x,
                guidance_label=last_guidance_label,
                draw_center_lines=True,
            )

            if writer is not None and not writer_failed:
                try:
                    writer.write(rendered)
                except Exception:
                    LOGGER.exception("Failed to write annotated frame for %s", overlay_path)
                    writer_failed = True
                    writer.release()
                    writer = None

            if (frame_count - 1) % sample_interval == 0 and len(preview_frames) < max_preview_frames:
                preview_index = len(preview_frames) + 1
                preview_path = run_output_dir / f"frame_{preview_index:03d}.jpg"
                cv2.imwrite(str(preview_path), rendered)
                preview_frames.append(f"/outputs/{run_id}/frame_{preview_index:03d}.jpg")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    overlay_exists = overlay_path.exists()
    overlay_size = overlay_path.stat().st_size if overlay_exists else 0
    overlay_video = None
    if overlay_exists and overlay_size > 0:
        overlay_video = f"/outputs/{run_id}/{overlay_filename}"

    LOGGER.info(
        "Annotated video final path=%s exists=%s size=%s bytes",
        overlay_path,
        overlay_exists,
        overlay_size,
    )

    signed_offset_px = statistics.mean(offsets) if offsets else 0.0
    return {
        "frame_count": frame_count,
        "overall_confidence": max(confidences) if confidences else 0.0,
        "preview_frames": preview_frames,
        "overlay_video": overlay_video,
        "signed_offset_px": signed_offset_px,
        "offsets": offsets,
    }


@router.get("/analyze-sample", response_model=AnalysisResponse)
def analyze_sample():
    """Return a hardcoded mock analysis response for frontend development."""
    return AnalysisResponse(
        alignment="centered",
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
        overlay_video=None,
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

    try:
        pipeline = _run_pipeline(saved_path)
    except Exception:
        LOGGER.exception("Analysis pipeline failed for file '%s'", filename)
        raise HTTPException(status_code=500, detail="Analysis pipeline failed.")

    frame_count = metadata["frame_count"] if metadata["frame_count"] > 0 else pipeline["frame_count"]
    scores = compute_alignment_scores(
        {
            "signed_offset_px": pipeline["signed_offset_px"],
            "offset_per_frame": pipeline["offsets"],
        },
        frame_count,
        pipeline["overall_confidence"],
    )

    return AnalysisResponse(
        alignment=scores["alignment"],
        stability=scores["stability"],
        confidence=scores["confidence"],
        frame_count=scores["frame_count"],
        average_offset_px=scores["average_offset_px"],
        offsets=scores["offsets"],
        wind=None,
        overlay_video=pipeline["overlay_video"],
        preview_frames=pipeline["preview_frames"],
    )
