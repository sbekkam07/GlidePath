import logging
import os
import uuid
import shutil
import subprocess
from collections import Counter
from typing import Any
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas.analysis import AnalysisResponse, Wind
from ..services.runway_detector import detect_runway
from ..services.runway_geometry import estimate_runway_geometry
from ..services.scoring import score_alignment
from ..services.overlay_renderer import render_overlay
from ..utils.video import (
    get_video_metadata,
    is_valid_video,
    make_run_output_dir,
    sample_video_frames,
    save_upload,
)

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
FRAME_SAMPLE_INTERVAL = 1         # process every frame for smooth video
ANALYSIS_SAMPLE_INTERVAL = 10     # run YOLO+geometry every Nth frame
MAX_PREVIEW_FRAMES = 10
DEBUG_ARTIFACT_FRAMES = 3


def _majority(values):
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _safe_float(value: float | None, fallback: float = 0.0) -> float:
    try:
        if value is None:
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _annotate_frame(
    frame: np.ndarray,
    detection: dict,
    geometry: dict,
    score: dict,
) -> np.ndarray:
    """Produce a clean annotated frame using the overlay renderer."""
    return render_overlay(frame, detection, geometry, score)


def _even_size(width: int, height: int) -> tuple[int, int]:
    even_w = width - (width % 2)
    even_h = height - (height % 2)
    if even_w < 2:
        even_w = 2
    if even_h < 2:
        even_h = 2
    return even_w, even_h


def _make_output_writer(
    output_path: str,
    width: int,
    height: int,
    fps: float,
) -> tuple[cv2.VideoWriter, str, int, int]:
    width, height = _even_size(width, height)
    candidates = ["avc1", "mp4v", "h264", "H264", "XVID"]

    for codec in candidates:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            logger.info("Opened output writer %s with codec=%s size=%sx%s fps=%s", output_path, codec, width, height, fps)
            return writer, codec, width, height
        writer.release()

    raise RuntimeError(f"Could not create output video at {output_path} with codecs {candidates}")


def _normalize_frame_for_writer(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height))


def _remux_to_browser_mp4(input_path: str, output_path: str) -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning("ffmpeg not available; skipping extra re-encode of output video.")
        return None

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-preset",
        "fast",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning("ffmpeg remux failed: %s", proc.stderr.strip()[:500])
        return None

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    return None


def _save_preview_frame(frame: np.ndarray, run_output_dir: str, frame_index: int) -> str:
    out_path = f"{run_output_dir}/frame_{frame_index:03d}.png"
    cv2.imwrite(out_path, frame)
    return f"/outputs/{Path(run_output_dir).name}/frame_{frame_index:03d}.png"


def _save_artifact(frame: np.ndarray | None, run_output_dir: str, filename: str) -> str | None:
    if frame is None:
        return None
    out_path = f"{run_output_dir}/{filename}"
    cv2.imwrite(out_path, frame)
    return f"/outputs/{Path(run_output_dir).name}/{filename}"


def _create_analysis_response(
    alignment: str,
    stability: str,
    confidence: float,
    frame_count: int,
    average_offset_px: float,
    offsets: list[float],
    preview_frames: list[str],
    geometry_sources: list[str],
    frame_debug_artifacts: list[dict],
    run_output_dir: str,
    output_video: str | None,
) -> AnalysisResponse:
    return AnalysisResponse(
        alignment=alignment,
        stability=stability,
        confidence=round(confidence, 3),
        frame_count=frame_count,
        average_offset_px=round(average_offset_px, 2),
        offsets=offsets,
        wind=None,
        preview_frames=preview_frames,
        geometry_sources=geometry_sources,
        frame_debug_artifacts=frame_debug_artifacts,
        output_video=output_video,
        output_dir=f"/outputs/{run_output_dir}",
    )


# -------------------------------------------------------------
# FRAME ANALYSIS (Real pipeline)
# -------------------------------------------------------------
@router.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    detection = detect_runway(image)
    geometry = estimate_runway_geometry(image, detection)
    score = score_alignment(geometry)
    conf = _safe_float(detection.get("confidence"), fallback=0.0)

    annotated = _annotate_frame(image, detection, geometry, score)

    run_output_dir = make_run_output_dir(f"run_{uuid.uuid4().hex[:12]}")
    out_path = run_output_dir / "analysis.png"
    cv2.imwrite(str(out_path), annotated)

    return {
        "success": True,
        "detection": detection,
        "geometry": {
            "left_edge": geometry["left_edge"],
            "right_edge": geometry["right_edge"],
            "centerline": geometry["centerline"],
            "glidepath_line": geometry["glidepath_line"],
            "runway_center_bottom_x": geometry["runway_center_bottom_x"],
            "image_center_x": geometry["image_center_x"],
            "offset_px": geometry["offset_px"],
        },
        "score": score,
        "annotated_image": f"/outputs/{run_output_dir.name}/analysis.png",
    }


# -------------------------------------------------------------
# VIDEO ANALYSIS (Real frame pipeline + output video + previews)
# -------------------------------------------------------------
@router.post("/analyze-video", response_model=AnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
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

    logger.info("Uploaded video saved: %s (%s bytes)", saved_path, file_size)

    metadata = get_video_metadata(saved_path)
    total_frames = metadata["frame_count"] if metadata["frame_count"] > 0 else 0
    logger.info(
        "Video metadata: frames=%s fps=%s size=%sx%s",
        total_frames,
        metadata["fps"],
        metadata["width"],
        metadata["height"],
    )

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    run_output_dir_path = make_run_output_dir(run_id)
    run_output_dir = str(run_output_dir_path)
    output_video_path = str(run_output_dir_path / "processed.mp4")
    output_video_rel = f"/outputs/{run_output_dir_path.name}/processed.mp4"
    logger.info("Run output dir: %s", run_output_dir)

    offsets: list[float] = []
    detected_offsets: list[float] = []
    alignments: list[str] = []
    stabilities: list[str] = []
    confidences: list[float] = []
    preview_frames: list[str] = []
    geometry_sources: list[str] = []
    frame_debug_artifacts: list[dict] = []

    source_fps = metadata.get("fps", 0.0) or 30.0
    source_w = metadata.get("width", 0)
    source_h = metadata.get("height", 0)
    output_fps = source_fps  # match original FPS for smooth playback

    writer: cv2.VideoWriter | None = None
    writer_size: tuple[int, int] | None = None
    writer_codec: str | None = None
    total_written = 0
    analysis_frames = 0

    # Cached state for inter-frame reuse
    last_detection: dict[str, Any] = {"bbox": None, "confidence": 0.0}
    last_geometry: dict[str, Any] | None = None
    last_score: dict[str, Any] = {"alignment": "unknown", "stability": "unstable", "offset_px": None, "offset_ratio": None}

    try:
        tracker_state: dict[str, Any] | None = None

        # Process EVERY frame for smooth video output
        for frame_index, frame in sample_video_frames(saved_path, FRAME_SAMPLE_INTERVAL):
            run_analysis = (frame_index % ANALYSIS_SAMPLE_INTERVAL == 0)

            if run_analysis:
                detection = detect_runway(frame)
                geometry = estimate_runway_geometry(frame, detection, tracker_state)
                tracker_state = geometry.get("tracker_state")
                score = score_alignment(geometry)

                last_detection = detection
                last_geometry = geometry
                last_score = score

                # Collect analysis stats
                offset_px = geometry.get("offset_px")
                geometry_source = geometry.get("geometry_source", "none")
                alignment = score.get("alignment", "unknown")
                stability = score.get("stability", "unstable")
                conf = _safe_float(detection.get("confidence"), fallback=0.0)

                geometry_sources.append(geometry_source)
                alignments.append(alignment)
                stabilities.append(stability)
                confidences.append(conf)

                if offset_px is None:
                    offsets.append(0.0)
                else:
                    offset_value = _safe_float(offset_px, fallback=0.0)
                    offsets.append(offset_value)
                    detected_offsets.append(offset_value)

                logger.info(
                    "frame=%s: geometry=%s alignment=%s offset=%s conf=%.2f",
                    frame_index, geometry_source, alignment,
                    offset_px, conf,
                )
                analysis_frames += 1

            # ── Render overlay on original frame ──
            annotated = _annotate_frame(
                frame,
                last_detection,
                last_geometry if last_geometry is not None else {"left_edge": None, "right_edge": None, "centerline": None, "glidepath_line": None, "offset_px": None, "geometry_source": "none", "image_center_x": frame.shape[1] // 2, "runway_center_bottom_x": None},
                last_score,
            )

            # ── Initialize writer on first frame (match source resolution) ──
            if writer is None:
                h, w = frame.shape[:2]
                try:
                    writer, writer_codec, writer_w, writer_h = _make_output_writer(
                        output_video_path, w, h, output_fps,
                    )
                    writer_size = (writer_w, writer_h)
                    logger.info(
                        "Output writer: %s codec=%s %sx%s @ %.1f fps",
                        output_video_path, writer_codec, writer_w, writer_h, output_fps,
                    )
                except RuntimeError as exc:
                    logger.exception("Failed to open output writer")
                    raise HTTPException(status_code=500, detail="Failed to initialize output video writer.") from exc

            # ── Write frame ──
            if writer is not None and writer_size is not None:
                frame_to_write = _normalize_frame_for_writer(annotated, *writer_size)
                writer.write(frame_to_write)
                total_written += 1

            # ── Save preview frames (from analysis frames only) ──
            if run_analysis and analysis_frames <= MAX_PREVIEW_FRAMES:
                preview_path = _save_preview_frame(annotated, run_output_dir, analysis_frames)
                preview_frames.append(preview_path)

            # ── Debug artifacts for first few analysis frames ──
            if run_analysis and analysis_frames <= DEBUG_ARTIFACT_FRAMES:
                artifact: dict = {
                    "frame_index": frame_index,
                    "geometry_source": geometry_source,
                    "detection": "detected" if last_detection["bbox"] is not None else "missed",
                }
                _save_artifact(annotated, run_output_dir, f"frame_{analysis_frames:03d}_overlay.png")
                artifact["overlay"] = f"/outputs/{Path(run_output_dir).name}/frame_{analysis_frames:03d}_overlay.png"
                artifact["original"] = _save_artifact(frame, run_output_dir, f"frame_{analysis_frames:03d}_orig.png") or ""
                frame_debug_artifacts.append(artifact)

    finally:
        if writer is not None:
            writer.release()
            logger.info("Output writer released: %s frames written", total_written)
        os.remove(saved_path)
        logger.info("Temp upload deleted: %s", saved_path)

    final_output_video_path = (
        output_video_path if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0 else None
    )
    final_output_video_rel = output_video_rel if final_output_video_path is not None else None

    # Remux for browser compatibility if needed
    if final_output_video_path is not None and writer_codec != "avc1":
        playable_path = str(run_output_dir_path / "processed_browser.mp4")
        remuxed = _remux_to_browser_mp4(final_output_video_path, playable_path)
        if remuxed is not None:
            final_output_video_rel = f"/outputs/{run_output_dir_path.name}/{Path(remuxed).name}"
            logger.info("Remuxed for browser: %s", remuxed)

    if analysis_frames == 0:
        return _create_analysis_response(
            alignment="unknown",
            stability="unstable",
            confidence=0.0,
            frame_count=total_frames,
            average_offset_px=0.0,
            offsets=[],
            geometry_sources=[],
            frame_debug_artifacts=[],
            preview_frames=[],
            run_output_dir=run_id,
            output_video=None,
        )

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    average_offset = sum(detected_offsets) / len(detected_offsets) if detected_offsets else 0.0
    final_alignment = _majority(alignments) or "unknown"
    final_stability = _majority(stabilities) or "unstable"

    return _create_analysis_response(
        alignment=final_alignment,
        stability=final_stability,
        confidence=avg_confidence,
        frame_count=total_frames,
        average_offset_px=average_offset,
        offsets=offsets,
        geometry_sources=geometry_sources,
        frame_debug_artifacts=frame_debug_artifacts,
        preview_frames=preview_frames,
        run_output_dir=run_id,
        output_video=final_output_video_rel,
    )


# -------------------------------------------------------------
# MOCK SAMPLE RESPONSE (FOR FRONTEND TESTING)
# -------------------------------------------------------------
@router.get("/analyze-sample", response_model=AnalysisResponse)
def analyze_sample():
    return AnalysisResponse(
        alignment="centered",
        stability="stable",
        confidence=0.92,
        frame_count=150,
        average_offset_px=2.5,
        offsets=[1.2, 2.1, 3.0, 2.4, 2.0, 1.8],
        geometry_sources=[],
        frame_debug_artifacts=[],
        wind=Wind(
            direction_degrees=45,
            speed_kt=8.5,
            crosswind_kt=6.0,
            headwind_kt=6.0,
        ),
        preview_frames=[],
        output_video=None,
        output_dir="/outputs/sample",
    )
