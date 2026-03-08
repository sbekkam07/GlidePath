# import os

# from fastapi import APIRouter, File, HTTPException, UploadFile

# from ..schemas.analysis import AnalysisResponse, Wind
# from ..services.runway_detector import detect_runway
# from ..services.runway_geometry import estimate_runway_geometry
# from ..services.scoring import compute_alignment_scores
# from ..utils.video import (
#     extract_overlay_previews,
#     get_video_metadata,
#     is_valid_video,
#     save_upload,
# )


# router = APIRouter()

# MAX_FILE_SIZE_MB = 500
# MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# def _run_pipeline(video_path: str, frame_count: int) -> dict:
#     """Run the current mock runway analysis pipeline."""
#     detections = detect_runway(video_path)
#     geometry = estimate_runway_geometry(detections)
#     return compute_alignment_scores(
#         geometry,
#         frame_count,
#         detections["confidence"],
#     )


# @router.get("/analyze-sample", response_model=AnalysisResponse)
# def analyze_sample():
#     """Return a hardcoded mock analysis response for frontend development."""
#     return AnalysisResponse(
#         alignment="centered",
#         stability="stable",
#         confidence=0.92,
#         frame_count=150,
#         average_offset_px=2.5,
#         offsets=[1.2, 2.1, 3.0, 2.4, 2.0, 1.8],
#         wind=Wind(
#             direction_degrees=45,
#             speed_kt=8.5,
#             crosswind_kt=6.0,
#             headwind_kt=6.0,
#         ),
#         preview_frames=[],
#     )


# @router.post("/analyze-video", response_model=AnalysisResponse)
# async def analyze_video(file: UploadFile = File(...)):
#     """Save the upload, generate demo overlays, and return mock runway analysis."""
#     filename = file.filename or ""
#     if not filename:
#         raise HTTPException(status_code=400, detail="No filename provided.")

#     if not is_valid_video(filename, file.content_type):
#         raise HTTPException(
#             status_code=415,
#             detail={
#                 "error": "Unsupported file type.",
#                 "allowed_extensions": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
#             },
#         )

#     try:
#         saved_path = await save_upload(file)
#     except Exception:
#         raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

#     file_size = os.path.getsize(saved_path)
#     if file_size == 0:
#         os.remove(saved_path)
#         raise HTTPException(status_code=400, detail="Uploaded file is empty.")

#     if file_size > MAX_FILE_SIZE_BYTES:
#         os.remove(saved_path)
#         raise HTTPException(
#             status_code=413,
#             detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB.",
#         )

#     metadata = get_video_metadata(saved_path)
#     frame_count = metadata["frame_count"] if metadata["frame_count"] > 0 else 243

#     try:
#         scores = _run_pipeline(saved_path, frame_count)
#     except Exception:
#         raise HTTPException(status_code=500, detail="Analysis pipeline failed.")

#     try:
#         preview_frames, _ = extract_overlay_previews(
#             video_path=saved_path,
#             alignment=scores["alignment"],
#             offsets=scores["offsets"],
#             sample_interval=20,
#             max_frames=10,
#         )
#     except Exception:
#         preview_frames = []

#     return AnalysisResponse(
#         alignment=scores["alignment"],
#         stability=scores["stability"],
#         confidence=scores["confidence"],
#         frame_count=scores["frame_count"],
#         average_offset_px=scores["average_offset_px"],
#         offsets=scores["offsets"],
#         wind=None,
#         preview_frames=preview_frames,
#     )


import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas.analysis import AnalysisResponse, Wind
from ..services.runway_detector import detect_runway
from ..services.runway_geometry import estimate_runway_geometry
from ..services.scoring import score_alignment
from ..utils.video import (
    extract_overlay_previews,
    get_video_metadata,
    is_valid_video,
    save_upload,
)

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# -------------------------------------------------------------
# FRAME ANALYSIS (REAL PIPELINE)
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

    annotated = geometry["annotated_image"]

    cv2.putText(
        annotated,
        f"ALIGNMENT: {score['alignment'].replace('_', ' ').upper()}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        annotated,
        f"STABILITY: {score['stability'].upper()}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    out_name = f"analysis_{uuid.uuid4().hex[:8]}.png"
    out_path = OUTPUT_DIR / out_name

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
        "annotated_image": str(out_path),
    }


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
        wind=Wind(
            direction_degrees=45,
            speed_kt=8.5,
            crosswind_kt=6.0,
            headwind_kt=6.0,
        ),
        preview_frames=[],
    )


# -------------------------------------------------------------
# VIDEO ANALYSIS (TEMPORARY MOCK VERSION)
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

    metadata = get_video_metadata(saved_path)
    frame_count = metadata["frame_count"] if metadata["frame_count"] > 0 else 243

    # For now this still returns mock results
    alignment = "centered"
    stability = "stable"
    confidence = 0.9
    offsets = [1.0, 2.0, 3.0, 2.5]

    try:
        preview_frames, _ = extract_overlay_previews(
            video_path=saved_path,
            alignment=alignment,
            offsets=offsets,
            sample_interval=20,
            max_frames=10,
        )
    except Exception:
        preview_frames = []

    return AnalysisResponse(
        alignment=alignment,
        stability=stability,
        confidence=confidence,
        frame_count=frame_count,
        average_offset_px=sum(offsets) / len(offsets),
        offsets=offsets,
        wind=None,
        preview_frames=preview_frames,
    )