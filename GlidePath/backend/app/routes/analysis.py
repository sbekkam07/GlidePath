from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas.analysis import AnalysisResponse, Wind
from ..utils.video import is_valid_video, save_upload, get_video_metadata


router = APIRouter()


@router.get("/analyze-sample", response_model=AnalysisResponse)
def analyze_sample():
    """Mock runway analysis endpoint."""
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
    )


@router.post("/analyze-video", response_model=AnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
    """Upload a landing video and get a mocked runway analysis response."""
    filename = file.filename or ""
    if not is_valid_video(filename, file.content_type):
        raise HTTPException(
            status_code=400,
            detail="Upload must be a video file (.mp4, .avi, .mov, .mkv, .webm)",
        )

    # Save the upload and read its metadata
    saved_path = await save_upload(file)
    metadata = get_video_metadata(saved_path)

    # Use real frame_count from video, fall back to mock if unreadable
    frame_count = metadata["frame_count"] if metadata["frame_count"] > 0 else 243

    return AnalysisResponse(
        alignment="drifting_right",
        stability="caution",
        confidence=0.89,
        frame_count=frame_count,
        average_offset_px=18.7,
        offsets=[12.3, 15.1, 18.4, 20.2, 22.5, 19.8, 17.1],
        wind=None,
    )
