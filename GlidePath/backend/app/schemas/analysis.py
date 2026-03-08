from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Wind(BaseModel):
    direction_degrees: Optional[int] = None
    speed_kt: Optional[float] = None
    crosswind_kt: Optional[float] = None
    headwind_kt: Optional[float] = None


class AnalysisResponse(BaseModel):
    alignment: str
    stability: str
    confidence: float
    frame_count: int
    average_offset_px: float
    offsets: list[float]
    wind: Optional[Wind] = None
    preview_frames: list[str]
    geometry_sources: list[str] = Field(default_factory=list)
    frame_debug_artifacts: list[Dict[str, Any]] = Field(default_factory=list)
    output_video: Optional[str] = None
    output_dir: Optional[str] = None
