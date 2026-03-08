from typing import Optional

from pydantic import BaseModel


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
