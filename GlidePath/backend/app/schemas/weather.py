from typing import Optional

from pydantic import BaseModel


class WeatherResponse(BaseModel):
    airport_code: str
    metar_raw: Optional[str] = None
    direction_degrees: Optional[int] = None
    speed_kt: Optional[float] = None
    crosswind_kt: Optional[float] = None
    headwind_kt: Optional[float] = None
