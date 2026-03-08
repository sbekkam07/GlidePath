import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..schemas.weather import WeatherResponse
from ..services.metar import get_metar


router = APIRouter()

# ICAO codes are 4 uppercase alphanumeric characters
ICAO_PATTERN = re.compile(r'^[A-Z0-9]{3,4}$')


def validate_icao(code: str) -> str:
    """Normalize and validate an ICAO airport code."""
    code = code.upper().strip()
    if not ICAO_PATTERN.match(code):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid airport code '{code}'. Expected a 3-4 character ICAO code (e.g. KDEN, KJFK)."
        )
    return code


@router.get("/weather/{airport_code}", response_model=WeatherResponse)
async def get_weather(
    airport_code: str,
    runway_heading: Optional[float] = Query(
        default=None,
        ge=0,
        lt=360,
        description="Runway magnetic heading in degrees (e.g. 80 for runway 08). "
                    "If provided, crosswind and headwind components are computed."
    ),
):
    """Fetch live METAR weather for an ICAO airport code (e.g. KJFK, KLAX).

    Optionally provide runway_heading to compute crosswind and headwind components.
    """
    code = validate_icao(airport_code)

    try:
        data = await get_metar(code, runway_heading=runway_heading)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return WeatherResponse(**data)
