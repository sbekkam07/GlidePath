# METAR weather service
# Fetches real METAR data from the NOAA Aviation Weather Center (AWC) API.
# No API key required. Free public endpoint.
#
# API: https://aviationweather.gov/api/data/metar?ids={ICAO}&format=raw

import math
import re
import httpx

AWC_METAR_URL = "https://aviationweather.gov/api/data/metar"

# Regex to match METAR wind token, e.g. 09014KT, 25008G15KT, VRB05KT
WIND_PATTERN = re.compile(
    r'(?P<dir>\d{3}|VRB)(?P<speed>\d{2,3})(?:G\d{2,3})?KT'
)


def parse_wind(metar_raw: str) -> dict:
    """Extract wind direction and speed from a raw METAR string."""
    match = WIND_PATTERN.search(metar_raw)
    if not match:
        return {"direction_degrees": None, "speed_kt": None}

    direction_str = match.group("dir")
    speed_str = match.group("speed")

    return {
        # VRB means variable wind direction — represent as None
        "direction_degrees": None if direction_str == "VRB" else int(direction_str),
        "speed_kt": float(speed_str),
    }


def compute_wind_components(
    wind_direction: int, wind_speed: float, runway_heading: float
) -> dict:
    """Compute headwind and crosswind components for a given runway heading.

    Uses standard aviation trigonometry:
      - angle = wind_direction - runway_heading
      - headwind = wind_speed * cos(angle)  (positive = headwind, negative = tailwind)
      - crosswind = wind_speed * sin(angle) (absolute value = crosswind magnitude)

    Args:
        wind_direction: Wind direction in degrees (from METAR)
        wind_speed: Wind speed in knots (from METAR)
        runway_heading: Magnetic heading of the runway in degrees

    Returns:
        Dict with headwind_kt and crosswind_kt, both rounded to 1 decimal place.
    """
    angle_rad = math.radians(wind_direction - runway_heading)
    headwind = round(wind_speed * math.cos(angle_rad), 1)
    crosswind = round(abs(wind_speed * math.sin(angle_rad)), 1)
    return {"headwind_kt": headwind, "crosswind_kt": crosswind}


async def get_metar(airport_code: str, runway_heading: float | None = None) -> dict:
    """Fetch live METAR data for an ICAO airport code.

    Args:
        airport_code: 4-character ICAO code (e.g. KDEN)
        runway_heading: Optional runway magnetic heading in degrees.
                        If provided, crosswind_kt and headwind_kt are computed.

    Returns a dict ready to unpack into WeatherResponse.
    Raises ValueError if no data is found or the request fails.
    """
    code = airport_code.upper().strip()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                AWC_METAR_URL, params={"ids": code, "format": "raw"}
            )
            response.raise_for_status()
    except httpx.TimeoutException:
        raise ValueError(f"Request timed out fetching METAR for {code}")
    except httpx.HTTPStatusError as e:
        raise ValueError(f"HTTP error {e.response.status_code} fetching METAR for {code}")
    except httpx.RequestError as e:
        raise ValueError(f"Network error fetching METAR: {e}")

    # AWC returns the raw METAR string as plain text, one per line
    metar_raw = response.text.strip()
    if not metar_raw:
        raise ValueError(f"No METAR data found for airport: {code}")

    # Take only the first line in case multiple are returned
    metar_raw = metar_raw.splitlines()[0].strip()

    wind = parse_wind(metar_raw)

    # Compute wind components only if runway heading and valid wind data are available
    crosswind_kt = None
    headwind_kt = None
    if (
        runway_heading is not None
        and wind["direction_degrees"] is not None
        and wind["speed_kt"] is not None
    ):
        components = compute_wind_components(
            wind["direction_degrees"], wind["speed_kt"], runway_heading
        )
        crosswind_kt = components["crosswind_kt"]
        headwind_kt = components["headwind_kt"]

    return {
        "airport_code": code,
        "metar_raw": metar_raw,
        "direction_degrees": wind["direction_degrees"],
        "speed_kt": wind["speed_kt"],
        "crosswind_kt": crosswind_kt,
        "headwind_kt": headwind_kt,
    }
