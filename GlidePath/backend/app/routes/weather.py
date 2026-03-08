from fastapi import APIRouter

from ..schemas.weather import WeatherResponse


router = APIRouter()


@router.get("/weather/{airport_code}", response_model=WeatherResponse)
def get_weather(airport_code: str):
    """Get weather data for airport"""
    return WeatherResponse(
        airport_code=airport_code,
        metar_raw="KJFK 071851Z 09014KT 10SM FEW250 M04/M17 A3034 RMK",
        direction_degrees=90,
        speed_kt=14.0,
        crosswind_kt=10.0,
        headwind_kt=10.0
    )
