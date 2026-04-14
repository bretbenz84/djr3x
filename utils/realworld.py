"""
utils/realworld.py — Real-world awareness helpers for DJ-R3X.

Provides time, date, holiday, location, and weather data without requiring
API keys for most operations.  All functions return plain Python types and
fail gracefully — callers receive None (or sensible defaults) when offline.

External dependencies:
  - holidays (pip install holidays) — US holiday calendar
  - requests — already in requirements.txt

APIs used (all free, no key required):
  - ip-api.com/json/ — IP geolocation
  - api.open-meteo.com — weather forecast
"""

from __future__ import annotations

import logging
import time as _time_module
from datetime import date, datetime
from typing import Optional

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Location cache — refreshed at most once per hour
# ---------------------------------------------------------------------------

_location_cache: Optional[dict] = None
_location_fetched_at: float = 0.0
_LOCATION_TTL = 3600.0   # seconds


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def get_current_time() -> str:
    """Return the current local time as a human-readable string, e.g. '3:42 PM'."""
    return datetime.now().strftime("%-I:%M %p")


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    """Return the ordinal suffix for an integer, e.g. 11 → '11th'."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]}"


def get_current_date() -> dict:
    """Return a dict describing today's date.

    Keys:
        weekday   — e.g. 'Saturday'
        month     — e.g. 'April'
        day       — int, e.g. 11
        year      — int, e.g. 2026
        ordinal   — e.g. '11th'
        formatted — e.g. 'Saturday April 11, 2026'
    """
    now = datetime.now()
    weekday  = now.strftime("%A")
    month    = now.strftime("%B")
    day      = now.day
    year     = now.year
    ord_day  = _ordinal(day)
    formatted = f"{weekday} {month} {day}, {year}"
    return {
        "weekday":   weekday,
        "month":     month,
        "day":       day,
        "year":      year,
        "ordinal":   ord_day,
        "formatted": formatted,
    }


# ---------------------------------------------------------------------------
# Holidays
# ---------------------------------------------------------------------------

def get_holiday(check_date: Optional[date] = None) -> Optional[str]:
    """Return the holiday name if *check_date* (default: today) is a special day.

    Covers:
      - Standard US federal holidays via the `holidays` package
      - Star Wars Day — May the 4th Be With You (May 4)
      - Halloween (October 31)
      - Valentine's Day (February 14)

    Returns None if today is not a holiday.
    """
    try:
        import holidays as _holidays_lib
    except ImportError:
        log.warning("realworld: 'holidays' package not installed — holiday detection disabled")
        return None

    if check_date is None:
        check_date = date.today()

    # Star Wars Day — check before US holidays so it takes priority on May 4.
    if check_date.month == 5 and check_date.day == 4:
        return "Star Wars Day — May the 4th Be With You"

    if check_date.month == 10 and check_date.day == 31:
        return "Halloween"

    if check_date.month == 2 and check_date.day == 14:
        return "Valentine's Day"

    us = _holidays_lib.US(years=check_date.year)
    return us.get(check_date)   # returns str or None


# ---------------------------------------------------------------------------
# Location (ip-api.com — no API key, 45 req/min free tier)
# ---------------------------------------------------------------------------

def get_location() -> Optional[dict]:
    """Return the approximate location based on the device's public IP.

    Returns a dict with keys: city, region, country, lat, lon.
    Result is cached for 1 hour.  Returns None if the request fails.
    """
    global _location_cache, _location_fetched_at

    now = _time_module.monotonic()
    if _location_cache is not None and (now - _location_fetched_at) < _LOCATION_TTL:
        log.debug("realworld: location cache hit")
        return _location_cache

    try:
        resp = requests.get("http://ip-api.com/json/", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            log.warning("realworld: ip-api returned status=%r", data.get("status"))
            return None
        result = {
            "city":    data.get("city", "Unknown City"),
            "region":  data.get("regionName", "Unknown Region"),
            "country": data.get("country", "Unknown Country"),
            "lat":     data.get("lat"),
            "lon":     data.get("lon"),
        }
        _location_cache = result
        _location_fetched_at = now
        log.info("realworld: location fetched — %s, %s", result["city"], result["region"])
        return result
    except Exception:
        log.warning("realworld: location fetch failed (offline?)")
        return None


# ---------------------------------------------------------------------------
# Weather (Open-Meteo — no API key required)
# ---------------------------------------------------------------------------

_WMO_CODES: dict[int, str] = {
    0:  "clear",
    1:  "mostly clear",
    2:  "partly cloudy",
    3:  "overcast",
    45: "foggy",
    48: "foggy",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def get_weather(lat: float, lon: float) -> Optional[dict]:
    """Fetch current weather from Open-Meteo for the given coordinates.

    Returns a dict with keys: temp_f, description, wind_mph.
    Returns None if the request fails.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true&temperature_unit=fahrenheit"
        f"&wind_speed_unit=mph"
    )
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        cw = resp.json().get("current_weather", {})
        wmo = int(cw.get("weathercode", 0))
        description = _WMO_CODES.get(wmo, "conditions unknown")
        result = {
            "temp_f":      round(float(cw.get("temperature", 0))),
            "description": description,
            "wind_mph":    round(float(cw.get("windspeed", 0))),
        }
        log.info(
            "realworld: weather fetched — %d°F, %s, %d mph wind",
            result["temp_f"], result["description"], result["wind_mph"],
        )
        return result
    except Exception:
        log.warning("realworld: weather fetch failed (offline?)")
        return None
