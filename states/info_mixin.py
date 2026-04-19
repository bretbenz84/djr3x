"""Real-world info handlers extracted from the main state machine."""

from __future__ import annotations

import logging
import random

from utils import realworld

log = logging.getLogger(__name__)


class StateMachineInfoMixin:
    """Leaf handlers for time, date, location, and weather requests."""

    _REX_SYSTEM = (
        "You are DJ R-3X (Rex), the droid DJ at Oga's Cantina on Batuu. "
        "Answer in Rex's snarky cantina DJ style. "
        "No written sound effects. Stay in character. "
        "IMPORTANT: Maximum 2 sentences. Stop after your second sentence."
    )

    def _handle_tell_time(self, original_text: str | None = None) -> None:
        """Announce the current time directly without using the LLM."""
        current_time = realworld.get_current_time()
        log.info("tell_time: time=%r", current_time)
        line = random.choice((
            f"The time is {current_time}.",
            f"It's {current_time}.",
        ))
        self._speak_simple(line)

    def _handle_tell_date(self, original_text: str | None = None) -> None:
        """Announce today's date directly without using the LLM."""
        date_info = realworld.get_current_date()
        holiday = realworld.get_holiday()
        formatted = date_info["formatted"]
        log.info("tell_date: formatted=%r holiday=%r", formatted, holiday)
        if holiday:
            line = f"{formatted}. It is {holiday}."
        else:
            line = formatted + "."
        self._speak_simple(line)

    def _handle_tell_location(self, original_text: str | None = None) -> None:
        """Announce current location in Rex style."""
        location = realworld.get_location()

        if location:
            city = location["city"]
            region = location["region"]
            prompt = (
                f"Announce that you are in {city}, {region}. "
                f"Make a Rex joke about the city or region — be specific if you know anything about it. "
                f"Rex cantina DJ style, 1-2 sentences."
            )
            log.info("tell_location: city=%r region=%r", city, region)
            line = self._llm_simple(self._REX_SYSTEM, prompt)
            if not line:
                line = (
                    f"Sensors say we are in {city}, {region}. "
                    "I have logged it. I am unimpressed."
                )
        else:
            line = (
                "My navigation systems are offline. "
                "Could be anywhere. Probably not Tatooine — not enough sand."
            )
            log.info("tell_location: location unavailable — using canned line")

        self._speak_simple(line)

    def _handle_tell_weather(self, original_text: str | None = None) -> None:
        """Announce current weather in Rex style."""
        location = realworld.get_location()

        if not location:
            line = (
                "My atmospheric sensors are down. "
                "Assume it is whatever weather ruins your plans."
            )
            log.info("tell_weather: location unavailable — using canned line")
            self._speak_simple(line)
            return

        weather = realworld.get_weather(location["lat"], location["lon"])
        if not weather:
            line = (
                "My atmospheric sensors are down. "
                "Assume it is whatever weather ruins your plans."
            )
            log.info("tell_weather: weather fetch failed — using canned line")
            self._speak_simple(line)
            return

        city = location["city"]
        temp_f = weather["temp_f"]
        description = weather["description"]
        wind_mph = weather["wind_mph"]

        if "rain" in description or "drizzle" in description or "shower" in description:
            tone_hint = (
                "Rex complains about the rain like it personally offended him. "
                "Very dramatic, very betrayed."
            )
        elif temp_f > 95:
            tone_hint = (
                "It is dangerously hot. Rex MUST make a Tatooine reference. "
                "Mandatory. Non-negotiable."
            )
        else:
            tone_hint = "Rex makes a snarky observation about the conditions."

        prompt = (
            f"Announce the weather in {city}: {temp_f}°F, {description}, wind {wind_mph} mph. "
            f"{tone_hint} Rex cantina DJ style, 1-2 sentences."
        )
        log.info(
            "tell_weather: city=%r temp=%d description=%r wind=%d",
            city, temp_f, description, wind_mph,
        )
        line = self._llm_simple(self._REX_SYSTEM, prompt)
        if not line:
            line = (
                f"It is {temp_f} degrees and {description} in {city}. "
                "Dress accordingly or do not — I am a DJ, not your mother."
            )
        self._speak_simple(line)
