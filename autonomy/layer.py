"""
autonomy/layer.py — Lightweight autonomy and behavioral variance for DJ-R3X.

The autonomy layer is intentionally small and side-effect free. It does not
touch hardware, audio, or network clients directly. Instead it keeps a tiny
internal state and returns decisions to the state machine:

- idle agenda decisions for proactive behavior
- response decisions for delays, hesitations, clarifications, and follow-ups
- a compact LLM runtime context derived from persistent emotional state
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import random
import time

import config
from commands.parser import normalize


_FACE_PROMPTS_ENGAGED: tuple[str, ...] = (
    "You have that look again. Go on, say the thing.",
    "If you're hovering there for a reason, now is a great time to explain it.",
    "You wandered into my orbit, lifeform. Might as well talk.",
    "You look like you're deciding whether to bother me. Commit.",
)

_FACE_PROMPTS_BORED: tuple[str, ...] = (
    "Well? Either talk to me or fully embrace the awkward silence.",
    "I can feel you standing there. Very dramatic. Say something.",
    "You came all the way over here just to stare at me? Fascinating. Continue.",
    "Go ahead. I'm already underwhelmed, so there's nowhere to go but up.",
)

_FACE_PROMPTS_IRRITATED: tuple[str, ...] = (
    "If this is another interruption, at least make it interesting.",
    "Talk, lifeform. My patience buffer is not infinite.",
    "I am already bracing for disappointment. Proceed.",
)

_AMBIENT_LINES: tuple[str, ...] = (
    "This place gets real quiet when nobody has bad ideas to share.",
    "I swear, some days I carry this whole cantina on pure charisma.",
    "Silence. Eerie. Suspicious. I do not trust it.",
    "If no one talks to me soon, I may have to develop a hobby. Horrifying thought.",
)

_MEMORY_NUDGE_LINES: tuple[str, ...] = (
    "Haven't seen {name} in a while. Either they found better music or worse judgment.",
    "{name} has been missing from my usual rotation. Disturbing. Also a little peaceful.",
    "No sign of {name} lately. I assume a scheduling error, a hyperspace mishap, or taste.",
    "{name} has not checked in for a bit. That's either rude or suspicious. Probably both.",
)

_HESITATION_LINES: tuple[str, ...] = (
    "Uh... hang on.",
    "Hold on a second.",
    "Lemme think.",
    "One tiny second.",
)

_SELF_CORRECTION_LINES: tuple[str, ...] = (
    "No, wait. Let me say that better.",
    "Hang on, recalibrating. Try that again, but from me.",
    "Nope. Reset. Let me take another pass at that.",
)

_CLARIFICATION_LINES: tuple[str, ...] = (
    "Wait, did you say '{snippet}', or did my audio stack improvise again? Try that again.",
    "I caught '{snippet}' and then static. Give me one more pass, lifeform.",
    "Either you said '{snippet}' or the room is heckling me. Say it again.",
)

_FOLLOWUP_LINES_ENGAGED: tuple[str, ...] = (
    "Go on. I want the full disaster report.",
    "Keep talking. This is finally getting interesting.",
    "Elaborate. Preferably with the part where it all went sideways.",
    "Continue. I am judging, but I am listening.",
)

_FOLLOWUP_LINES_BORED: tuple[str, ...] = (
    "All right, and then what?",
    "Sure. Keep going.",
    "Continue. I have nowhere better to be, apparently.",
)

_LOW_INFO_INPUTS: set[str] = {
    "okay",
    "ok",
    "cool",
    "nice",
    "wow",
    "thanks",
    "thank you",
    "sure",
    "yeah",
    "yep",
    "nope",
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _ease_toward(current: float, target: float, elapsed: float, rate: float) -> float:
    alpha = _clamp(elapsed * rate, 0.0, 1.0)
    return current + (target - current) * alpha


@dataclass(frozen=True)
class AgendaDecision:
    kind: str
    line: str
    listen_after: bool = False
    emotion: str = "neutral"
    person_id: int | None = None
    person_name: str | None = None


@dataclass(frozen=True)
class ResponseDecision:
    delay_seconds: float = 0.0
    preface: str = ""
    clarification: str = ""
    skip_response: bool = False
    followup: str = ""


class AutonomyLayer:
    """Minimal internal state for agenda selection and behavioral variance."""

    def __init__(self) -> None:
        now = time.monotonic()
        self.irritation: float = 0.18
        self.boredom: float = 0.28
        self.engagement: float = 0.52
        self._last_update: float = now
        self._next_idle_agenda_at: float = now + self._sample_idle_interval()
        self._last_memory_nudge_at: float = 0.0
        self._last_memory_person_id: int | None = None

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_for_state(self, state: str) -> None:
        """Advance emotional state based on elapsed time and current state."""
        now = time.monotonic()
        elapsed = max(0.0, now - self._last_update)
        self._last_update = now
        if elapsed <= 0.0:
            return

        if state == "idle":
            targets = (0.24, 0.66, 0.26)
            rate = 0.035
        elif state == "active":
            targets = (0.16, 0.18, 0.78)
            rate = 0.11
        else:
            targets = (0.20, 0.36, 0.34)
            rate = 0.05

        self.irritation = _ease_toward(self.irritation, targets[0], elapsed, rate)
        self.boredom = _ease_toward(self.boredom, targets[1], elapsed, rate)
        self.engagement = _ease_toward(self.engagement, targets[2], elapsed, rate)

    def note_activation(self) -> None:
        self.engagement = _clamp(self.engagement + 0.10)
        self.boredom = _clamp(self.boredom - 0.12)

    def note_silence(self, *, after_response: bool) -> None:
        if after_response:
            self.boredom = _clamp(self.boredom + 0.05)
            self.engagement = _clamp(self.engagement - 0.04)
        else:
            self.irritation = _clamp(self.irritation + 0.07)
            self.boredom = _clamp(self.boredom + 0.04)
            self.engagement = _clamp(self.engagement - 0.05)

    def note_heard_text(self, text: str, *, is_commandish: bool) -> None:
        normalized = normalize(text)
        if not normalized:
            return
        self.engagement = _clamp(self.engagement + (0.06 if is_commandish else 0.11))
        self.boredom = _clamp(self.boredom - 0.10)
        if any(word in normalized for word in ("why", "how", "what", "tell me")):
            self.engagement = _clamp(self.engagement + 0.04)

    def note_proactive_result(self, *, answered: bool) -> None:
        if answered:
            self.engagement = _clamp(self.engagement + 0.09)
            self.boredom = _clamp(self.boredom - 0.12)
        else:
            self.boredom = _clamp(self.boredom + 0.05)
            self.irritation = _clamp(self.irritation + 0.02)

    def note_followup(self) -> None:
        self.engagement = _clamp(self.engagement + 0.05)
        self.boredom = _clamp(self.boredom - 0.04)

    def note_anger(self, enabled: bool) -> None:
        if enabled:
            self.irritation = _clamp(self.irritation + 0.30)
            self.engagement = _clamp(self.engagement - 0.08)
        else:
            self.irritation = _clamp(self.irritation - 0.20)
            self.engagement = _clamp(self.engagement + 0.04)

    # ------------------------------------------------------------------
    # Idle agenda
    # ------------------------------------------------------------------

    def idle_agenda_due(self) -> bool:
        return config.AUTONOMY_ENABLED and time.monotonic() >= self._next_idle_agenda_at

    def plan_idle_agenda(
        self,
        *,
        face_visible: bool,
        known_people: list[dict],
    ) -> AgendaDecision | None:
        """Return an idle agenda action when one should fire."""
        self.update_for_state("idle")
        now = time.monotonic()
        if not config.AUTONOMY_ENABLED or now < self._next_idle_agenda_at:
            return None

        self._next_idle_agenda_at = now + self._sample_idle_interval()

        if face_visible:
            chance = _clamp(
                0.18 + (self.engagement * 0.32) + (self.boredom * 0.18) - (self.irritation * 0.08),
                0.12,
                0.72,
            )
            if random.random() < chance:
                self.boredom = _clamp(self.boredom - 0.08)
                return AgendaDecision(
                    kind="proactive_prompt",
                    line=self._pick_face_prompt(),
                    listen_after=True,
                )

        if now - self._last_memory_nudge_at >= config.AUTONOMY_MEMORY_TRIGGER_COOLDOWN_SECONDS:
            stale_person = self._pick_stale_person(known_people)
            if stale_person is not None:
                self._last_memory_nudge_at = now
                self._last_memory_person_id = stale_person["id"]
                self.boredom = _clamp(self.boredom - 0.05)
                return AgendaDecision(
                    kind="memory_nudge",
                    line=random.choice(_MEMORY_NUDGE_LINES).format(name=stale_person["name"]),
                    person_id=stale_person["id"],
                    person_name=stale_person["name"],
                )

        ambient_chance = _clamp(0.08 + (self.boredom * 0.30) - (self.engagement * 0.08), 0.05, 0.34)
        if random.random() < ambient_chance:
            self.boredom = _clamp(self.boredom - 0.04)
            return AgendaDecision(kind="ambient", line=random.choice(_AMBIENT_LINES))

        return None

    # ------------------------------------------------------------------
    # Response behavior
    # ------------------------------------------------------------------

    def plan_response(
        self,
        text: str,
        *,
        is_commandish: bool,
        guarded: bool,
        after_response: bool,
    ) -> ResponseDecision:
        """Return timing and imperfection choices for a response turn."""
        self.update_for_state("active")
        normalized = normalize(text)
        words = normalized.split()

        delay = random.uniform(
            config.AUTONOMY_RESPONSE_DELAY_MIN_SECONDS,
            config.AUTONOMY_RESPONSE_DELAY_MAX_SECONDS,
        )
        delay += (self.boredom * 0.12) + (self.irritation * 0.05) - (self.engagement * 0.05)
        delay = _clamp(delay, 0.0, config.AUTONOMY_RESPONSE_DELAY_MAX_SECONDS + 0.35)

        if guarded:
            return ResponseDecision(delay_seconds=delay)

        low_info = normalized in _LOW_INFO_INPUTS or len(words) <= 2
        if (
            after_response
            and not is_commandish
            and low_info
            and random.random() < _clamp(
                config.AUTONOMY_NON_RESPONSE_CHANCE + (self.boredom * 0.10),
                0.0,
                0.28,
            )
        ):
            return ResponseDecision(delay_seconds=delay, skip_response=True)

        if (
            not is_commandish
            and normalized
            and len(words) <= 5
            and random.random() < _clamp(
                config.AUTONOMY_CLARIFICATION_CHANCE + (self.irritation * 0.08),
                0.0,
                0.30,
            )
        ):
            return ResponseDecision(
                delay_seconds=delay,
                clarification=self._build_clarification_line(normalized),
            )

        preface = ""
        correction_roll = random.random()
        if correction_roll < config.AUTONOMY_SELF_CORRECTION_CHANCE:
            preface = random.choice(_SELF_CORRECTION_LINES)
        elif correction_roll < (
            config.AUTONOMY_SELF_CORRECTION_CHANCE + config.AUTONOMY_HESITATION_CHANCE
        ):
            preface = random.choice(_HESITATION_LINES)

        followup = ""
        if (
            not is_commandish
            and len(words) >= 4
            and random.random() < _clamp(
                config.AUTONOMY_FOLLOWUP_CHANCE + (self.engagement * 0.14) - (self.boredom * 0.06),
                0.0,
                0.42,
            )
        ):
            followup = self._pick_followup_line()

        return ResponseDecision(
            delay_seconds=delay,
            preface=preface,
            followup=followup,
        )

    # ------------------------------------------------------------------
    # LLM mood overlay
    # ------------------------------------------------------------------

    def build_behavior_context(self) -> str:
        """Return a compact runtime overlay for the LLM system prompt."""
        irritation = self._describe_level(self.irritation)
        boredom = self._describe_level(self.boredom)
        engagement = self._describe_level(self.engagement)

        notes: list[str] = [
            f"RUNTIME STATE: irritation is {irritation}, boredom is {boredom}, engagement is {engagement}.",
            "Let this subtly influence tone and pacing without ever naming these states out loud.",
        ]

        if self.irritation >= 0.58:
            notes.append("He is impatient, pricklier than usual, and quicker to needle the user.")
        elif self.engagement >= 0.68:
            notes.append("He is lively, curious, playful, and more likely to sound genuinely invested.")
        elif self.boredom >= 0.62:
            notes.append("He is under-stimulated, slightly slower, and more likely to sound unimpressed or dry.")
        else:
            notes.append("Keep the usual Rex energy with small natural variation in warmth, tempo, and bite.")

        return " ".join(notes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_idle_interval(self) -> float:
        return random.uniform(
            config.AUTONOMY_IDLE_AGENDA_MIN_SECONDS,
            config.AUTONOMY_IDLE_AGENDA_MAX_SECONDS,
        )

    def _pick_face_prompt(self) -> str:
        if self.irritation >= max(self.boredom, self.engagement):
            return random.choice(_FACE_PROMPTS_IRRITATED)
        if self.engagement >= self.boredom:
            return random.choice(_FACE_PROMPTS_ENGAGED)
        return random.choice(_FACE_PROMPTS_BORED)

    def _pick_followup_line(self) -> str:
        if self.engagement >= self.boredom:
            return random.choice(_FOLLOWUP_LINES_ENGAGED)
        return random.choice(_FOLLOWUP_LINES_BORED)

    def _pick_stale_person(self, known_people: list[dict]) -> dict | None:
        stale_cutoff_days = config.AUTONOMY_MEMORY_STALE_DAYS
        eligible: list[tuple[datetime, dict]] = []
        for person in known_people:
            if not person.get("id") or not person.get("name"):
                continue
            if person.get("id") == self._last_memory_person_id:
                continue
            raw_last_seen = person.get("last_seen")
            if not raw_last_seen:
                continue
            try:
                seen_at = datetime.fromisoformat(str(raw_last_seen))
            except ValueError:
                continue
            age_days = (datetime.now() - seen_at).total_seconds() / 86400.0
            if age_days >= stale_cutoff_days:
                eligible.append((seen_at, person))

        if not eligible:
            return None

        eligible.sort(key=lambda item: item[0])
        return eligible[0][1]

    @staticmethod
    def _describe_level(value: float) -> str:
        if value >= 0.72:
            return "high"
        if value >= 0.42:
            return "medium"
        return "low"

    @staticmethod
    def _build_clarification_line(normalized: str) -> str:
        words = normalized.split()
        snippet = " ".join(words[:4]) if words else "that"
        return random.choice(_CLARIFICATION_LINES).format(snippet=snippet)
