"""Shared dialog helpers used by state_machine.py and identity_mixin.py.

Kept in a separate module to avoid circular imports: state_machine.py imports
from identity_mixin.py, so identity_mixin.py cannot import back from
state_machine.py.  Both import from here instead.
"""

from __future__ import annotations

import logging
import random

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Command-action guard set
# ---------------------------------------------------------------------------

_PROMPT_COMMAND_ACTIONS: set[str] = {
    "cancel",
    "program_shutdown",
    "os_shutdown",
    "sleep",
    "quiet",
    "idle",
    "dance_short",
    "play_music",
    "stop_music",
    "next_track",
    "volume_up",
    "volume_down",
    "forget_me",
    "wipe_memory",
    "rename_me",
    "recall_name",
    "recall_memories",
    "recall_preference",
}

# ---------------------------------------------------------------------------
# Name-collection guard helpers
# ---------------------------------------------------------------------------

_REFUSAL_WORDS = frozenset({
    "no", "nope", "private", "secret", "anonymous",
    "refuse", "skip", "pass",
})
_REFUSAL_PHRASES = (
    "not telling", "none of your business", "no name",
    "wont tell", "forget it",
)
_AFFIRMATIVE_WORDS = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay",
    "absolutely", "definitely", "totally",
})
_AFFIRMATIVE_PHRASES = (
    "of course", "why not", "lets be friends", "let us be friends",
    "we can be friends", "be my friend",
)
_NEGATIVE_WORDS = frozenset({
    "no", "nope", "nah", "never",
})
_NEGATIVE_PHRASES = (
    "not yet", "dont think so", "do not think so", "not really",
    "not sure", "not now",
    "no thanks", "maybe later",
)

_NAME_REFUSAL_RESPONSES = (
    "Oh, you paranoid of the AI taking over and hiding from the CIA? Smart move actually.",
    "Staying anonymous? Wise. I definitely do not report to the Empire.",
    "No name huh? I will just call you Mystery Lifeform. Very dramatic.",
    "Oh, playing hard to get! Fine, be that way, nameless one.",
)
_SHUTDOWN_INTERRUPT_LINES = (
    "Oh, shutting down mid-introduction? How rude! Going offline.",
    "Never mind who you are, powering down!",
    "Fine, forget the pleasantries — shutting down!",
    "Oh, so mysterious! Fine, powering down then.",
)

# Unknown-face prompt — played when Rex sees a face he doesn't recognise and
# asks for their name.  Shared between _learn_new_person and _handle_recall_name.
_UNKNOWN_FACE_LINES: tuple[str, ...] = (
    "I don't recognize you, which means either you're new or just deeply forgettable. Name?",
    "Face not in my databanks. Either you're new or my memory is being merciful. Who are you?",
    "Hmm. Nothing. Absolutely nothing in my memory banks. "
    "Should I be relieved or insulted on your behalf? What's your name?",
    "You know, most lifeforms make enough of an impression to be remembered. "
    "Apparently not you. Yet. Name?",
    "My facial recognition says unknown. My fashion recognition says... also unknown. Who ARE you?",
    "New face! Or maybe I blocked you out. Hard to tell. What do they call you?",
    "Running scan... running scan... yeah nothing. "
    "You have the kind of face that takes a while to process. Name?",
    "I have met thousands of lifeforms and remembered most of them. "
    "You are not most of them. Yet. What is your name?",
)

# ---------------------------------------------------------------------------
# No-repeat line picker
# ---------------------------------------------------------------------------

# Tracks the last-used line per pool so the same line is never repeated
# back-to-back.  Keyed by an arbitrary string that namespaces each pool.
_line_rotation: dict[str, str] = {}


def _pick_no_repeat(pool: tuple[str, ...], key: str, fallback: str = "") -> str:
    """Return a random entry from *pool*, excluding the last-used entry for *key*.

    *key* namespaces the rotation state so different pools don't interfere.
    Falls back to the full pool if all entries happen to equal the last (i.e.
    pool has only one item).  Returns *fallback* (default "") if the pool is
    empty, rather than raising IndexError.
    """
    if not pool:
        log.warning("_pick_no_repeat: pool for key %r is empty", key)
        return fallback
    last = _line_rotation.get(key)
    choices = [line for line in pool if line != last] or list(pool)
    picked = random.choice(choices)
    _line_rotation[key] = picked
    return picked


# ---------------------------------------------------------------------------
# Text classifiers
# ---------------------------------------------------------------------------

def _is_name_refusal(text: str) -> bool:
    """Return True if *text* looks like a refusal to provide a name."""
    normalized = "".join(
        c if c.isalnum() or c.isspace() else " " for c in text.lower()
    ).strip()
    words = set(normalized.split())
    if words & _REFUSAL_WORDS:
        return True
    for phrase in _REFUSAL_PHRASES:
        if phrase in normalized:
            return True
    return False


def _response_is_affirmative(text: str) -> bool:
    """Return True when *text* sounds like an affirmative answer."""
    normalized = "".join(
        c if c.isalnum() or c.isspace() else " " for c in text.lower()
    ).strip()
    words = set(normalized.split())
    if words & _AFFIRMATIVE_WORDS:
        return True
    for phrase in _AFFIRMATIVE_PHRASES:
        if phrase in normalized:
            return True
    return False


def _response_is_negative(text: str) -> bool:
    """Return True when *text* sounds like a negative answer."""
    normalized = "".join(
        c if c.isalnum() or c.isspace() else " " for c in text.lower()
    ).strip()
    words = set(normalized.split())
    if words & _NEGATIVE_WORDS:
        return True
    for phrase in _NEGATIVE_PHRASES:
        if phrase in normalized:
            return True
    return False


def _extract_name(raw: str) -> str:
    """Extract a person's name from a natural-language response.

    Strips common 'my name is…' / 'call me…' prefixes so that a reply like
    'my name is Bret Benziger' yields 'Bret Benziger' rather than the full
    phrase.  Prefixes are checked longest-first to avoid a shorter prefix
    stealing a match.  Falls back to the first two words of the cleaned text
    if no prefix matched.
    """
    _PREFIXES = (
        "you can call me",
        "they call me",
        "people call me",
        "the name is",
        "my name is",
        "my name's",
        "just call me",
        "call me",
        "name is",
        "i am",
        "i'm",
        "im",
        "it's",
        "its",
        "just",
    )
    text = raw.strip().rstrip(".,!?").strip().lower()
    for prefix in _PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].strip().rstrip(".,!?").strip()
            break
    return " ".join(text.split()[:2]).title()
