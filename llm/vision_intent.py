"""
llm/vision_intent.py — Lightweight heuristic to decide whether a transcribed
utterance is a visually-oriented query that warrants capturing a camera frame.

vision_intent(text) returns True if the text looks like it is asking Rex to
observe, describe, or react to something visible in the scene.  No ML model —
just phrase and keyword matching, which is fast, predictable, and sufficient
for this use-case.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Exact phrase triggers (matched as substrings, case-insensitive)
# ---------------------------------------------------------------------------

_PHRASES: tuple[str, ...] = (
    "what do you see",
    "what am i wearing",
    "what color",
    "what colour",
    "look at",
    "can you see",
    "take a picture",
    "take a photo",
    "what's in front",
    "whats in front",
    "who am i",
    "how many",
    "what is this",
    "what are these",
    "do you see",
    "look around",
    "what do i look like",
    "what's around",
    "whats around",
    "what's behind",
    "whats behind",
    "describe what",
    "describe the",
    "tell me what you see",
)

# ---------------------------------------------------------------------------
# Single-word triggers — only fire when the word appears in a question
# (i.e. text contains "?", or starts with a question word like what/who/can).
# This avoids matching "I love the color blue" as a vision query.
# ---------------------------------------------------------------------------

_QUESTION_WORDS: frozenset[str] = frozenset(
    ("what", "who", "how", "can", "could", "do", "does", "is", "are")
)

_KEYWORDS: tuple[str, ...] = (
    "see",
    "look",
    "color",
    "colour",
    "wearing",
    "holding",
    "carrying",
    "behind you",
    "in front of you",
    "next to you",
)


def vision_intent(text: str) -> bool:
    """Return True if *text* appears to be a visually-oriented query.

    Checks exact-phrase matches first (fast exit), then falls back to
    keyword scanning gated by a question-word heuristic.
    """
    if not text:
        return False

    lower = text.lower().strip()

    # 1. Exact phrase match — highest confidence, no gating needed.
    for phrase in _PHRASES:
        if phrase in lower:
            return True

    # 2. Keyword match — only if the utterance looks like a question.
    first_word = re.split(r"\W+", lower)[0] if lower else ""
    is_question = lower.endswith("?") or first_word in _QUESTION_WORDS

    if is_question:
        for keyword in _KEYWORDS:
            if keyword in lower:
                return True

    return False
