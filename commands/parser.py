"""
commands/parser.py — Normalize and match transcribed text against PHRASE_INDEX.

Match pipeline (in order, returns on first hit):
  1. Exact match  — normalized input is a key in PHRASE_INDEX.
  2. Fuzzy match  — difflib finds the closest phrase above
                    config.COMMAND_FUZZY_THRESHOLD.
  3. No match     — returns None; caller should escalate to LLM.

Both strategies operate on the normalized form of the input so that
differences in punctuation, case, and repeated spaces never matter.
"""

from __future__ import annotations

import difflib
import logging
import re
import string

import config
from commands.command_list import Command, PHRASE_INDEX

log = logging.getLogger(__name__)

# Pre-compiled translation table: removes every character in string.punctuation.
_STRIP_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Return a canonical form of *text* suitable for phrase comparison.

    Steps:
      1. Lowercase.
      2. Strip punctuation (! ? . , ' " etc.).
      3. Collapse runs of whitespace to a single space.
      4. Strip leading/trailing whitespace.

    >>> normalize("Hey, Rex!  What's up?")
    'hey rex  whats up'
    """
    text = text.lower()
    text = text.translate(_STRIP_PUNCT)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse(text: str) -> Command | None:
    """Attempt to match *text* against the predefined command list.

    Returns the matched Command on success, or None if no phrase is close
    enough (caller should fall back to the LLM).

    Matching is two-stage:
      - Exact: O(1) dict lookup after normalization.
      - Fuzzy: difflib.get_close_matches over all ~119 phrase keys,
               accepting the best match above COMMAND_FUZZY_THRESHOLD.
    """
    if not text or not text.strip():
        return None

    normalized = normalize(text)
    if not normalized:
        return None

    # --- Stage 1: exact match ---
    cmd = PHRASE_INDEX.get(normalized)
    if cmd is not None:
        log.debug("Command exact match: %r → %s", normalized, cmd.phrases[0])
        return cmd

    # --- Stage 2: fuzzy match ---
    # Skip fuzzy matching for very short inputs — a 1-3 character string can
    # accidentally score above the threshold against much longer phrases.
    if len(normalized) < 4:
        log.debug("Input too short for fuzzy match (%d chars): %r", len(normalized), normalized)
        return None

    candidates = difflib.get_close_matches(
        normalized,
        PHRASE_INDEX.keys(),
        n=1,
        cutoff=config.COMMAND_FUZZY_THRESHOLD,
    )
    if candidates:
        best = candidates[0]
        score = difflib.SequenceMatcher(None, normalized, best).ratio()
        cmd = PHRASE_INDEX[best]
        log.debug(
            "Command fuzzy match (%.2f): %r → %r",
            score, normalized, best,
        )
        return cmd

    log.debug("No command match for: %r", normalized)
    return None
