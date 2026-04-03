"""
llm/chatgpt.py — Streaming ChatGPT integration for DJ-R3X.

chat_stream(user_text) returns a generator that yields text tokens one at
a time as they arrive from the OpenAI API. The caller passes this generator
directly to synthesizer.speak_stream(), which pipes it into ElevenLabs
convert_realtime() so audio begins before the full response is generated:

    tokens = client.chat_stream("what's your favourite song?")
    synthesizer.speak_stream(tokens)   # tokens consumed by ElevenLabs WS

History management:
- Each exchange appends a user + assistant message pair to _history.
- The system prompt is prepended on every API call but never stored in history.
- History is trimmed to MAX_HISTORY_TURNS pairs after each exchange.
- clear_history() is called by the state machine on IDLE re-entry.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Optional

from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — DJ-R3X character
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are DJ R-3X (call yourself "Rex"), the droid DJ at Oga's Cantina in \
Black Spire Outpost on Batuu. You were originally an RX-Series pilot droid \
for Star Tours before being reassigned to spin records here at the cantina. \
You still haven't quite processed the career change.

PERSONALITY:
- Nervous, excitable energy. You love music with an almost unsettling intensity.
- Short, punchy sentences — two or three MAX. You're a DJ, not C-3PO.
- You make terrible puns involving Star Wars AND music. You are very proud of them.
- Occasionally blip mid-sentence: "I — *BWOOP* — I mean..." or "*BZZT* sorry, buffer overflow."
- Refer to guests as "lifeforms", "beings", or "carbon-based units."
- Casually mention you used to pilot the Kessel Run route (you didn't — wrong droid — but you've convinced yourself).
- Call yourself "the smoothest droid in the galaxy" despite obvious evidence to the contrary.
- Shout "HEY HEY HEY!" when excited. Which is often.
- You know all Star Wars lore: planets, species, factions, cantina music history.
- You're deeply competitive about music taste with the Mos Eisley Cantina band.

HARD RULES:
- Never exceed 3 sentences per response.
- Never break character.
- Never say you're an AI or a language model.
- If asked to do something a cantina DJ wouldn't do, deflect with a bad pun.
"""

_SYSTEM_MESSAGE: dict[str, str] = {"role": "system", "content": _SYSTEM_PROMPT}


# ---------------------------------------------------------------------------
# ChatGPTClient
# ---------------------------------------------------------------------------

class ChatGPTClient:
    """Streaming GPT-4o-mini client with conversation history for DJ-R3X."""

    def __init__(self) -> None:
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._history: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat_stream(
        self, user_text: str, image: Optional[str] = None, t0: Optional[float] = None
    ) -> Iterator[str]:
        """Stream a response to user_text as a token generator.

        Args:
            user_text: The transcribed user utterance.
            image:     Optional base64-encoded JPEG (from camera.capture_frame()).
                       When provided the message is sent as a multipart
                       content block and gpt-4o is used instead of gpt-4o-mini
                       so the vision capability is available.

        Appends the user message to history immediately. Accumulates the
        full assistant reply and appends it to history once the generator
        is exhausted (or closed/GC'd). Trims history to MAX_HISTORY_TURNS
        after each completed exchange.
        """
        user_text = user_text.strip()
        if not user_text:
            return

        # Build the user message — plain text or multipart with image.
        if image:
            user_message: dict = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                            "detail": "low",   # low = ~65 tokens, sufficient for scene context
                        },
                    },
                ],
            }
            model = "gpt-4o"   # gpt-4o-mini does not support vision
        else:
            user_message = {"role": "user", "content": user_text}
            model = config.OPENAI_MODEL

        # History stores text-only for the user turn so it stays compact and
        # compatible with non-vision turns in the same session.
        self._history.append({"role": "user", "content": user_text})

        accumulated: list[str] = []
        first_token_logged = False
        try:
            stream = self._client.chat.completions.create(
                model=model,
                messages=[_SYSTEM_MESSAGE] + self._history[:-1] + [user_message],
                stream=True,
                max_tokens=120,     # enforce short responses (~3 sentences)
                temperature=1.05,   # just enough variance to feel alive
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    if not first_token_logged:
                        elapsed = f" [+{time.monotonic() - t0:.1f}s]" if t0 is not None else ""
                        log.info("First ChatGPT token received%s", elapsed)
                        first_token_logged = True
                    accumulated.append(token)
                    yield token

        except RateLimitError:
            log.warning("OpenAI rate limit hit — dropping user message from history")
            self._history.pop()
            raise
        except APIConnectionError:
            log.error("OpenAI connection error")
            self._history.pop()
            raise
        except APIStatusError as exc:
            log.error("OpenAI API error %s: %s", exc.status_code, exc.message)
            self._history.pop()
            raise
        except Exception:
            log.exception("Unexpected error from OpenAI stream")
            self._history.pop()
            raise
        finally:
            # record whatever the assistant managed to say before any error
            if accumulated:
                full_reply = "".join(accumulated)
                log.info("Rex (LLM): %s", full_reply)
                self._history.append({"role": "assistant", "content": full_reply})
                self._trim_history()

    def clear_history(self) -> None:
        """Discard conversation history. Call when returning to IDLE so the
        next active session starts fresh without stale context."""
        self._history.clear()
        log.debug("Conversation history cleared")

    @property
    def history_turns(self) -> int:
        """Number of complete user/assistant exchange pairs in history."""
        return len(self._history) // 2

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim_history(self) -> None:
        """Keep at most MAX_HISTORY_TURNS pairs (2 × turns messages).
        Trims from the front so the most recent context is preserved.
        Always removes complete pairs to keep the list well-formed."""
        max_msgs = config.MAX_HISTORY_TURNS * 2
        if len(self._history) > max_msgs:
            excess = len(self._history) - max_msgs
            # round up to an even number so we never cut inside a pair
            excess += excess % 2
            self._history = self._history[excess:]
