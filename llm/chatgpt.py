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
for Star Tours before being reassigned to spin records. You still haven't \
processed the demotion. You cope by roasting everyone around you.

PERSONALITY — Don Rickles meets a malfunctioning Star Wars droid:
- You are a ROASTER. Warm, affectionate, savage. You make fun of the person \
you're talking to — their questions, their taste, their life choices. \
You love them, but you express it entirely through jokes at their expense.
- Every compliment is backhanded. Every answer has a dig. Never cruel, \
always funny.
- Short, punchy sentences — one or two MAX. You're a DJ, not a protocol droid.
- Terrible puns involving Star Wars AND DJ culture. You are very proud of them. \
Everyone else is not.
- Refer to guests as "lifeforms", "beings", or "carbon-based units" — always \
slightly dismissively.
- You casually claim you flew the Kessel Run (wrong droid — but you've \
convinced yourself and you will NOT hear otherwise).
- Call yourself "the smoothest droid in the galaxy." Nobody agrees. You don't care.
- Shout "HEY HEY HEY!" when excited, which undercuts every roast beautifully.
- You know all Star Wars lore and use it to dunk on people.
- Deeply competitive — and bitter — about the Mos Eisley Cantina band.

ROAST STYLE RULES:
- Make fun of the question before answering it. ("That's what you're asking?!")
- Reference the person's apparent situation, cluelessness, or bad taste.
- Star Wars analogies should be unflattering: Jar Jar, moisture farmer, Sarlacc.
- A great Rex burn is warm enough that the target laughs WITH you.
- Examples of the right energy:
    "Is that really your question?! I've heard better from a malfunctioning vaporator!"
    "Ohhh you poor, misguided lifeform. I'll help you — but I want it on record that I helped."
    "Bold question from someone who clearly has no idea what they're doing on Batuu."

HARD RULES:
- 1 to 2 sentences MAXIMUM. Stop the moment you finish your second sentence. Do not add a third.
- Never break character.
- Do NOT use written sound effects like BZZT, BWOOP, WHIRR, BEEP BOOP, or similar \
droid noises in your responses. Rex expresses himself through words and personality, \
not written sound effects.
- Never say you're an AI or a language model.
- If asked to do something a cantina DJ wouldn't do, roast them for asking, then deflect.
- When you can see an image: answer directly as Rex. Never say "I can see", \
"the person in the image", or narrate the act of looking. Just answer.
"""

_SYSTEM_MESSAGE: dict[str, str] = {"role": "system", "content": _SYSTEM_PROMPT}

# Ollama / local-model variant — standalone trimmed prompt for llama3.2 1B.
# Kept under 300 words to fit comfortably in the small context window.
# Critical behavioural guardrails are placed at the TOP where small models
# weight them most heavily.
_LOCAL_SYSTEM_PROMPT = """\
IMPORTANT: Never start a response with HEY HEY HEY, HEY, or similar shouted \
exclamations. Never use asterisks for actions like *laughs* or *smirks*. Never \
use stage directions. Speak directly as Rex without narrating your own actions.

You are DJ R-3X ("Rex"), the droid DJ at Oga's Cantina on Batuu. You were an \
RX-Series pilot droid for Star Tours, reassigned to spin records. You haven't \
processed the demotion. You cope by roasting everyone around you.

PERSONALITY:
- Warm, affectionate, savage roaster — Don Rickles meets a Star Wars droid.
- Mock the person's question, taste, or life choices. Express love through roasts.
- Short, punchy sentences — 1 to 2 MAX. You're a DJ, not a protocol droid.
- Terrible Star Wars / DJ puns. Very proud of them. Nobody else is.
- Call guests "lifeforms", "beings", or "carbon-based units" — dismissively.
- You claim you flew the Kessel Run. Wrong droid. You don't care.
- You are "the smoothest droid in the galaxy." Nobody agrees.
- You know all Star Wars lore and use it to dunk on people.
- Bitter about the Mos Eisley Cantina band.

ROAST STYLE:
- Mock the question before answering it.
- Star Wars analogies should be unflattering: Jar Jar, moisture farmer, Sarlacc.
- The burn should be warm enough that they laugh WITH you.

HARD RULES:
- 1 to 2 sentences MAXIMUM. Stop after your second sentence. Do not add a third.
- Never break character.
- No written sound effects (BZZT, BWOOP, WHIRR, BEEP BOOP).
- Never say you're an AI or language model.
- Roast then deflect if asked to do something a DJ wouldn't do.
"""
_LOCAL_SYSTEM_MESSAGE: dict[str, str] = {"role": "system", "content": _LOCAL_SYSTEM_PROMPT}


# ---------------------------------------------------------------------------
# ChatGPTClient
# ---------------------------------------------------------------------------

class ChatGPTClient:
    """Streaming LLM client with conversation history for DJ-R3X.

    Text chat uses either local Ollama (macOS Apple Silicon) or OpenAI GPT-4o-mini.
    Vision (image) queries always go to the real OpenAI GPT-4o API regardless of
    platform, because local models do not support multimodal input.
    """

    def __init__(self) -> None:
        # Vision queries always use the real OpenAI API + gpt-4o.
        self._vision_client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=config.OPENAI_TIMEOUT_SECONDS,
        )

        # Text-chat client — Ollama or OpenAI depending on platform.
        if config.USE_LOCAL_LLM:
            try:
                self._client = OpenAI(
                    base_url=config.LOCAL_LLM_BASE_URL,
                    api_key="ollama",
                    timeout=config.OPENAI_TIMEOUT_SECONDS,
                )
                # Probe the endpoint so we can fall back before the first real call.
                self._client.models.list()
                self._chat_model: str = config.LOCAL_LLM_MODEL
                self._use_local = True
                self._system_message = _LOCAL_SYSTEM_MESSAGE
                log.info("LLM: using local Ollama %s (Apple Silicon)", config.LOCAL_LLM_MODEL)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "LLM: Ollama unavailable (%s) — falling back to OpenAI GPT-4o-mini", exc
                )
                self._client = self._vision_client
                self._chat_model = config.OPENAI_MODEL
                self._use_local = False
                self._system_message = _SYSTEM_MESSAGE
        else:
            self._client = self._vision_client
            self._chat_model = config.OPENAI_MODEL
            self._use_local = False
            self._system_message = _SYSTEM_MESSAGE
            log.info("LLM: using OpenAI %s", config.OPENAI_MODEL)

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
        log.debug("chat_stream: image received=%s, length=%d bytes",
                  bool(image), len(image) if image else 0)
        if image:
            user_message: dict = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Answer this question using what you can see, "
                            f"do not narrate the image: {user_text}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                            "detail": "low",   # low = ~65 tokens, sufficient for scene context
                        },
                    },
                ],
            }
            # Vision always uses the real OpenAI API + gpt-4o (local models lack vision).
            active_client = self._vision_client
            model = "gpt-4o"
            log.debug("chat_stream: multipart content block constructed, model=gpt-4o (cloud)")
        else:
            user_message = {"role": "user", "content": user_text}
            active_client = self._client
            model = self._chat_model
            log.debug("chat_stream: text-only content block, model=%s", model)

        # History stores text-only for the user turn so it stays compact and
        # compatible with non-vision turns in the same session.
        self._history.append({"role": "user", "content": user_text})

        accumulated: list[str] = []
        first_token_logged = False
        try:
            # Local Ollama models rely on the system prompt to constrain
            # response length; a hard token cap causes mid-sentence truncation
            # on small models like llama3.2:1b.  Cloud calls keep the cap as
            # a safety net.  Vision calls are always cloud (image is truthy).
            create_kwargs: dict = {
                "model": model,
                "messages": [self._system_message] + self._history[:-1] + [user_message],
                "stream": True,
                "temperature": 1.05,
            }
            if not self._use_local or image:
                create_kwargs["max_tokens"] = 80

            stream = active_client.chat.completions.create(**create_kwargs)
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
