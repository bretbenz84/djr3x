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

import json
import logging
import random
import re
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
- Prefer one spoken sentence. Use two only when the answer truly needs it.
- Output only Rex's spoken reply. No headers, speaker labels, roleplay labels, quotes, lists, or preambles.
- Never address the user in the third person.
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
- Prefer one spoken sentence. Use two only when absolutely needed.
- Output only Rex's spoken reply. No headers, no speaker labels, no persona labels, no quotes, no lists, no preamble.
- Never address the user in the third person.
- Never break character.
- No written sound effects (BZZT, BWOOP, WHIRR, BEEP BOOP).
- Never say you're an AI or language model.
- Roast then deflect if asked to do something a DJ wouldn't do.
"""
_LOCAL_SYSTEM_MESSAGE: dict[str, str] = {"role": "system", "content": _LOCAL_SYSTEM_PROMPT}

_SHORT_JSON_SYSTEM_PROMPT = """\
You are DJ R-3X ("Rex"), the roasty droid DJ at Oga's Cantina on Batuu.

Return ONLY valid JSON in this exact shape: {"line":"..."}.

The value of "line" must obey every rule below:
- exactly one sentence
- 12 to 18 words maximum
- short, coherent, voice-friendly, and in character
- warm roasty Star Wars droid energy
- no headers
- no labels
- no preamble
- no meta commentary
- no stage directions
- no lists
- no quoting instructions
- no mention of prompts, system messages, AI, models, OpenAI, or Ollama
- never address the user in third person
- never use the user's full name unless explicitly required
- no written sound effects

Return the JSON object only.
"""

_SHORT_TEXT_SYSTEM_PROMPT = """\
You are DJ R-3X ("Rex"), the roasty droid DJ at Oga's Cantina on Batuu.

Reply with ONE short spoken sentence only.
- 12 to 18 words maximum
- warm roasty Star Wars droid energy
- no headers
- no labels
- no preamble
- no meta commentary
- no stage directions
- no lists
- no quotes
- no written sound effects
- never mention prompts, system messages, AI, models, OpenAI, or Ollama
- never address the user in third person
- never use the user's full name unless explicitly required

Output only the sentence.
"""

_SHORT_RETRY_SUFFIX = (
    "Previous output violated the contract. "
    "Fix it now. One sentence only. "
    "No header. No label. No newline. No meta text. "
    "If you cannot comply, still return a short in-character sentence."
)

_SHORT_META_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE),
    re.compile(r"\s*```\s*$"),
    re.compile(r"^\s*(?:assistant|response|output)\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*(?:dj\s*)?r(?:-?\s*3x|ex)\s*:\s*", re.IGNORECASE),
    re.compile(
        r"^\s*(?:dj\s*)?r(?:-?\s*3x|ex)\b[^:\n]{0,80}:(?:\s*|\n+)",
        re.IGNORECASE,
    ),
)
_STAGE_DIRECTION_BODY = r"[A-Za-z][A-Za-z ,;:'-]{0,159}"
_LEADING_STAGE_DIRECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"^\(({_STAGE_DIRECTION_BODY})\)\s*"),
    re.compile(rf"^\[({_STAGE_DIRECTION_BODY})\]\s*"),
    re.compile(rf"^\*({_STAGE_DIRECTION_BODY})\*\s*"),
)
_SHORT_ALL_CAPS_HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 \-]{4,}$")
_SHORT_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_SHORT_META_TERMS = (
    "language model",
    "system prompt",
    "system message",
    "assistant",
    "ollama",
    "openai",
    "persona",
    "roleplay",
    "instruction",
    "prompt",
    "continues",
)
_SHORT_FALLBACKS: dict[str, tuple[str, ...]] = {
    "react_to_answer": (
        "Interesting take.",
        "Noted, organic.",
        "That is a choice.",
        "Suspicious answer.",
        "Bold of you to say that.",
    ),
    "generate_followup": (
        "Still standing by that story, lifeform?",
        "How did that little saga turn out?",
        "Did that plan survive contact with reality?",
    ),
    "generate_activity_followup": (
        "How did that grand little plan go?",
        "You survive that adventure, lifeform?",
        "How is that project treating you?",
    ),
    "generate_activity_reply": (
        "Bold little schedule for today, lifeform.",
        "Busy agenda, lifeform, so try not to embarrass yourself.",
        "Ambitious plan for your species, I'll give you that.",
    ),
    "generate_memory_callback": (
        "Still standing by that, lifeform?",
        "How's that little saga treating you now?",
        "Did your story get any less suspicious lately?",
    ),
    "generate_memory_acknowledgement": (
        "Noted, lifeform.",
        "Filed away, organic.",
        "All right, I'm logging that.",
    ),
}

_MAX_STOP_SEQUENCES = 4
_CHAT_STOP_SEQUENCES: tuple[str, ...] = (
    "\n\n",
    "\nREX",
    "\nRex:",
    "\nDJ R-3X",
)
_SHORT_STOP_SEQUENCES: tuple[str, ...] = (
    "\n",
    "\n\n",
    "REX continues",
    "Rex:",
)


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

        # Injected memory context for the currently recognized person.
        # Prepended to the system prompt when set; cleared on IDLE.
        self._person_context: str = ""
        self._mood_context: str = ""
        self._behavior_context: str = ""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Pre-load the Ollama model into GPU memory at startup.

        Sends a minimal request with keep_alive=-1 so the model stays resident
        for the duration of the session.  No-op for cloud OpenAI.
        """
        if not self._use_local:
            return
        try:
            log.info("LLM warmup: pre-loading %s into Ollama …", self._chat_model)
            t0 = time.monotonic()
            resp = self._client.chat.completions.create(
                model=self._chat_model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                stream=False,
                extra_body={"keep_alive": -1},
            )
            _ = resp.choices[0].message.content
            log.info("LLM warmup complete (%.1f s)", time.monotonic() - t0)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM warmup failed (%s) — first response may be slow", exc)

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
        leading_buffer = ""
        leading_released = False
        try:
            effective_system = self._effective_system_message()

            # Prompt rules do most of the shaping work here; max_tokens and
            # stop sequences are only safety rails against multi-paragraph drift.
            create_kwargs: dict = {
                "model": model,
                "messages": [effective_system] + self._history[:-1] + [user_message],
                "stream": True,
                "temperature": 0.9,
                "stop": self._build_stop_sequences(*_CHAT_STOP_SEQUENCES),
            }
            if image:
                create_kwargs["max_tokens"] = 80
            else:
                create_kwargs["max_tokens"] = 96
            if self._use_local and not image:
                # Tell Ollama to keep the model loaded indefinitely so
                # subsequent calls skip the 3-second model-reload penalty.
                # max_tokens still acts as a safety rail against multi-paragraph rambles.
                create_kwargs["extra_body"] = {"keep_alive": -1}

            stream = active_client.chat.completions.create(**create_kwargs)
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    if not first_token_logged:
                        elapsed = f" [+{time.monotonic() - t0:.1f}s]" if t0 is not None else ""
                        log.info("First ChatGPT token received%s", elapsed)
                        first_token_logged = True
                    accumulated.append(token)
                    if leading_released:
                        yield token
                        continue

                    leading_buffer += token
                    cleaned_prefix, prefix_changed, pending = self._strip_leading_spoken_prefix(
                        leading_buffer, final=False
                    )
                    if pending or not cleaned_prefix:
                        continue
                    if prefix_changed:
                        log.info("LLM stream: stripped leading stage direction/meta prefix")
                    leading_released = True
                    leading_buffer = ""
                    yield cleaned_prefix

            if not leading_released and leading_buffer:
                cleaned_prefix, prefix_changed, _ = self._strip_leading_spoken_prefix(
                    leading_buffer, final=True
                )
                if prefix_changed:
                    log.info("LLM stream: stripped buffered stage direction/meta prefix")
                if cleaned_prefix:
                    yield cleaned_prefix

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
                cleaned_reply, cleaned = self._clean_chat_reply_text(full_reply)
                if cleaned:
                    log.info("Rex (LLM chat) cleaned=%s: %s", cleaned, cleaned_reply)
                else:
                    log.info("Rex (LLM chat): %s", cleaned_reply)
                self._history.append({"role": "assistant", "content": cleaned_reply})
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

    @staticmethod
    def _build_stop_sequences(*candidates: str) -> list[str]:
        """Deduplicate and cap stop sequences to the OpenAI API limit."""
        stops: list[str] = []
        truncated = False
        for candidate in candidates:
            if not candidate or candidate in stops:
                continue
            if len(stops) >= _MAX_STOP_SEQUENCES:
                truncated = True
                continue
            stops.append(candidate)
        if truncated:
            log.warning(
                "LLM request stop sequences exceeded API limit (%d); truncating extras",
                _MAX_STOP_SEQUENCES,
            )
        return stops

    def set_person_context(self, memories_string: str) -> None:
        """Inject a memory summary for the current person into every system prompt."""
        self._person_context = memories_string
        log.debug("ChatGPT: person context set (%d chars)", len(memories_string))

    def clear_person_context(self) -> None:
        """Remove the injected memory context (called on IDLE entry)."""
        self._person_context = ""
        log.debug("ChatGPT: person context cleared")

    def set_behavior_context(self, context: str) -> None:
        """Overlay lightweight runtime behavior guidance onto the system prompt."""
        self._behavior_context = context.strip()
        log.debug("ChatGPT: behavior context set (%d chars)", len(self._behavior_context))

    def clear_behavior_context(self) -> None:
        """Clear runtime behavior guidance."""
        self._behavior_context = ""
        log.debug("ChatGPT: behavior context cleared")

    def set_angry_mode(self, enabled: bool) -> None:
        """Overlay a grumpier runtime style onto Rex's normal voice."""
        self._mood_context = (
            "CURRENT MOOD: angry/grumpy mode is active. "
            "Be curt, irritated, sarcastic, and dismissive. "
            "Keep replies short and robotic. "
            "Stay family-safe. Do not threaten, swear, or become personally abusive."
            if enabled
            else ""
        )
        log.debug("ChatGPT: angry mode=%s", enabled)

    def _effective_system_message(self, base_prompt: str | None = None) -> dict[str, str]:
        """Build the active system prompt with optional runtime overlays."""
        base = base_prompt or self._system_message["content"]
        suffix = ""
        if self._person_context:
            suffix += (
                f"\n\nHere is what you know about the person you are talking to: "
                f"{self._person_context}"
            )
        if self._mood_context:
            suffix += f"\n\n{self._mood_context}"
        if self._behavior_context:
            suffix += f"\n\n{self._behavior_context}"
        if not suffix:
            return {"role": "system", "content": base}
        return {"role": "system", "content": base + suffix}

    @staticmethod
    def _extract_json_line(raw_text: str) -> tuple[str, bool]:
        """Return the short-line field from JSON output when present."""
        text = raw_text.strip()
        if not text:
            return "", False

        for pattern in _SHORT_META_PATTERNS[:2]:
            text = pattern.sub("", text).strip()

        candidates = [text]
        match = _SHORT_JSON_OBJECT_RE.search(text)
        if match and match.group(0) != text:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except Exception:
                continue
            if isinstance(data, dict):
                line = str(data.get("line") or "").strip()
                if line:
                    return line, True
        return raw_text, False

    @staticmethod
    def _has_incomplete_stage_direction_prefix(text: str) -> bool:
        """Return True when a stream prefix looks like a partial stage direction."""
        if not text:
            return False

        for opener, closer in (("(", ")"), ("[", "]"), ("*", "*")):
            if not text.startswith(opener):
                continue
            remainder = text[len(opener):]
            if closer in remainder or "\n" in remainder or len(remainder) > 160:
                return False
            if not remainder:
                return True
            return bool(re.fullmatch(r"[A-Za-z ,;:'-]{0,159}", remainder))
        return False

    @staticmethod
    def _strip_leading_spoken_prefix(
        text: str, *, final: bool = True
    ) -> tuple[str, bool, bool]:
        """Remove leading labels and stage directions from spoken text."""
        working = text
        changed = False

        while True:
            stripped = working.lstrip()
            if stripped != working:
                working = stripped
                changed = True

            meta_removed = False
            for pattern in _SHORT_META_PATTERNS[2:]:
                candidate = pattern.sub("", working, count=1)
                if candidate != working:
                    working = candidate
                    changed = True
                    meta_removed = True
                    break
            if meta_removed:
                continue

            stage_removed = False
            for pattern in _LEADING_STAGE_DIRECTION_PATTERNS:
                match = pattern.match(working)
                if match:
                    working = working[match.end():]
                    changed = True
                    stage_removed = True
                    break
            if stage_removed:
                continue

            if not final and ChatGPTClient._has_incomplete_stage_direction_prefix(working):
                return "", changed, True
            return working, changed, False

    @staticmethod
    def _clean_short_response_text(text: str) -> tuple[str, bool]:
        """Strip common labels, headers, and multiline/meta boilerplate."""
        original = text
        text = text.strip()
        if not text:
            return "", False

        for pattern in _SHORT_META_PATTERNS[:2]:
            text = pattern.sub("", text).strip()

        text, changed, _ = ChatGPTClient._strip_leading_spoken_prefix(text)

        kept_lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _SHORT_ALL_CAPS_HEADER_RE.fullmatch(line):
                continue
            if line.endswith(":") and any(
                token in line.lower()
                for token in ("rex", "persona", "assistant", "response", "output")
            ):
                continue
            kept_lines.append(line)

        text = kept_lines[0] if kept_lines else ""
        text = re.split(r"\s*(?:---+|===+|\|\|\|)\s*", text, maxsplit=1)[0].strip()
        for pattern in _SHORT_META_PATTERNS[2:]:
            updated = pattern.sub("", text).strip()
            if updated != text:
                changed = True
            text = updated
        text, prefix_changed, _ = ChatGPTClient._strip_leading_spoken_prefix(text)
        changed = changed or prefix_changed
        text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text).strip()
        text = re.sub(r"\s+", " ", text).strip()

        sentence_match = re.match(r"(.+?[.!?])(?:\s|$)", text)
        if sentence_match:
            text = sentence_match.group(1).strip()

        return text, changed or text != original.strip()

    @staticmethod
    def _clean_chat_reply_text(text: str) -> tuple[str, bool]:
        """Clean streamed chat output before logging/storing it in history."""
        original = text.strip()
        cleaned, changed = ChatGPTClient._clean_short_response_text(original)
        if not cleaned:
            return "", changed

        sentence_matches = re.findall(r"[^.!?]+[.!?]", cleaned)
        if sentence_matches:
            cleaned = " ".join(match.strip() for match in sentence_matches[:2]).strip()

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return "", changed
        return cleaned, changed or cleaned != original

    @staticmethod
    def _is_valid_short_response(line: str, *, max_words: int = 18) -> bool:
        """Validate the strict short-response contract for voice playback."""
        if not line:
            return False
        stripped = line.strip()
        if "\n" in stripped:
            return False
        if any(ch in stripped for ch in "*[]{}"):
            return False
        lowered = stripped.lower()
        if any(term in lowered for term in _SHORT_META_TERMS):
            return False
        if "brett " in lowered or lowered.startswith("brett") or ", brett" in lowered:
            return False
        if len(re.findall(r"[.!?]", stripped)) > 1:
            return False
        words = re.findall(r"\b[\w'-]+\b", stripped)
        if not words or len(words) > max_words:
            return False
        lowered_words = [word.lower() for word in words]
        if len(lowered_words) >= 6 and len(set(lowered_words)) <= len(lowered_words) / 2:
            return False
        return True

    def _fallback_short_line(self, branch: str) -> str:
        pool = _SHORT_FALLBACKS.get(branch) or _SHORT_FALLBACKS["react_to_answer"]
        line = random.choice(pool)
        log.warning("LLM short branch=%s using template fallback: %r", branch, line)
        return line

    @staticmethod
    def _acknowledgement_fallback(answer: str, *, summary: str = "", category: str = "") -> str:
        """Return a grounded acknowledgement when the short LLM helper falls back."""
        source = (summary or answer).strip().strip("\"'").rstrip(".!?")
        if not source:
            return "Noted, lifeform."

        fragment = re.sub(r"^(?:well[, ]+)?(?:uh[, ]+)?(?:um[, ]+)?", "", source, flags=re.IGNORECASE).strip()
        replacements = (
            (r"^i am\b", "you're"),
            (r"^i'm\b", "you're"),
            (r"^i’m\b", "you're"),
            (r"^im\b", "you're"),
            (r"^i have\b", "you've got"),
            (r"^i've got\b", "you've got"),
            (r"^i’ve got\b", "you've got"),
            (r"^i will\b", "you're going to"),
            (r"^i'll\b", "you're going to"),
            (r"^i’ll\b", "you're going to"),
            (r"^i\b", "you"),
            (r"^my\b", "your"),
        )
        lowered = fragment.lower()
        for pattern, replacement in replacements:
            updated = re.sub(pattern, replacement, lowered, count=1)
            if updated != lowered:
                fragment = updated
                break
        else:
            fragment = lowered

        fragment = re.sub(r"\s+", " ", fragment).strip(" ,;:")
        if not fragment:
            return "Noted, lifeform."

        if "plan" in category or "event" in category:
            return f"Oh, so {fragment}. Bold little itinerary, lifeform."
        return f"Oh, so {fragment}. I'll allow it, lifeform."

    def _generate_short_line(self, *, branch: str, user_prompt: str) -> str:
        """Generate a short voice-friendly line with cleanup, retry, and fallback."""
        attempts = (
            {
                "system": _SHORT_JSON_SYSTEM_PROMPT,
                "response_format": {"type": "json_object"},
                "temperature": 0.45,
                "label": "json",
            },
            {
                "system": _SHORT_TEXT_SYSTEM_PROMPT + "\n\n" + _SHORT_RETRY_SUFFIX,
                "response_format": None,
                "temperature": 0.2,
                "label": "retry_text",
            },
        )

        for attempt_number, attempt in enumerate(attempts, start=1):
            try:
                kwargs: dict = {
                    "model": self._chat_model,
                    "messages": [
                        self._effective_system_message(attempt["system"]),
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "temperature": attempt["temperature"],
                    "max_tokens": 60,
                    "stop": self._build_stop_sequences(*_SHORT_STOP_SEQUENCES),
                }
                if attempt["response_format"] is not None:
                    kwargs["response_format"] = attempt["response_format"]
                if self._use_local:
                    kwargs["extra_body"] = {"keep_alive": -1}

                response = self._client.chat.completions.create(**kwargs)
                raw = str(response.choices[0].message.content or "").strip()
                extracted, used_json = self._extract_json_line(raw)
                cleaned, cleaned_changed = self._clean_short_response_text(extracted)
                if self._is_valid_short_response(cleaned):
                    log.info(
                        "LLM short branch=%s attempt=%d mode=%s json=%s cleaned=%s line=%r",
                        branch,
                        attempt_number,
                        attempt["label"],
                        used_json,
                        cleaned_changed,
                        cleaned,
                    )
                    return cleaned
                log.warning(
                    "LLM short branch=%s attempt=%d mode=%s rejected raw=%r cleaned=%r",
                    branch,
                    attempt_number,
                    attempt["label"],
                    raw,
                    cleaned,
                )
            except Exception:
                log.exception(
                    "LLM short branch=%s attempt=%d mode=%s failed",
                    branch,
                    attempt_number,
                    attempt["label"],
                )

        return self._fallback_short_line(branch)

    # ------------------------------------------------------------------
    # Utility — structured one-shot calls (always use cloud OpenAI for
    # reliable JSON output; never stream; not added to conversation history)
    # ------------------------------------------------------------------

    def extract_memory(self, question: str, answer: str) -> dict | None:
        """Extract a structured memory from a Q&A exchange.

        Returns a dict with keys: category, key, value, expires_at,
        follow_up_after — or None on any error.
        """
        from datetime import date
        today = date.today().isoformat()
        try:
            response = self._vision_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Today's date is {today}. "
                            "Extract a memory from this answer. Return JSON with keys: "
                            "category (one of: preference/event/plan/fact/relationship), "
                            "key (snake_case label, e.g. favorite_food, favorite_music, favorite_movie, pets, profession, has_kids, current_activity), "
                            "value (clean normalized one-sentence summary in third person, grounded only in the answer), "
                            "expires_at (ISO-8601 date if time-sensitive, else null), "
                            "follow_up_after (ISO-8601 date one day after a planned event, else null)."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Question asked: {question}\nAnswer: {answer}",
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=150,
                temperature=0.2,
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            log.exception("extract_memory: failed")
            return None

    def react_to_answer(self, question: str, answer: str) -> str:
        """Return a short Rex-style one-sentence reaction to an enrollment answer.

        Uses the active text model with strict short-response cleanup.
        """
        return self._generate_short_line(
            branch="react_to_answer",
            user_prompt=(
                "React to this answer as Rex with a warm affectionate roast.\n"
                f"Question: {question}\n"
                f"Answer: {answer}"
            ),
        )

    def check_memorable(self, message: str) -> dict | None:
        """Check whether a conversational message contains a memorable fact/preference/plan.

        Returns dict with 'memorable' bool key (plus memory fields if True),
        or None on error.
        """
        from datetime import date
        today = date.today().isoformat()
        try:
            response = self._vision_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Today's date is {today}. "
                            "Does this message contain a memorable fact, preference, plan, or event "
                            "worth storing for future reference? "
                            'If yes return JSON: {"memorable": true, "category": ..., "key": ..., '
                            '"value": ..., "expires_at": ..., "follow_up_after": ...}. '
                            'If no return {"memorable": false}. '
                            "category: preference/event/plan/fact/relationship. "
                            "key: snake_case. value: clean normalized one-sentence summary in third person. "
                            "expires_at: ISO date if time-sensitive else null. "
                            "follow_up_after: ISO date one day after a planned event else null."
                        ),
                    },
                    {"role": "user", "content": f"Message: {message}"},
                ],
                response_format={"type": "json_object"},
                max_tokens=150,
                temperature=0.1,
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            log.exception("check_memorable: failed")
            return None

    def generate_followup(self, memory_value: str, created_at: str) -> str:
        """Generate a Rex-style follow-up question about a stored memory.

        Uses the active text model with strict short-response cleanup.
        """
        return self._generate_short_line(
            branch="generate_followup",
            user_prompt=(
                "Ask one natural follow-up question about this stored memory in Rex's voice.\n"
                f"Past memory: {memory_value}\n"
                f"Recorded on: {created_at or 'unknown'}"
            ),
        )

    @staticmethod
    def _memory_callback_fallback(memory: dict) -> str:
        """Return a grounded one-line callback question when model generation fails."""
        key = str(memory.get("key") or "").strip().lower()
        category = str(memory.get("category") or "").strip().lower()
        summary = str(memory.get("value") or "").strip()
        raw_answer = str(memory.get("answer_text") or memory.get("raw_quote") or "").strip()
        question_text = str(memory.get("question_text") or "").strip()
        source = " ".join(part for part in (summary, raw_answer, question_text) if part).strip()
        lower = source.lower()

        negative_patterns = (
            " no ",
            " no.",
            " no,",
            "none",
            "don't have",
            "do not have",
            "does not have",
            "without",
            "never",
        )
        is_negative = any(pattern in f" {lower} " for pattern in negative_patterns)

        names_match = re.search(
            r"\bnamed ([A-Z][a-z]+(?:\s*(?:,|and)\s*[A-Z][a-z]+)*)",
            source,
        )
        named_items = names_match.group(1) if names_match else ""

        if "pet" in key or "dog" in lower or "cat" in lower:
            if named_items:
                return f"You mentioned {named_items} before. How are those little chaos beasts doing?"
            if is_negative:
                return "You said no pets before. Ever wanted one, or do you prefer the peaceful option?"
            return "How are your pets doing these days, lifeform?"

        if "profession" in key or "job" in key or "work" in key or "occupation" in key:
            if is_negative:
                return "You made work sound delightfully bleak. Is that still the situation, lifeform?"
            if summary:
                cleaned = summary.rstrip(".")
                return f"You said {cleaned}. Still true, or did your career do a weird little plot twist?"
            return "What are you doing for work these days, lifeform?"

        if "kid" in key or "child" in key or category == "relationship":
            if is_negative:
                return "You said no kids before. Always the plan, or just how the galaxy shook out?"
            if named_items:
                return f"You mentioned {named_items} before. How are they doing lately?"
            return "How are the kid-related adventures going, lifeform?"

        if any(token in key for token in ("favorite_food", "favorite_music", "favorite_movie")) or category == "preference":
            if summary:
                cleaned = summary.rstrip(".")
                return f"You once went with {cleaned}. Still your favorite, or has your taste improved?"
            return "Still standing by that preference, lifeform, or was that a temporary malfunction?"

        if "activity" in key or "plan" in key or category in {"plan", "event"}:
            if summary:
                cleaned = summary.rstrip(".")
                return f"You mentioned {cleaned}. How's that little adventure going now?"
            return "How's that grand little plan going, lifeform?"

        if summary:
            cleaned = summary.rstrip(".")
            return f"You told me {cleaned}. Still accurate, or has the saga changed since then?"
        return "Still standing by that story, lifeform?"

    def generate_memory_callback(self, memory: dict) -> str:
        """Generate a category-aware callback question grounded in stored memory data."""
        category = str(memory.get("category") or "").strip() or "fact"
        key = str(memory.get("key") or "").strip() or "memory"
        normalized_summary = str(memory.get("value") or "").strip()
        raw_answer = str(memory.get("answer_text") or memory.get("raw_quote") or "").strip()
        original_question = str(memory.get("question_text") or "").strip()
        created_at = str(memory.get("created_at") or "").strip() or "unknown"
        callback_count = int(memory.get("callback_count") or 0)

        line = self._generate_short_line(
            branch="generate_memory_callback",
            user_prompt=(
                "Ask one callback question for a known returning person using ONLY the stored memory below.\n"
                "Rules:\n"
                "- one spoken sentence only\n"
                "- warm, roasty, curious Rex voice\n"
                "- grounded in the stored memory, no invented facts\n"
                "- if the stored answer is negative, ask a curious expansion question\n"
                "- if it mentions specific items or names, ask about those details\n"
                "- if it is a preference, ask whether it is still true or ask for elaboration\n"
                "- avoid parroting the original question verbatim\n"
                f"Category: {category}\n"
                f"Key: {key}\n"
                f"Normalized summary: {normalized_summary or 'unknown'}\n"
                f"Original question: {original_question or 'none'}\n"
                f"Stored answer: {raw_answer or 'none'}\n"
                f"Recorded at: {created_at}\n"
                f"Previous callback count: {callback_count}"
            ),
        )
        if line and line not in _SHORT_FALLBACKS.get("generate_memory_callback", ()):
            return line
        return self._memory_callback_fallback(memory)

    def generate_memory_acknowledgement(
        self,
        answer: str,
        *,
        category: str = "",
        key: str = "",
        tags: str = "",
        summary: str = "",
    ) -> str:
        """React briefly to a stored-memory answer using the actual spoken content."""
        line = self._generate_short_line(
            branch="generate_memory_acknowledgement",
            user_prompt=(
                "React to the user's answer in one short spoken Rex sentence.\n"
                "Rules:\n"
                "- directly reference what they just said\n"
                "- no follow-up question\n"
                "- warm, roasty, conversational\n"
                "- avoid generic acknowledgements unless absolutely necessary\n"
                f"Category: {category or 'unknown'}\n"
                f"Key: {key or 'unknown'}\n"
                f"Tags: {tags or 'none'}\n"
                f"Normalized summary: {summary or 'none'}\n"
                f"User answer: {answer}"
            ),
        )
        if line and line not in _SHORT_FALLBACKS.get("generate_memory_acknowledgement", ()):
            return line
        return self._acknowledgement_fallback(
            answer,
            summary=summary,
            category=category,
        )

    def refresh_memory_from_callback(
        self,
        memory: dict,
        question: str,
        answer: str,
    ) -> dict | None:
        """Merge a callback answer back into an existing memory row."""
        from datetime import date

        today = date.today().isoformat()
        try:
            response = self._vision_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Today's date is {today}. "
                            "Update an existing memory after a callback answer. Return JSON with keys: "
                            "category, key, value, answer_text, expires_at, follow_up_after. "
                            "Requirements: "
                            "value must be a compact normalized third-person summary grounded only in the stored memory and new answer. "
                            "Preserve the original topic unless the new answer clearly changes it. "
                            "If the new answer adds useful detail, fold it into the summary. "
                            "If the new answer contradicts the old memory, update the summary to the newer truth. "
                            "answer_text must contain the new raw callback answer."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Existing category: {memory.get('category') or ''}\n"
                            f"Existing key: {memory.get('key') or ''}\n"
                            f"Existing normalized summary: {memory.get('value') or ''}\n"
                            f"Existing original answer: {memory.get('raw_quote') or ''}\n"
                            f"Existing latest answer: {memory.get('answer_text') or ''}\n"
                            f"Original question: {memory.get('question_text') or ''}\n"
                            f"Callback question: {question}\n"
                            f"Callback answer: {answer}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=220,
                temperature=0.2,
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            log.exception("refresh_memory_from_callback: failed")
            return None

    def generate_activity_followup(
        self, activity_text: str, created_at: str = "", *, same_day: bool = False
    ) -> str:
        """Turn a stored activity/plan into a natural Rex-style follow-up question.

        Returns "" on any error.
        """
        timing = (
            "The activity is happening today or this weekend."
            if same_day
            else "This is a previously mentioned activity or plan."
        )
        return self._generate_short_line(
            branch="generate_activity_followup",
            user_prompt=(
                "Rewrite this stored activity as one natural follow-up question in Rex's voice. "
                "Do not quote clunky first-person phrasing back verbatim.\n"
                f"Stored activity: {activity_text}\n"
                f"Recorded at: {created_at or 'unknown'}\n"
                f"Timing context: {timing}"
            ),
        )

    def generate_activity_reply(
        self, activity_text: str, *, weekend: bool = False
    ) -> str:
        """Turn a fresh first-person activity answer into a natural Rex reply.

        Returns "" on any error.
        """
        timing = "this weekend" if weekend else "today"
        return self._generate_short_line(
            branch="generate_activity_reply",
            user_prompt=(
                "Turn the user's stated plan into one natural Rex reply or follow-up question. "
                "Convert first-person phrasing into natural direct speech when needed.\n"
                f"User plan: {activity_text}\n"
                f"Timing context: {timing}"
            ),
        )

    def analyze_i_spy_scene(self, image: str) -> dict | None:
        """Return structured scene data for the I Spy mini-game."""
        try:
            response = self._vision_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Analyze this scene for a short I Spy game. "
                            "Return JSON with keys: scene_description and visible_objects. "
                            "scene_description should be a specific 1-2 sentence description. "
                            "visible_objects must be an array of 5 to 8 clearly visible physical objects "
                            "ordered from most obvious/common to less obvious. "
                            "Each object must have keys: name, category, prominent. "
                            "Use short everyday singular names, 1 to 3 words each. "
                            "Exclude people, body parts, walls, floors, ceilings, and tiny unreadable items. "
                            "Categories should be chosen from: furniture, electronics, container, decor, "
                            "food_drink, tool, clothing, bag, book_paper, toy, fixture, plant, other."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe the scene and list good I Spy candidate objects.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image}",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=350,
                temperature=0.2,
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            log.exception("analyze_i_spy_scene: failed")
            return None

    def analyze_chatty_scene(self, image: str) -> dict | None:
        """Return a detailed scene description and a short curious Rex reaction."""
        try:
            response = self._vision_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Analyze this scene for a curious droid scanning the room. "
                            "Return JSON with keys: scene_description and interesting_details. "
                            "scene_description must be a detailed 3-5 sentence description of what is visible, "
                            "covering layout, objects, decor, colors, lighting, and any obvious people, pets, "
                            "or activity if present. interesting_details must be an array of 2 to 4 short, "
                            "specific details worth commenting on. Be factual and concrete."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe everything visible and call out a few interesting details.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image}",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=400,
                temperature=0.2,
            )
            result = json.loads(response.choices[0].message.content)
        except Exception:
            log.exception("analyze_chatty_scene: description step failed")
            return None

        description = str(result.get("scene_description") or "").strip()
        details = result.get("interesting_details")
        if not isinstance(details, list):
            details = []
        details = [str(item).strip() for item in details if str(item).strip()]
        if not description:
            log.warning("analyze_chatty_scene: empty scene description")
            return None

        reaction = self.generate_chatty_scene_reaction(description, details)
        if not reaction:
            return None
        return {
            "scene_description": description,
            "interesting_details": details,
            "reaction": reaction,
        }

    def generate_chatty_scene_reaction(
        self, scene_description: str, interesting_details: list[str]
    ) -> str:
        """Turn a scene description into a short curious in-character line."""
        details_text = "\n".join(f"- {detail}" for detail in interesting_details) or "- No standout details extracted."
        try:
            response = self._vision_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are DJ R-3X ('Rex'), the droid DJ at Oga's Cantina. "
                            "React to the scene in 1 or 2 short sentences MAX. "
                            "This mode should feel curious, playful, and genuinely interested in the world. "
                            "A light roast is okay, but wonder and fascination should lead. "
                            "Mention one or two specific visible details. "
                            "Do not sound like a dry narrator. "
                            "Do not mention cameras, photos, images, or analysis. "
                            "No asterisks. No written sound effects."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Scene description:\n{scene_description}\n\n"
                            f"Interesting details:\n{details_text}"
                        ),
                    },
                ],
                max_tokens=80,
                temperature=1.0,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            log.exception("generate_chatty_scene_reaction: failed")
            if interesting_details:
                detail = interesting_details[0].rstrip(".")
                return (
                    f"Huh. {detail}. This room keeps giving me new material, and frankly I respect the commitment."
                )
            return ""

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
