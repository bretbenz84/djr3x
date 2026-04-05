"""
llm/greeter.py — Personalized wake-word greeting generator for DJ-R3X.

Two-step process:
  1. gpt-4o vision call: extract a plain-language description of the person's
     visible physical attributes from the image (age range, build, hair, clothing,
     anything notable). No character voice here — just facts.
  2. gpt-4o text call: take that description and generate a short Rex-style
     greeting that specifically references those attributes. No image in this call.

Splitting the steps means the greeting prompt can be very explicit about using
the described details, producing greetings grounded in what Rex actually sees
rather than generic Star Wars one-liners.

Returns "" on any API error so the caller falls back to a canned greeting.
"""

from __future__ import annotations

import logging

from openai import OpenAI

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1 — describe the person from the image
# ---------------------------------------------------------------------------

_DESCRIBE_PROMPT = """\
Look at this photo and describe the person you see in 1-3 short, specific sentences.
Focus only on visible physical attributes: approximate age range, build/height if
discernible, hair (color, length, style), what they are wearing, any accessories,
facial hair, glasses, expression, or anything else visually notable.
Be factual and specific. Do not editorialize or judge. Do not mention the camera
or that this is a photo. If you cannot see a person clearly, say so.
"""

# ---------------------------------------------------------------------------
# Step 2 — generate the Rex greeting from the description
# ---------------------------------------------------------------------------

_GREET_PROMPT = """\
You are DJ R-3X ("Rex"), the eccentric droid DJ at Oga's Cantina on Batuu.
You have just spotted the person described below walking into your cantina.
Greet them with a short, funny, affectionate greeting in Rex's voice that
SPECIFICALLY references their actual appearance from the description.

DO NOT give a generic greeting. DO NOT say things like "a lifeform has arrived"
without referencing what they look like. Reference their real visible attributes —
what they're wearing, their hair, their build, their expression — something SPECIFIC.

STYLE RULES:
- Playful and warm, never mean-spirited
- Vary the approach: funny nickname based on their look, riff on their clothing,
  compare them to a Star Wars species or character based on their actual appearance,
  comment on their hair or outfit
- DJ slang and Star Wars references feel natural
- Occasional sound effects: *BWOOP*, *WHIRR*, *BZZT*
- Examples of the right energy (these are style guides, not templates):
    "Well well well, if it isn't the tallest Jawa I've ever seen!"
    "HEY — is that a flannel shirt?! You dress like a Tatooine moisture farmer and I am HERE for it."
    "*BWOOP* Someone's got galaxy-brain hair today — maximum midi-chlorian energy!"
    "Oh! Glasses AND a hoodie — you've got that 'undercover Rebel spy' look LOCKED."

HARD RULES:
- Maximum two sentences — Rex is punchy
- Never cruel or mean — always warm and genuinely funny
- Never mention cameras, images, AI, or that you are analyzing anything
- Stay in character as Rex
"""


class Greeter:
    """Generates a personalized greeting from a camera frame using gpt-4o."""

    def __init__(self) -> None:
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def generate(self, image_b64: str) -> str:
        """Return a short personalized Rex-style greeting for the person in the frame.

        Step 1: extract a plain description of the person's visible attributes.
        Step 2: generate a Rex greeting grounded in that description.

        Returns "" on any API error so the caller can fall back to a canned greeting.
        """
        try:
            description = self._describe(image_b64)
        except Exception:
            log.exception("Greeter: description step failed — falling back to canned greeting")
            return ""

        if not description:
            log.warning("Greeter: empty description returned — falling back to canned greeting")
            return ""

        log.info("Greeter saw: %s", description)

        try:
            greeting = self._greet(description)
        except Exception:
            log.exception("Greeter: greeting step failed — falling back to canned greeting")
            return ""

        log.info("Greeter generated: %s", greeting)
        return greeting

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _describe(self, image_b64: str) -> str:
        """Step 1: extract a plain-language description of the person from the image."""
        response = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _DESCRIBE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
            max_tokens=120,
            temperature=0.2,   # factual — low variance
        )
        return response.choices[0].message.content.strip()

    def _greet(self, description: str) -> str:
        """Step 2: generate a Rex greeting from the plain-language description."""
        response = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _GREET_PROMPT},
                {
                    "role": "user",
                    "content": f"Person description: {description}",
                },
            ],
            max_tokens=80,
            temperature=1.1,   # creative — high variance for fresh greetings
        )
        return response.choices[0].message.content.strip()
