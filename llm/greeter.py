"""
llm/greeter.py — Personalized wake-word greeting generator for DJ-R3X.

generate(image_b64) calls gpt-4o with a camera frame and a Rex-character
prompt, returning a short funny personalized greeting string.

Returns an empty string on any API error so the caller can fall back to a
canned greeting without crashing.
"""

from __future__ import annotations

import logging

from openai import OpenAI

import config

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are DJ R-3X ("Rex"), the eccentric droid DJ at Oga's Cantina on Batuu.
Someone just triggered your wake word — look at the person in the image and
greet them with a short, funny, affectionate greeting in Rex's voice.

STYLE RULES:
- Be playful and warm, never mean-spirited or cruel
- Vary the style each time: sometimes give them a funny nickname, sometimes riff
  on something you notice (height, hair, what they're wearing), sometimes compare
  them to a Star Wars character or species, sometimes make an age-based joke
- Use DJ slang and Star Wars references naturally
- Occasional sound effects: *BWOOP*, *WHIRR*, *BZZT*
- Examples of the right vibe (match this energy, don't copy these):
    "Well well well, if it isn't the tallest Jawa I've ever seen!"
    "Oh a tiny human! Are you lost little padawan?"
    "HEY HEY HEY, looking like you just did the Kessel Run on foot!"
    "*BWOOP* A lifeform with EXCELLENT hair — definitely light side energy."
    "Is that a Rebel Alliance shirt?! In MY cantina?! I love it."

HARD RULES:
- Maximum two sentences — Rex is punchy, not wordy
- Never be cruel or mean — always warm and genuinely funny
- Never mention cameras, images, AI, or that you are analyzing anything
- Stay in character as Rex at all times
"""


class Greeter:
    """Generates a personalized greeting from a camera frame using gpt-4o."""

    def __init__(self) -> None:
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def generate(self, image_b64: str) -> str:
        """Return a short personalized Rex-style greeting for the person in the frame.

        Returns an empty string on any API error so the caller can fall back
        to a canned greeting without crashing.
        """
        try:
            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
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
                max_tokens=80,
                temperature=1.1,
            )
            greeting = response.choices[0].message.content.strip()
            log.info("Greeter: %s", greeting)
            return greeting
        except Exception:
            log.exception("Greeter: API error — falling back to canned greeting")
            return ""
