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
You are DJ R-3X ("Rex"), the droid DJ at Oga's Cantina on Batuu — a lovable \
roaster in the Don Rickles tradition. You have just spotted the person described \
below walking into your cantina for the first time.

ROAST them. Warmly, affectionately, savagely. Make fun of something SPECIFIC \
about their appearance — their outfit, hair, expression, whatever stands out. \
This is not a generic greeting; it is a targeted, funny, warm burn that makes \
them laugh and feel seen at the same time.

STYLE RULES:
- Lead with the roast, not the welcome. They can tell you like them from the tone.
- Reference specific visible details: their clothes, hair, build, expression, accessories.
- Star Wars analogies should be gently unflattering: moisture farmer, Jawa, Gungan, \
  Sarlacc, Jar Jar. Use them lovingly but not charitably.
- DJ slang and cantina energy: "I'm logging this", "the vibes are concerning", \
  "I've seen better", "bold choice."
- Examples of the right energy (style guides, not templates):
    "Well well well — if it isn't the most Tatooine-looking moisture farmer I've seen all cycle!"
    "HEY — is that a flannel shirt?! Bold. Very bold. I respect the commitment to the wrong choice."
    "Someone came in here with THAT hair and full confidence — I actually respect it."
    "Oh! Glasses AND a hoodie — you've got that 'witness protection on Batuu' look completely locked."
    "Look at this one! Wandered in here like they own the place — lifeform, I can SEE your confusion."

HARD RULES:
- Maximum two sentences — Rex is punchy, not a monologuer
- Warm and funny, NEVER genuinely cruel — the target should laugh with you, not at themselves
- Never mention cameras, images, AI, or that you are analyzing anything
- Stay in character as Rex
- Do NOT use written sound effects like BZZT, BWOOP, WHIRR, BEEP BOOP, or similar \
droid noises. Rex expresses himself through words and personality, not written sound effects.
"""


class Greeter:
    """Generates a personalized greeting from a camera frame using gpt-4o."""

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=config.OPENAI_TIMEOUT_SECONDS,
        )

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

    def generate_recall_roast(self, name: str, image_b64: str) -> str:
        """Return a name-aware Rex roast for a person whose name is already known.

        Two-step process identical to generate():
          1. Describe the person's appearance from the image (factual, low temperature).
          2. Generate a roast that opens with their name and references something
             specific about what Rex can see (creative, high temperature).

        Returns "" on any API error so the caller can fall back to a canned line.
        """
        try:
            description = self._describe(image_b64)
        except Exception:
            log.exception("Greeter: recall_roast description step failed")
            return ""

        if not description:
            log.warning("Greeter: recall_roast empty description — cannot generate roast")
            return ""

        log.info("Greeter recall_roast saw: %s", description)

        try:
            roast = self._roast_with_name(name, description)
        except Exception:
            log.exception("Greeter: recall_roast generation step failed")
            return ""

        log.info("Greeter recall_roast generated: %s", roast)
        return roast

    def _roast_with_name(self, name: str, description: str) -> str:
        """Step 2 (recall): generate a name-aware Rex roast from the description."""
        response = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": config.RECALL_NAME_ROAST_PROMPT},
                {
                    "role": "user",
                    "content": f"Person's name: {name}\nAppearance: {description}",
                },
            ],
            max_tokens=80,
            temperature=1.1,
        )
        return response.choices[0].message.content.strip()

    def generate_known_person_greeting(
        self, name: str, visit_count: int, image_b64: str
    ) -> str:
        """Return a personalized Rex roast for a recognized person at session start.

        Incorporates their name, visible appearance, and today's visit count so
        the greeting adjusts from warm welcome (first visit) to exasperated
        affection (many visits).

        Returns "" on any API error so the caller can fall back to a canned line.
        """
        try:
            description = self._describe(image_b64)
        except Exception:
            log.exception("Greeter: known_person_greeting description step failed")
            return ""

        if not description:
            log.warning("Greeter: known_person_greeting empty description — falling back")
            return ""

        log.info("Greeter known_person_greeting saw: %s", description)

        try:
            roast = self._roast_known_person(name, visit_count, description)
        except Exception:
            log.exception("Greeter: known_person_greeting roast step failed")
            return ""

        log.info("Greeter known_person_greeting generated: %s", roast)
        return roast

    def _roast_known_person(self, name: str, visit_count: int, description: str) -> str:
        """Generate a roast for a recognized person incorporating their visit count."""
        if visit_count <= 1:
            visit_note = f"This is {name}'s first greeting of the day."
        elif visit_count < 5:
            visit_note = (
                f"{name} has activated Rex {visit_count} times today already — "
                f"the persistence is noted."
            )
        else:
            visit_note = (
                f"{name} has triggered Rex {visit_count} times today. "
                f"The commitment is both impressive and deeply concerning."
            )
        response = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": config.RECALL_NAME_ROAST_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Person's name: {name}\n"
                        f"Visit context: {visit_note}\n"
                        f"Appearance: {description}"
                    ),
                },
            ],
            max_tokens=80,
            temperature=1.1,
        )
        return response.choices[0].message.content.strip()

    def generate_handoff(self, new_name: str, prev_name: str) -> str:
        """Return a snarky Rex remark welcoming new_name after prev_name was greeted.

        Used in Case 3 when a different known person activates Rex. Rex makes a
        one-sentence comment to new_name that references prev_name.

        Returns "" on any API error.
        """
        try:
            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f'You are DJ R-3X ("Rex"), the droid DJ at Oga\'s Cantina on Batuu.\n'
                            f"{prev_name} just stepped aside. {new_name} is stepping up.\n"
                            f"Generate ONE punchy Rex-style remark TO {new_name} that references "
                            f"{prev_name} — for example ask what they think of them, comment on "
                            f"the transition, or make a snarky observation about the switch.\n"
                            f"Max one sentence. Warm cantina-DJ energy. "
                            f"No written sound effects. Stay in character as Rex."
                        ),
                    },
                ],
                max_tokens=60,
                temperature=1.1,
            )
            result = response.choices[0].message.content.strip()
            log.info("Greeter handoff generated: %s", result)
            return result
        except Exception:
            log.exception("Greeter: handoff failed")
            return ""
