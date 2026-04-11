"""Helpers for the DJ-R3X I Spy mini-game.

The selection and clue logic stays local and deterministic so it is easy to
test and doesn't depend on free-form LLM phrasing.
"""

from __future__ import annotations

from dataclasses import dataclass

from commands.parser import normalize

_OBJECT_CLUES: dict[str, str] = {
    "apple": "fruit",
    "backpack": "bag",
    "banana": "fruit",
    "bottle": "drink container",
    "book": "book",
    "chair": "piece of furniture",
    "clock": "timepiece",
    "coffee mug": "drink container",
    "computer": "electronic device",
    "couch": "piece of furniture",
    "cup": "drink container",
    "desk": "piece of furniture",
    "door": "house fixture",
    "headphones": "audio gear",
    "keyboard": "computer accessory",
    "lamp": "light",
    "laptop": "computer",
    "monitor": "screen",
    "mouse": "computer accessory",
    "mug": "drink container",
    "notebook": "paper item",
    "phone": "electronic device",
    "picture frame": "decor item",
    "plant": "plant",
    "remote": "controller",
    "shoe": "piece of clothing",
    "sofa": "piece of furniture",
    "table": "piece of furniture",
    "television": "screen",
    "tv": "screen",
    "water bottle": "drink container",
    "window": "house fixture",
}

_CATEGORY_CLUES: dict[str, str] = {
    "bag": "bag",
    "book_paper": "paper item",
    "clothing": "piece of clothing",
    "container": "container",
    "decor": "decor item",
    "electronics": "electronic device",
    "fixture": "house fixture",
    "food_drink": "food item",
    "furniture": "piece of furniture",
    "plant": "plant",
    "tool": "tool",
    "toy": "toy",
}

_ALIASES: dict[str, set[str]] = {
    "cell phone": {"cell phone", "phone", "smartphone", "mobile phone"},
    "coffee mug": {"coffee mug", "mug", "cup"},
    "couch": {"couch", "sofa"},
    "laptop": {"laptop", "computer"},
    "television": {"television", "tv", "monitor", "screen"},
    "water bottle": {"water bottle", "bottle"},
}

_GENERIC_OBJECTS = {
    "floor", "wall", "ceiling", "room", "person", "man", "woman", "child",
    "face", "hand", "arm", "shirt", "pants",
}


@dataclass(frozen=True)
class ISpyRound:
    scene_description: str
    answer: str
    clue: str

    @property
    def article(self) -> str:
        return "an" if self.clue[:1].lower() in "aeiou" else "a"


def build_round(analysis: dict | None) -> ISpyRound | None:
    """Pick a hidden answer from structured vision output and derive a clue."""
    if not analysis:
        return None

    scene_description = str(analysis.get("scene_description", "")).strip()
    for item in analysis.get("visible_objects", []):
        if not isinstance(item, dict):
            continue
        answer = _clean_name(item.get("name"))
        if not answer or answer in _GENERIC_OBJECTS:
            continue
        clue = _derive_clue(answer, _clean_name(item.get("category")))
        if clue and normalize(clue) != normalize(answer):
            return ISpyRound(
                scene_description=scene_description,
                answer=answer,
                clue=clue,
            )
    return None


def guess_matches(guess: str, answer: str) -> bool:
    """Return True when the spoken guess matches the hidden answer."""
    g = _strip_guess_prefixes(normalize(guess))
    a = normalize(answer)
    if not g or not a:
        return False

    answer_aliases = _alias_set(a)
    guess_aliases = _alias_set(g)
    if answer_aliases & guess_aliases:
        return True

    if a in g or g in a:
        return True

    answer_head = _head_noun(a)
    guess_head = _head_noun(g)
    return bool(answer_head and guess_head and _alias_set(answer_head) & _alias_set(guess_head))


def _derive_clue(answer: str, category: str) -> str:
    answer_norm = normalize(answer)
    if answer_norm in _OBJECT_CLUES:
        return _OBJECT_CLUES[answer_norm]
    if category in _CATEGORY_CLUES:
        return _CATEGORY_CLUES[category]
    head = _head_noun(answer_norm)
    if head in _OBJECT_CLUES:
        return _OBJECT_CLUES[head]
    return ""


def _head_noun(text: str) -> str:
    parts = text.split()
    return parts[-1] if parts else ""


def _clean_name(value: object) -> str:
    if not value:
        return ""
    text = str(value).strip().lower()
    return " ".join(text.split()[:3])


def _alias_set(text: str) -> set[str]:
    base = {text}
    head = _head_noun(text)
    if head:
        base.add(head)
    for canonical, aliases in _ALIASES.items():
        if text == canonical or text in aliases or head in aliases:
            return set(aliases) | {canonical}
    return base


def _strip_guess_prefixes(text: str) -> str:
    prefixes = (
        "is it ", "its ", "it's ", "i think its ", "i think it's ",
        "my guess is ", "is the answer ", "is that ", "maybe ",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text.strip()
