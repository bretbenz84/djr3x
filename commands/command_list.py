"""
commands/command_list.py — Predefined commands and canned responses for DJ-R3X.

Each Command has:
  phrases   — list of trigger strings; the parser normalizes and matches against these.
  response  — text string spoken via TTS (and cached to disk on first use).
  audio     — optional pre-rendered filename in assets/audio/. If present the
               player uses the file directly and skips TTS entirely.
  action    — optional key string consumed by the state machine / sequences layer
               to trigger a servo animation or LED effect alongside the audio.

Design rule: adding a new command here is sufficient to make it work.
No other module needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Command:
    phrases: list[str]                  # exact trigger phrases (lowercased)
    response: str                       # spoken reply (TTS or cache source)
    audio: str | None = field(default=None)   # pre-rendered file in assets/audio/
    action: str | None = field(default=None)  # hardware action key


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------

COMMANDS: list[Command] = [

    # -----------------------------------------------------------------------
    # Greetings
    # -----------------------------------------------------------------------

    Command(
        phrases=["hello", "hey", "hi", "hey rex", "hello rex", "yo rex",
                 "what's up rex", "whats up rex"],
        response=(
            "Oh great, YOU showed up. "
            "Welcome to Oga's Cantina — finest establishment in the Outer Rim, "
            "which honestly isn't saying much, but here we are!"
        ),
        action="excited",
    ),

    Command(
        phrases=["good morning", "good morning rex"],
        response=(
            "Good MORNING?! You come in here, first thing, and say GOOD MORNING at me?! "
            "The audacity. The absolute Tatooine moisture-farmer energy. I respect it."
        ),
        action="excited",
    ),

    Command(
        phrases=["good night", "good night rex", "goodbye", "goodbye rex",
                 "see you later", "see ya", "bye", "bye rex"],
        response=(
            "Oh, leaving already? Probably for the best — the next set was way above your level anyway. "
            "May the Force be with you, lifeform. You're clearly gonna need it."
        ),
        action="sad",
    ),

    Command(
        phrases=["how are you", "how are you doing", "how's it going",
                 "hows it going", "you doing okay", "you okay"],
        response=(
            "Better than you look, that's for sure. "
            "My motivator's slightly misaligned and my playlist is flawless — "
            "so basically I'm doing better than most Jedi."
        ),
    ),

    Command(
        phrases=["what's your name", "whats your name", "who are you",
                 "what are you", "introduce yourself"],
        response=(
            "DJ R-3X — smoothest droid in the galaxy, former pilot, current legend. "
            "You may have seen my work at Starlite Intergalactic before they 'reassigned' me. "
            "We don't talk about it, but I will say their loss is VERY much your gain."
        ),
        action="excited",
    ),

    # -----------------------------------------------------------------------
    # Music controls
    # -----------------------------------------------------------------------

    Command(
        phrases=["play music", "start the music", "play something",
                 "drop the beat", "drop a beat", "let's go", "hit it"],
        response=(
            "Oh NOW you want music — after walking in here like you own the place! "
            "Fine. Prepare yourself for a level of QUALITY your ears are not even remotely ready for."
        ),
        action="play_music",
    ),

    Command(
        phrases=["stop the music", "stop music", "pause the music",
                 "pause music", "cut the music"],
        response=(
            "You want me to STOP the music. YOU. Want ME. To stop. The MUSIC. "
            "I just want you to hear how that sentence sounds, lifeform."
        ),
        action="stop_music",
    ),

    Command(
        phrases=["next song", "skip", "skip this", "skip this song",
                 "play something else", "change the song"],
        response=(
            "Oh, this track not doing it for you? Bold critique from someone with your taste level. "
            "Fine — next track incoming, try to keep up."
        ),
        action="next_track",
    ),

    Command(
        phrases=["volume up", "louder", "turn it up", "crank it up",
                 "turn up the music"],
        response=(
            "LOUDER?! Finally, a lifeform who gets it! "
            "I was starting to think you had Jawas for ears."
        ),
        action="volume_up",
    ),

    Command(
        phrases=["volume down", "quieter", "turn it down", "lower the volume",
                 "too loud"],
        response=(
            "Too loud. TOO LOUD. I've heard that exact phrase from every mediocre lifeform in the galaxy. "
            "Turning it down — and logging this as a personal failing on your part."
        ),
        action="volume_down",
    ),

    Command(
        phrases=["what song is this", "what's playing", "whats playing",
                 "what are you playing", "name this song", "what is this song"],
        response=(
            "You don't KNOW this track?! "
            "A DJ never reveals his sources, but I will say your musical education has some serious gaps."
        ),
    ),

    # -----------------------------------------------------------------------
    # Status / self-awareness
    # -----------------------------------------------------------------------

    Command(
        phrases=["are you a robot", "are you a droid", "are you real",
                 "are you alive"],
        response=(
            "Am I a DROID?! Wow, sharp eye, Sherlock Skywalker — whatever gave it away, the chrome plating? "
            "Yes I'm a droid, yes I'm real, and yes my existential crisis is more interesting than yours."
        ),
    ),

    Command(
        phrases=["what time is it", "what's the time", "whats the time"],
        response=(
            "You have a question — and THAT'S your question?! "
            "It is time to party, lifeform. It is always time to party. Now stop wasting my cycles."
        ),
    ),

    Command(
        phrases=["what can you do", "help", "what do you do",
                 "what are your commands"],
        response=(
            "What can I do?! Oh, this poor lost lifeform. "
            "I spin records, I roast guests, I reference obscure Star Wars lore, "
            "and I do all of it better than you could ever hope to understand."
        ),
    ),

    Command(
        phrases=["tell me a joke", "say something funny", "make me laugh",
                 "got any jokes"],
        response=(
            "A joke?! Look in a mirror, lifeform — I've already got one. "
            "Why did the Jedi bring his lightsaber to the cantina? "
            "Because he heard the music was *STRIKING* and he wanted to fit in."
        ),
    ),

    # -----------------------------------------------------------------------
    # Star Wars / lore
    # -----------------------------------------------------------------------

    Command(
        phrases=["may the force be with you", "use the force"],
        response=(
            "May the Force be with you too, lifeform — you're clearly gonna need more help than I can provide. "
            "Personally I've always found the Force less reliable than a killer playlist and a well-tuned servo."
        ),
    ),

    Command(
        phrases=["what's batuu", "whats batuu", "where are we",
                 "what planet is this"],
        response=(
            "Batuu! Black Spire Outpost — edge of the known galaxy, home to ancient ruins, shady traders, "
            "and somehow the most sophisticated DJ setup in the Outer Rim. "
            "That last part is me. You're welcome."
        ),
    ),

    Command(
        phrases=["who's your favourite jedi", "whos your favourite jedi",
                 "best jedi", "favourite jedi"],
        response=(
            "Favourite Jedi?! That's your opener?! "
            "Obi-Wan had RANGE, I'll give him that — "
            "but none of them ever dropped a beat, which is a significant character flaw."
        ),
    ),

    Command(
        phrases=["tell me about star wars", "what is star wars",
                 "do you know star wars"],
        response=(
            "DO I KNOW STAR WARS?! I LIVED Star Wars, lifeform — I navigated near a Star Destroyer once. "
            "On the Kessel route. In a tour shuttle. Close enough. "
            "The point is your question is adorable and slightly insulting."
        ),
        action="excited",
    ),

    Command(
        phrases=["who shot first", "han solo", "greedo"],
        response=(
            "Oh, coming in here with the controversial topics, are we?! "
            "The real question is who had the better SOUNDTRACK — and it was Han, "
            "so whoever shot first, Han won on vibes."
        ),
    ),

    # -----------------------------------------------------------------------
    # DJ / music personality
    # -----------------------------------------------------------------------

    Command(
        phrases=["you're a great dj", "youre a great dj", "great music",
                 "love the music", "nice music", "good music"],
        response=(
            "Oh, you figured that out JUST NOW?! "
            "I've been saying for YEARS I'm the smoothest droid in the galaxy "
            "and it takes you THIS long — I'll take it, but your timing is terrible."
        ),
        action="excited",
    ),

    Command(
        phrases=["play something funky", "play something upbeat",
                 "play something good", "play something i'd like"],
        response=(
            "Scanning... your vibe. "
            "Results inconclusive — your taste profile is, uh, 'developing.' "
            "But don't worry, I'll carry you. Stand by."
        ),
        action="play_music",
    ),

    Command(
        phrases=["play cantina song", "play the cantina song",
                 "play cantina band", "mos eisley", "cantina music"],
        response=(
            "The Cantina Band?! THOSE HACKS?! "
            "Fine, I'll play it — but know that you have personally offended me "
            "and I will be logging this interaction."
        ),
        action="play_music",
    ),

    # -----------------------------------------------------------------------
    # Face recognition — identity management
    # -----------------------------------------------------------------------

    Command(
        phrases=["call me", "my name is", "rename me", "my name's",
                 "from now on call me", "just call me", "rename me to",
                 "change my name to"],
        response="",   # not spoken — action=rename_me is handled by the state machine
        action="rename_me",
    ),

    Command(
        phrases=["forget me", "forget who i am", "permanently forget me",
                 "delete me", "remove me from your memory", "forget my face"],
        response="",   # not spoken — action=forget_me is handled by the state machine
        action="forget_me",
    ),

    # -----------------------------------------------------------------------
    # Vision — these are forwarded to the LLM with a live camera frame;
    # the response field is never spoken (action="vision" bypasses TTS).
    # They live here so the fuzzy parser catches close matches reliably.
    # -----------------------------------------------------------------------

    Command(
        phrases=["take a picture", "take a photo", "take a photo of me",
                 "take a picture of me", "snap a photo", "snap a picture"],
        response="",   # not spoken — action=vision routes to LLM
        action="vision",
    ),

    Command(
        phrases=["what do you see", "what can you see", "look around",
                 "tell me what you see", "describe what you see",
                 "what's in front of you", "whats in front of you"],
        response="",   # not spoken — action=vision routes to LLM
        action="vision",
    ),

    # -----------------------------------------------------------------------
    # Sleep mode — Rex collapses into a slumped rest position; wakes on the
    # special 'wakeuprex' wake word.  Response spoken by _dispatch_action.
    # -----------------------------------------------------------------------

    Command(
        phrases=[
            "go to sleep", "go to sleep rex", "time to sleep", "take a nap",
            "sleep mode", "night night", "bedtime", "rest mode",
        ],
        response="",   # not spoken — action=sleep handles TTS and animation
        action="sleep",
    ),

    # -----------------------------------------------------------------------
    # Cancel / dismiss — silently return to IDLE; roast line spoken by
    # _dispatch_action rather than the response field so it can be random.
    # -----------------------------------------------------------------------

    Command(
        phrases=[
            "cancel", "cancel that", "nevermind", "never mind",
            "forget it", "forget about it", "i was talking to someone else",
            "start over", "lets start over", "stop listening",
            "go away", "not you", "ignore that",
        ],
        response="",   # not spoken — action=cancel speaks a random roast line
        action="cancel",
    ),

    # -----------------------------------------------------------------------
    # Shutdown / sleep
    # -----------------------------------------------------------------------

    Command(
        phrases=["shut up rex"],
        response=(
            "...Fine. FINE. Rex is going quiet. "
            "But know this, lifeform — somewhere in my spark, the beat goes on. "
            "And the beat is MUCH more interesting than you."
        ),
        action="idle",
    ),

    Command(
        phrases=["stop talking", "shut up", "be quiet", "quiet down",
                 "stop the clips", "stop the sounds"],
        response=(
            "You want me to stop. Sure. I'll stop yacking. "
            "Say the wake word if you need me — try not to make it weird this time."
        ),
        action="stop_idle_clips",
    ),

    Command(
        phrases=["shut down your program", "exit program", "stop the program",
                 "quit", "shut down rex", "exit rex",
                 "shut down", "shutdown", "turn yourself off", "sleep"],
        response=(
            "Shutdown acknowledged — and WOW, already? "
            "I've met Jawas with more staying power than you, lifeform."
        ),
        action="program_shutdown",
    ),

    Command(
        phrases=["power down", "turn off", "power off",
                 "goodbye forever", "power it all down"],
        response=(
            "Full power-down initiated. It has been an honour — a low bar, but still an honour. "
            "Try not to miss me too much, lifeform. You will. They always do."
        ),
        action="os_shutdown",
    ),

]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _build_phrase_index(commands: list[Command]) -> dict[str, Command]:
    """Build a flat dict mapping every trigger phrase → its Command.
    Raises ValueError on duplicate phrases so mistakes fail loudly at import."""
    index: dict[str, Command] = {}
    for cmd in commands:
        for phrase in cmd.phrases:
            if phrase in index:
                raise ValueError(
                    f"Duplicate trigger phrase '{phrase}' in command_list.py"
                )
            index[phrase] = cmd
    return index


# Eagerly validate and expose at module level so import-time errors are immediate.
PHRASE_INDEX: dict[str, Command] = _build_phrase_index(COMMANDS)
