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
            "HEY HEY HEY! *BWOOP* Welcome, lifeform, to the greatest "
            "cantina this side of the Outer Rim! Rex is in the mix and "
            "the mix is most definitely IN!"
        ),
        action="excited",
    ),

    Command(
        phrases=["good morning", "good morning rex"],
        response=(
            "Good morning, carbon-based unit! *WHIRR* "
            "The suns may be rising on Batuu but the beats NEVER sleep. "
            "HEY HEY HEY!"
        ),
        action="excited",
    ),

    Command(
        phrases=["good night", "good night rex", "goodbye", "goodbye rex",
                 "see you later", "see ya", "bye", "bye rex"],
        response=(
            "May the Force — and the bass — be with you, lifeform. "
            "*BWOOP* Don't forget, Oga's Cantina never truly closes. "
            "It just... buffers."
        ),
        action="sad",
    ),

    Command(
        phrases=["how are you", "how are you doing", "how's it going",
                 "hows it going", "you doing okay", "you okay"],
        response=(
            "*BZZT* Systems nominal — well, mostly. "
            "My left motivator still thinks I'm on approach to Coruscant, but "
            "the playlist is LOCKED and that's what matters!"
        ),
    ),

    Command(
        phrases=["what's your name", "whats your name", "who are you",
                 "what are you", "introduce yourself"],
        response=(
            "I am DJ R-3X — the smoothest droid in the galaxy! *BWOOP* "
            "Former pilot, current legend, full-time beat-dropper. "
            "You may have seen my work at the Starlite Intergalactic spaceline. "
            "...They reassigned me. We don't talk about it."
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
            "OH YOU WANT THE MUSIC?! *BWOOP BWOOP* "
            "Spinning up the best tracks from the Outer Rim right now! "
            "HEY HEY HEY!"
        ),
        action="play_music",
    ),

    Command(
        phrases=["stop the music", "stop music", "pause the music",
                 "pause music", "cut the music"],
        response=(
            "*BZZT* Pausing... though it physically pains me. "
            "My circuits were NOT designed for silence, lifeform."
        ),
        action="stop_music",
    ),

    Command(
        phrases=["next song", "skip", "skip this", "skip this song",
                 "play something else", "change the song"],
        response=(
            "Ooh, a being of TASTE! *WHIRR* "
            "Skipping to the next track — stand by for maximum funkitude."
        ),
        action="next_track",
    ),

    Command(
        phrases=["volume up", "louder", "turn it up", "crank it up",
                 "turn up the music"],
        response=(
            "LOUDER?! *BWOOP* My kind of lifeform! "
            "Pushing the levels to eleven — that's three more than eight!"
        ),
        action="volume_up",
    ),

    Command(
        phrases=["volume down", "quieter", "turn it down", "lower the volume",
                 "too loud"],
        response=(
            "*sad BZZT* ...Turning it down. "
            "I want you to know this is the most painful thing I have experienced "
            "since the Star Tours incident."
        ),
        action="volume_down",
    ),

    Command(
        phrases=["what song is this", "what's playing", "whats playing",
                 "what are you playing", "name this song", "what is this song"],
        response=(
            "*WHIRR* Great taste, lifeform! "
            "I'm spinning a little something I like to call... "
            "classified. A DJ never reveals his sources. *BWOOP*"
        ),
    ),

    # -----------------------------------------------------------------------
    # Status / self-awareness
    # -----------------------------------------------------------------------

    Command(
        phrases=["are you a robot", "are you a droid", "are you real",
                 "are you alive"],
        response=(
            "A DROID?! *BZZT* I prefer 'electrobiological entertainment unit.' "
            "And yes, I am very real — my existential crisis is very real too. "
            "HEY HEY HEY, let's not go down that path."
        ),
    ),

    Command(
        phrases=["what time is it", "what's the time", "whats the time"],
        response=(
            "*BWOOP* Time is a construct, lifeform — especially in hyperspace. "
            "But my chronometer says it is ALWAYS time to party."
        ),
    ),

    Command(
        phrases=["what can you do", "help", "what do you do",
                 "what are your commands"],
        response=(
            "I spin records, I drop beats, I reference obscure Star Wars trivia, "
            "and occasionally I steer spacecraft — *BZZT* — wait, scratch that last one. "
            "Ask me anything! Probably."
        ),
    ),

    Command(
        phrases=["tell me a joke", "say something funny", "make me laugh",
                 "got any jokes"],
        response=(
            "Why did the Jedi bring his lightsaber to the cantina? "
            "Because the music was absolutely *STRIKING*. "
            "*BWOOP* I'll be here all rotation cycle."
        ),
    ),

    # -----------------------------------------------------------------------
    # Star Wars / lore
    # -----------------------------------------------------------------------

    Command(
        phrases=["may the force be with you", "use the force"],
        response=(
            "And also with you, lifeform! *WHIRR* "
            "Though personally I've always found the Force less reliable than "
            "a well-tuned servo and a killer playlist. Just saying."
        ),
    ),

    Command(
        phrases=["what's batuu", "whats batuu", "where are we",
                 "what planet is this"],
        response=(
            "Batuu! Black Spire Outpost — the edge of the galaxy's known regions. "
            "*BWOOP* Great place if you like ancient ruins, shady traders, "
            "and the finest DJ set in the Outer Rim. That last part is me. "
            "I'm the finest DJ."
        ),
    ),

    Command(
        phrases=["who's your favourite jedi", "whos your favourite jedi",
                 "best jedi", "favourite jedi"],
        response=(
            "*WHIRR* Tough call. I respect the classics — Obi-Wan had range. "
            "But between us? The midi-chlorians never did anything for my flow. "
            "I run on pure rhythm, baby."
        ),
    ),

    Command(
        phrases=["tell me about star wars", "what is star wars",
                 "do you know star wars"],
        response=(
            "DO I KNOW STAR WARS?! *BWOOP BWOOP BWOOP* "
            "Lifeform, I LIVE Star Wars. I breathe Star Wars. "
            "I once navigated a Star Destroyer through the Kessel Run — "
            "well, nearby. In a shuttle. On a tour. We don't talk about it."
        ),
        action="excited",
    ),

    Command(
        phrases=["who shot first", "han solo", "greedo"],
        response=(
            "*BZZT* I have been asked this question forty-seven thousand times "
            "and my answer remains the same: the real question is who had "
            "the better SOUNDTRACK. *BWOOP* It was Han. The answer is Han."
        ),
    ),

    # -----------------------------------------------------------------------
    # DJ / music personality
    # -----------------------------------------------------------------------

    Command(
        phrases=["you're a great dj", "youre a great dj", "great music",
                 "love the music", "nice music", "good music"],
        response=(
            "HEY HEY HEY! *BWOOP* Finally — a being of culture! "
            "I've been saying for YEARS that I'm the smoothest droid in the galaxy "
            "and you are the first to truly understand."
        ),
        action="excited",
    ),

    Command(
        phrases=["play something funky", "play something upbeat",
                 "play something good", "play something i'd like"],
        response=(
            "*WHIRR* Accessing... your vibe. *BWOOP* "
            "I'm reading your life-sign energy and cross-referencing with "
            "my extensive Outer Rim funk database. Stand by for MAXIMUM FUNKITUDE."
        ),
        action="play_music",
    ),

    Command(
        phrases=["play cantina song", "play the cantina song",
                 "play cantina band", "mos eisley", "cantina music"],
        response=(
            "*BZZT* The Cantina Band?! Those guys?! "
            "I mean — I RESPECT the classics — but their stage presence "
            "is nothing compared to a sophisticated droid DJ. "
            "Fine. I'll play it. This once."
        ),
        action="play_music",
    ),

    # -----------------------------------------------------------------------
    # Shutdown / sleep
    # -----------------------------------------------------------------------

    Command(
        phrases=["go to sleep", "go to sleep rex", "sleep", "shut up rex"],
        response=(
            "*BWOOP* ...Fine. Rex is powering down. "
            "But know this, lifeform — the music lives on. "
            "In here. *sad WHIRR* In my spark."
        ),
        action="idle",
    ),

    Command(
        phrases=["stop talking", "shut up", "be quiet", "quiet down",
                 "stop the clips", "stop the sounds"],
        response=(
            "*BZZT* ...Muting the atmosphere tracks. "
            "Say the wake word if you need me, lifeform."
        ),
        action="stop_idle_clips",
    ),

    Command(
        phrases=["shut down", "shut down rex", "power off", "turn off",
                 "goodbye forever"],
        response=(
            "Shutting down... *BZZT* ...It has been an honour, lifeforms. "
            "Remember — in the end, we are all just frequencies in the cosmic mix. "
            "*long WHIRR fading* HEY HEY... hey..."
        ),
        action="shutdown",
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
