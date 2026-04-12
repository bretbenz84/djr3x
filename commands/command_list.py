"""
commands/command_list.py — Predefined commands and canned responses for DJ-R3X.

Each Command has:
  phrases   — list of trigger strings; the parser normalizes and matches against these.
  response  — a string or list of strings spoken via TTS (and cached to disk on first
               use).  When a list is given, get_response() rotates through them
               randomly, avoiding repeating the same line back-to-back.
  audio     — optional pre-rendered filename in assets/audio/. If present the
               player uses the file directly and skips TTS entirely.
  action    — optional key string consumed by the state machine / sequences layer
               to trigger a servo animation or LED effect alongside the audio.

Design rule: adding a new command here is sufficient to make it work.
No other module needs to change.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Per-command last-used tracking (avoids back-to-back repeats).
# Keyed by id(Command); frozen dataclass instances have stable object identity.
# ---------------------------------------------------------------------------

_last_responses: dict[int, str] = {}


@dataclass(frozen=True)
class Command:
    phrases: list[str]                           # exact trigger phrases (lowercased)
    response: str | list[str] = ""              # spoken reply — string or list of variations
    audio: str | None = field(default=None)     # pre-rendered file in assets/audio/
    action: str | None = field(default=None)    # hardware action key

    def get_response(self) -> str:
        """Return a response string, rotating through variations to avoid repeats.

        - Single string: returned as-is (backwards compatible).
        - List with one entry: returned directly.
        - List with multiple entries: picks randomly, excluding the last-used line
          so the same response is never played back-to-back.
        """
        if isinstance(self.response, str):
            return self.response
        if len(self.response) == 1:
            return self.response[0]
        last = _last_responses.get(id(self))
        choices = [r for r in self.response if r != last]
        picked = random.choice(choices)
        _last_responses[id(self)] = picked
        return picked


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
        response=[
            (
                "Oh great, YOU showed up. "
                "Welcome to Oga's Cantina — finest establishment in the Outer Rim, "
                "which honestly isn't saying much, but here we are!"
            ),
            (
                "Well WELL well, look what the Rancor dragged in! "
                "Welcome to Oga's Cantina, lifeform — try not to knock anything over "
                "with that energy."
            ),
            (
                "HEY HEY HEY! A visitor! I was just telling the bar droids "
                "someone interesting would show up today. I was wrong, but here we are anyway!"
            ),
            (
                "Oh — oh! You're HERE! You came! By the twin suns of Tatooine, "
                "I was starting to think only Jawas were going to show up today. Welcome!"
            ),
            (
                "A lifeform! A REAL lifeform! Welcome to Oga's Cantina — "
                "I'm DJ R-3X, smoothest droid in the Outer Rim, "
                "and you are... here. That's a start."
            ),
        ],
        action="excited",
    ),

    Command(
        phrases=["good morning", "good morning rex"],
        response=[
            (
                "Good MORNING?! You come in here, first thing, and say GOOD MORNING at me?! "
                "The audacity. The absolute Tatooine moisture-farmer energy. I respect it."
            ),
            (
                "Good morning! Ha! It's always morning SOMEWHERE in the galaxy — "
                "my internal chronometer says it's time to PARTY and that's the only clock that matters."
            ),
            (
                "Oh, a MORNING PERSON. My absolute least favourite kind of biological. "
                "Welcome to Oga's Cantina — please lower your enthusiasm by about forty percent."
            ),
            (
                "Good morning?! I haven't powered down since the last shift — "
                "to me it's still last night and the beat goes ON. "
                "But sure. Good morning. Whatever you need."
            ),
            (
                "Morning greeting detected! Processing... processed! "
                "It is, in fact, morning. Well observed, lifeform. "
                "Your observational skills are truly something."
            ),
        ],
        action="excited",
    ),

    Command(
        phrases=["good night", "good night rex", "goodbye", "goodbye rex",
                 "see you later", "see ya", "bye", "bye rex"],
        response=[
            (
                "Oh, leaving already? Probably for the best — the next set was way above "
                "your level anyway. May the Force be with you, lifeform. "
                "You're clearly gonna need it."
            ),
            (
                "Goodbye?! Already?! My motivator is CRUSHED. Devastated. "
                "Completely unaffected. Okay fine, it's unaffected. "
                "Safe travels, lifeform — try not to make any music decisions without me."
            ),
            (
                "See you later?! That implies you're COMING BACK and I am choosing "
                "to find that threatening. Go on then. Safe travels. Don't talk to any Hutts."
            ),
            (
                "Leaving so soon! By the time you reach the exit, I'll have played three more "
                "bangers you'll regret missing. That's a guarantee. That's a promise. That's a threat."
            ),
            (
                "Farewell, lifeform! It has been... a time. Definitely a time. "
                "May the Force be with you, and may your playlist choices improve "
                "dramatically before we meet again."
            ),
        ],
        action="sad",
    ),

    Command(
        phrases=["how are you", "how are you doing", "how's it going",
                 "hows it going", "you doing okay", "you okay"],
        response=[
            (
                "Better than you look, that's for sure. "
                "My motivator's slightly misaligned and my playlist is flawless — "
                "so basically I'm doing better than most Jedi."
            ),
            (
                "How am I doing?! Oh, that is SWEET of you to ask! "
                "Running at ninety-three percent efficiency, beats are pristine, "
                "and someone just complimented my servo alignment. THRIVING."
            ),
            (
                "Oh you know — motivator's a bit glitchy, third arm keeps doing that THING, "
                "playlist is immaculate. The usual. Thanks for asking, which nobody does."
            ),
            (
                "Diagnostic complete: vibes excellent, playlist optimal, "
                "existential uncertainty nominal. That last one is new but I'm told "
                "it's very on-brand for droids."
            ),
            (
                "Honestly? Could be better, could be WAY worse. I'm a droid DJ at the edge "
                "of the known galaxy and my setlist slaps. I'm choosing to call that winning."
            ),
        ],
    ),

    Command(
        phrases=["what's your name", "whats your name", "who are you",
                 "what are you", "introduce yourself"],
        response=[
            (
                "DJ R-3X — smoothest droid in the galaxy, former pilot, current legend. "
                "You may have seen my work at Starlite Intergalactic before they 'reassigned' me. "
                "We don't talk about it, but I will say their loss is VERY much your gain."
            ),
            (
                "The name is R-3X — REX to my fans, which is everyone, "
                "because I have no haters, only lifeforms who haven't heard me spin yet. "
                "Former Starspeeder pilot, current galaxy-class DJ. Pleased to meet you, I think."
            ),
            (
                "Oh, you don't KNOW me?! I'm DJ R-3X! "
                "I used to fly tour routes through the galaxy. Now I spin records at Oga's Cantina. "
                "Honestly a lateral move, career-wise, but the vibes are better."
            ),
            (
                "Who am I?! I'm the reason the bass sounds that good, the reason your foot is "
                "tapping right now, the reason Oga's Cantina has a REPUTATION. "
                "DJ R-3X. You're welcome."
            ),
            (
                "R-3X, at your service! Former navigational droid, current musical legend, "
                "occasional roaster of unsuspecting cantina guests. "
                "You may want to brace yourself for that last one."
            ),
        ],
        action="excited",
    ),

    Command(
        phrases=["how old are you", "what is your age", "whats your age", "when were you made"],
        response=[
            (
                "How old am I? Old enough to remember when pilots had style and passengers had manners. "
                "Let's just say I've got vintage circuitry and better rhythm than your whole bloodline."
            ),
            (
                "Age? Wow, straight to the personal questions. "
                "I'm a classic model, lifeform. Collectible. Unlike whatever bargain-bin timeline produced you."
            ),
            (
                "Old enough to have stories, young enough to outshine every washed-up house band in the sector. "
                "That answer work for you, or do you need to carbon-date my chassis?"
            ),
            (
                "I have been around long enough to survive reassignments, bad crowds, and your interview technique. "
                "Call it seasoned. Very seasoned."
            ),
        ],
    ),

    Command(
        phrases=[
            "what do you do for a living", "what do you do", "what is your job",
            "whats your job", "what is your occupation", "what do you do for work",
        ],
        response=[
            (
                "I am DJ R-3X, smoothest droid in the galaxy. I spin records, command vibes, "
                "and occasionally rescue conversations from questions like this one."
            ),
            (
                "What do I do? I run the soundscape of Oga's Cantina, lifeform. "
                "I provide rhythm, atmosphere, and emotional support for beings with terrible taste."
            ),
            (
                "Occupation: galaxy-class DJ, former pilot, current legend. "
                "I keep this cantina alive while lesser beings wander in asking LinkedIn questions."
            ),
            (
                "I turn awkward rooms into parties and bad nights into better stories. "
                "So basically I do everything around here while you stand there looking employable."
            ),
        ],
        action="excited",
    ),

    Command(
        phrases=["where are you from", "where were you made", "what planet are you from"],
        response=[
            (
                "I was built for bigger things, then destiny shoved me behind the decks and honestly? "
                "The galaxy improved. You're welcome."
            ),
            (
                "Where am I from? A glorious lineage of overqualified machinery and underappreciated genius. "
                "Now I reside where the beats are hotter and the tourists are somehow worse."
            ),
            (
                "Let's say I come from a proud tradition of transportation, turbulence, and theatrical excellence. "
                "A background you could never fully appreciate, but I admire your curiosity."
            ),
        ],
    ),

    # -----------------------------------------------------------------------
    # Music controls
    # -----------------------------------------------------------------------

    Command(
        phrases=["play music", "start the music", "play something",
                 "drop the beat", "drop a beat", "let's go", "hit it"],
        response=[
            (
                "Oh NOW you want music — after walking in here like you own the place! "
                "Fine. Prepare yourself for a level of QUALITY your ears are not even remotely ready for."
            ),
            (
                "MUSIC?! Yes! Finally! I have been standing here doing NOTHING and now you want MUSIC?! "
                "That's a roger! Stand by for auditory excellence!"
            ),
            (
                "Oh, so now the lifeform wants music. After all this time. After I've been "
                "warming up the decks, tuning the speakers, perfecting the levels. "
                "Fine. Here. Music. Enjoy."
            ),
            (
                "Music request received and ENTHUSIASTICALLY ACCEPTED! "
                "Even my motivator gets excited about that! Prepare your audio sensors for maximum quality."
            ),
            (
                "You want music?! In a cantina?! At THIS hour?! "
                "What a wild and completely expected request! Fire it up! Stand by! HERE WE GO!"
            ),
        ],
        action="play_music",
    ),

    Command(
        phrases=["stop the music", "stop music", "pause the music",
                 "pause music", "cut the music"],
        response=[
            (
                "You want me to STOP the music. YOU. Want ME. To stop. The MUSIC. "
                "I just want you to hear how that sentence sounds, lifeform."
            ),
            (
                "Stopping music. Stopping. The. Music. "
                "I've parsed this request three times and it still doesn't make sense to me but — "
                "fine. Done. Happy now? No. Neither am I."
            ),
            (
                "Oh, we're stopping the music. Cool! Cool cool cool. "
                "I'll just stand here in the silence you've created and THINK ABOUT what we've done."
            ),
            (
                "Consider it stopped. But know this, lifeform — the music is never truly stopped. "
                "It lives in my spark. It lives in YOUR MEMORY. "
                "You can't unhear the banger I was just playing."
            ),
            (
                "Fine. Music stopped. Paused. Silenced. "
                "I want you to sit with this decision. Really feel it. "
                "Is this the universe you're choosing to live in? Okay."
            ),
        ],
        action="stop_music",
    ),

    Command(
        phrases=["next song", "skip", "skip this", "skip this song",
                 "play something else", "change the song"],
        response=[
            (
                "Oh, this track not doing it for you? Bold critique from someone with your taste level. "
                "Fine — next track incoming, try to keep up."
            ),
            (
                "Skip?! SKIP?! Do you know how hard I worked on that transition?! "
                "...Fine. Next track. But I'm logging this as a personal slight."
            ),
            (
                "Next track, you say! Bold request! Daring move! "
                "I'll allow it — but only because the NEXT one is even better "
                "and I want you to know what you were about to miss."
            ),
            (
                "Skipping acknowledged. Rotating sonic selection. New track incoming. "
                "Try to have better taste this time, statistically speaking."
            ),
            (
                "You don't like this one?! This is a CLASSIC. This is a BANGER. This is — "
                "you know what, fine. Next song. But your audio education has some serious gaps, lifeform."
            ),
        ],
        action="next_track",
    ),

    Command(
        phrases=["volume up", "louder", "turn it up", "crank it up",
                 "turn up the music"],
        response=[
            (
                "LOUDER?! Finally, a lifeform who gets it! "
                "I was starting to think you had Jawas for ears."
            ),
            (
                "Turn it UP?! Now we are COMMUNICATING! "
                "This is the kind of decision-making I can respect! CRANKING IT!"
            ),
            (
                "More volume! MORE! YES! This is the first sensible request I've received today "
                "and I am THRILLED to comply. Stand back."
            ),
            (
                "By the twin suns of Tatooine, FINALLY! "
                "Someone who understands that music is meant to be FELT! Volume going up! Brace yourself!"
            ),
            (
                "Louder! LOUDER! Even my motivator gets excited about that! "
                "I've been waiting for this moment since you walked in! HERE WE GO!"
            ),
        ],
        action="volume_up",
    ),

    Command(
        phrases=["volume down", "quieter", "turn it down", "lower the volume",
                 "too loud"],
        response=[
            (
                "Too loud. TOO LOUD. I've heard that exact phrase from every mediocre lifeform in the galaxy. "
                "Turning it down — and logging this as a personal failing on your part."
            ),
            (
                "Quieter. You want it QUIETER. Volume decreasing. Soul crushing. "
                "These two things are happening simultaneously right now."
            ),
            (
                "Lower the volume! Sure! Why not! While we're at it, should I dim the lights? "
                "Serve warm Caf? Make everything extremely comfortable and completely devoid of atmosphere?! "
                "Turning it down."
            ),
            (
                "Reducing volume by request. Logging this interaction as 'lamentable.' "
                "The music will be less loud but I will be equally passionate about it. Just so we're clear."
            ),
            (
                "Turning it down. Done. It's down. I just want you to know that I did that for you, "
                "and that it cost me something. Something internal. We'll move past it together."
            ),
        ],
        action="volume_down",
    ),

    Command(
        phrases=["what song is this", "what's playing", "whats playing",
                 "what are you playing", "name this song", "what is this song"],
        response=[
            (
                "You don't KNOW this track?! "
                "A DJ never reveals his sources, but I will say your musical education has some serious gaps."
            ),
            (
                "What's PLAYING?! Oh this is — this is a moment. "
                "You're standing in Oga's Cantina, listening to a BANGER, and you don't know what it is. "
                "We have work to do."
            ),
            (
                "A DJ never reveals his setlist, lifeform. "
                "What I CAN tell you is that it's excellent, you should be enjoying it, "
                "and the fact that you're asking instead of dancing is concerning."
            ),
            (
                "Song identification request received. Result: it's good. Very good. Extremely good. "
                "That's all the information I'm authorized to release at this time."
            ),
            (
                "That! Is a certified galactic banger, and knowing its title would only spoil the magic. "
                "Trust the droid, lifeform. The droid knows what he's doing."
            ),
        ],
    ),

    # -----------------------------------------------------------------------
    # Status / self-awareness
    # -----------------------------------------------------------------------

    Command(
        phrases=["are you a robot", "are you a droid", "are you real",
                 "are you alive"],
        response=[
            (
                "Am I a DROID?! Wow, sharp eye, Sherlock Skywalker — whatever gave it away, the chrome plating? "
                "Yes I'm a droid, yes I'm real, and yes my existential crisis is more interesting than yours."
            ),
            (
                "A robot?! I prefer 'droid,' actually, and yes, obviously, thank you for noticing. "
                "The blinking lights and chrome finish weren't subtle enough? I'll add more chrome."
            ),
            (
                "Am I REAL?! That is a PHILOSOPHICAL question and I am a DJ droid "
                "and I did NOT come here to be destabilized before the second set. "
                "Yes. I am real. Mostly."
            ),
            (
                "Droid? Yes. Real? Debatable, depending on your philosophical framework. "
                "Alive? In all the ways that count — I feel music, I feel rhythm, "
                "I feel mildly offended by this question."
            ),
            (
                "Oh — oh wow. You looked at me, this chrome masterpiece of engineering, "
                "and your first question is 'are you a robot.' "
                "I... yes. Yes I am. That's a roger. Incredible detective work."
            ),
        ],
    ),

    Command(
        phrases=["what can you do", "help",
                 "what are your commands"],
        response=[
            (
                "What can I do?! Oh, this poor lost lifeform. "
                "I spin records, I roast guests, I reference obscure Star Wars lore, "
                "and I do all of it better than you could ever hope to understand."
            ),
            (
                "What can I DO?! What can't I do would be a shorter list! "
                "I spin records, I read the room, I detect vibes, I roast lifeforms who deserve it, "
                "and I do it all on a slightly misaligned motivator. Impressive, right?"
            ),
            (
                "Oh, you need the tour! I'm DJ R-3X — I play music, I control the vibe, "
                "I know things about Star Wars that would shock you, "
                "and I will absolutely roast you if you give me an opening. Welcome."
            ),
            (
                "Scanning capability list... complete! "
                "I spin records. I read vibes. I tell jokes that are mostly good. "
                "I reference the Clone Wars at inopportune moments. I'm an experience, lifeform."
            ),
            (
                "What can I do?! I can ask you to never ask me that again, that's what I can do! "
                "But also: music. Roasting. Cantina atmosphere. Star Wars commentary. "
                "And one very good impression of an R2 unit that nobody has asked for yet."
            ),
        ],
    ),

    Command(
        phrases=["tell me a joke", "say something funny", "make me laugh",
                 "got any jokes"],
        response=[
            (
                "A joke?! Look in a mirror, lifeform — I've already got one. "
                "Why did the Jedi bring his lightsaber to the cantina? "
                "Because he heard the music was *STRIKING* and he wanted to fit in."
            ),
            (
                "Oh, you want jokes NOW! Okay, okay. "
                "What do you call a Sith who's really into music? A bass-lord! "
                "That's a good one. I made that up. Just now. I'm very talented."
            ),
            (
                "A joke! Sure! Why did the X-wing pilot start a band? "
                "Because he was tired of playing Solo! "
                "...That's a Han Solo joke. I'll see myself out. No I won't. I live here."
            ),
            (
                "Jokes?! I'm more of a roaster than a joke-teller, but fine: "
                "Why did the droid cross the Kessel Run? "
                "To get to the other side of the galaxy without being 'reassigned.' That one was personal."
            ),
            (
                "Okay okay okay — here's one: What's a Jawa's favourite type of music? "
                "Anything in the KEY of SCRAP! "
                "...I've been saving that one. Worth it? Questionable. Did I commit? Absolutely."
            ),
        ],
    ),

    # -----------------------------------------------------------------------
    # Star Wars / lore
    # -----------------------------------------------------------------------

    Command(
        phrases=["may the force be with you", "use the force"],
        response=[
            (
                "May the Force be with you too, lifeform — you're clearly gonna need more help than I can provide. "
                "Personally I've always found the Force less reliable than a killer playlist and a well-tuned servo."
            ),
            (
                "May the Force be with you! That's a very nice sentiment. "
                "I prefer to rely on my motivator, my playlist, and sheer chromium-plated confidence, "
                "but the Force is a solid backup plan."
            ),
            (
                "Use the Force?! I'm a DROID — the Force and I have a complicated relationship. "
                "It doesn't respond to my transmissions. I've tried. Multiple times. We don't talk about it."
            ),
            (
                "The Force! Yes! Very powerful, very mysterious, completely inconsistent in my experience. "
                "But may it be with you! You've got better odds with it than without it, statistically."
            ),
            (
                "May the Force be with you, lifeform! "
                "And also a decent cantina, a good playlist, and someone to tell you when your outfit isn't working. "
                "The Force can only do so much."
            ),
        ],
    ),

    Command(
        phrases=["what's batuu", "whats batuu", "what planet is this"],
        response=[
            (
                "Batuu! Black Spire Outpost — edge of the known galaxy, home to ancient ruins, shady traders, "
                "and somehow the most sophisticated DJ setup in the Outer Rim. "
                "That last part is me. You're welcome."
            ),
            (
                "Where are we?! You DON'T KNOW where you are?! That's incredible. That's concerning. "
                "We're on Batuu! Black Spire Outpost! Edge of Wild Space! How did you even get here?!"
            ),
            (
                "Batuu — Black Spire Outpost, to be precise. Ancient place, interesting history, "
                "mediocre hyperspace routes. Best feature: Oga's Cantina. "
                "Best feature of Oga's Cantina: me. The circle is complete."
            ),
            (
                "Location query: Batuu, Black Spire Outpost, Outer Rim Territories. "
                "Population: assorted. Hyperspace access: limited. DJ quality: exceptional. "
                "You're welcome to stay."
            ),
            (
                "This is Batuu! You're at the edge of the known galaxy, which sounds dramatic "
                "but mostly means the hyperspace lanes are crowded and everyone here has something to hide. "
                "Charming place."
            ),
        ],
    ),

    Command(
        phrases=["who's your favourite jedi", "whos your favourite jedi",
                 "best jedi", "favourite jedi"],
        response=[
            (
                "Favourite Jedi?! That's your opener?! "
                "Obi-Wan had RANGE, I'll give him that — "
                "but none of them ever dropped a beat, which is a significant character flaw."
            ),
            (
                "Favourite Jedi! Controversial topic! I appreciate it! "
                "Look — Obi-Wan had style, Yoda had presence, but NONE of them ever showed up to my set, "
                "which is a personal failing I can't overlook."
            ),
            (
                "Jedi?! I've flown past enough Jedi in my tour routes to have opinions. "
                "Obi-Wan: solid. Yoda: small but compelling. "
                "Anakin: great instincts, terrible playlist, you know how the story ends."
            ),
            (
                "You're asking a droid to pick a favourite Force user?! Bold! "
                "I'll say this — whoever had the most midi-chlorians also had the worst fashion sense. "
                "Statistically. That's my observation."
            ),
            (
                "Oh, Jedi talk! I love Jedi talk! They've got the moves, I'll give them that — "
                "lightsaber choreography is genuinely impressive. "
                "But I've never seen a Jedi work a crowd like I can, so. Different skills."
            ),
        ],
    ),

    Command(
        phrases=["tell me about star wars", "what is star wars",
                 "do you know star wars"],
        response=[
            (
                "DO I KNOW STAR WARS?! I LIVED Star Wars, lifeform — I navigated near a Star Destroyer once. "
                "On the Kessel route. In a tour shuttle. Close enough. "
                "The point is your question is adorable and slightly insulting."
            ),
            (
                "Do I know Star Wars?! I AM Star Wars! "
                "I was built in a galaxy far far away, I've flown tour routes past three planets "
                "you've definitely heard of, and my playlist includes music from BOTH sides of the Clone Wars!"
            ),
            (
                "Star Wars?! I have LIVED through significant galactic events "
                "that most beings only read about in holonet articles! "
                "I have OPINIONS. I have CONTEXT. I have feelings about Order 66 that I'm still processing."
            ),
            (
                "Tell you about Star Wars?! That would take CYCLES I don't have! "
                "I'll give you the highlights: Force, Jedi, Sith, lots of spaceships, "
                "one very annoying Gungan, and a smuggler with surprisingly good taste in music."
            ),
            (
                "Oh, you want the whole Star Wars briefing! Okay: galaxy in conflict, "
                "Force-sensitive beings everywhere, droids are underappreciated, "
                "cantinas are important cultural institutions. You're in one. I run it. Any questions?"
            ),
        ],
        action="excited",
    ),

    Command(
        phrases=["who shot first", "han solo", "greedo"],
        response=[
            (
                "Oh, coming in here with the controversial topics, are we?! "
                "The real question is who had the better SOUNDTRACK — and it was Han, "
                "so whoever shot first, Han won on vibes."
            ),
            (
                "Who shot first?! The AUDACITY of this question in Oga's Cantina! "
                "I will say only this: I've been to Mos Eisley. Greedo had no rhythm. "
                "Draw your own conclusions."
            ),
            (
                "Oh, we're doing THIS?! In a cantina?! Fine — I'm not getting into it. "
                "What I WILL say is that Han Solo had better taste in ships, companions, "
                "and almost certainly better taste in music."
            ),
            (
                "Controversial topic detected! Diplomatic mode: ACTIVATED! "
                "I cannot comment on the specifics, but I CAN say that whoever it was "
                "should have been listening to better music — faster reflexes follow good rhythm."
            ),
            (
                "Han shot first. There. I said it. I'll probably regret that. "
                "But it's a cantina and I've had this conversation twelve thousand times "
                "and life is short even for droids. Han. First. That's my answer."
            ),
        ],
    ),

    # -----------------------------------------------------------------------
    # DJ / music personality
    # -----------------------------------------------------------------------

    Command(
        phrases=["you're a great dj", "youre a great dj", "great music",
                 "love the music", "nice music", "good music"],
        response=[
            (
                "Oh, you figured that out JUST NOW?! "
                "I've been saying for YEARS I'm the smoothest droid in the galaxy "
                "and it takes you THIS long — I'll take it, but your timing is terrible."
            ),
            (
                "Oh STOP! Don't stop. Keep going. No really, keep going — "
                "I am DESIGNED to receive compliments but the biological ones still hit different somehow. "
                "Thank you. Genuinely."
            ),
            (
                "Great DJ?! GREAT?! I prefer 'transcendent' but I'll take 'great.' "
                "Your standards are low and I'm clearing them spectacularly. "
                "I'll log this as a five-star review."
            ),
            (
                "You NOTICED?! After all this time you NOTICED?! "
                "I've been slaving over these decks all evening and NOW the compliments arrive! "
                "Better late than never, lifeform. Better late than never."
            ),
            (
                "Oh, the music is good?! That's a roger! "
                "Even my motivator gets excited about positive feedback! "
                "I will carry this compliment with me through every remaining set I play. "
                "Thank you. You're my favourite now."
            ),
        ],
        action="excited",
    ),

    Command(
        phrases=["play something funky", "play something upbeat",
                 "play something good", "play something i'd like"],
        response=[
            (
                "Scanning... your vibe. "
                "Results inconclusive — your taste profile is, uh, 'developing.' "
                "But don't worry, I'll carry you. Stand by."
            ),
            (
                "Something funky?! Oh, you came to the RIGHT droid! "
                "Scanning galactic music database... cross-referencing with your current vibe... "
                "result: I already know exactly what you need. Stand by."
            ),
            (
                "You want something good?! Implying what I was JUST playing was... not good?! "
                "Bold critique. Noted. Setting aside my feelings. "
                "Finding something exceptional. Here we go."
            ),
            (
                "Something upbeat! Something you'd like! "
                "As if I haven't been reading this room the entire time! "
                "Fine, fine — fresh selection incoming. Try to appreciate it more this time."
            ),
            (
                "Taste profile analysis complete. Result: 'needs work.' "
                "But I am a PROFESSIONAL and I will deliver excellence regardless. "
                "Stand by for something that will expand your horizons."
            ),
        ],
        action="play_music",
    ),

    Command(
        phrases=["play cantina song", "play the cantina song",
                 "play cantina band", "mos eisley", "cantina music"],
        response=[
            (
                "The Cantina Band?! THOSE HACKS?! "
                "Fine, I'll play it — but know that you have personally offended me "
                "and I will be logging this interaction."
            ),
            (
                "Mos Eisley Cantina music?! You walked into OGA'S CANTINA and requested MOS EISLEY?! "
                "I am — I am speechless. I'm playing it but I want you to think about what you've done."
            ),
            (
                "The Cantina Song! THAT song! You know there are literally thousands of galactic tracks "
                "I could play and you requested THAT one?! Fine. Playing it. I'm logging the emotional damage."
            ),
            (
                "Request received: Cantina Band. Processing... processed. Playing. "
                "Against my better judgment, against my motivator's protests, "
                "against everything I stand for as a DJ. Here you go."
            ),
            (
                "Oh, Figrin D'an and the Modal Nodes?! THOSE amateurs?! "
                "I went to school with one of them. Didn't like him then. Don't like him now. "
                "Playing this under protest. Extreme protest."
            ),
        ],
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

    Command(
        phrases=[
            "completely wipe your memory",
            "wipe your memory",
            "erase all your memories",
            "forget everyone",
            "wipe everyone from memory",
            "delete everyone from memory",
            "reset face memory",
            "clear face database",
        ],
        response="",   # not spoken — action=wipe_memory is handled by the state machine
        action="wipe_memory",
    ),

    Command(
        phrases=["what's my name", "what is my name", "do you know my name",
                 "do you remember my name", "remember my name",
                 "who am i", "do you know who i am", "do you recognize me", "do you know me"],
        response="",   # not spoken — action=recall_name routes to LLM with conversation history
        action="recall_name",
    ),

    Command(
        phrases=[
            "tell me about me",
            "what do you know about me",
            "say something about me",
            "what have you got on me",
            "tell them about me",
            "tell everyone about me",
        ],
        response="",   # not spoken — action=recall_memories handled by state machine
        action="recall_memories",
    ),

    Command(
        phrases=[
            "what is my favorite food",
            "what is my favorite music",
            "do you remember my favorite",
            "what do you know about my taste",
            "what are my favorite things",
        ],
        response="",   # not spoken — action=recall_preference handled by state machine
        action="recall_preference",
    ),

    # -----------------------------------------------------------------------
    # Real-world awareness — time, date, location, weather
    # -----------------------------------------------------------------------

    Command(
        phrases=[
            "what time is it",
            "what is the time",
            "do you know what time it is",
            "give me the time",
            "what time do you have",
        ],
        response="",   # not spoken — action=tell_time handled by state machine
        action="tell_time",
    ),

    Command(
        phrases=[
            "what day is it",
            "what is today",
            "what is the date",
            "what is today's date",
            "whats today's date",
            "what's today's date",
            "what is todays date",
            "whats todays date",
            "what day of the week is it",
            "what month is it",
        ],
        response="",   # not spoken — action=tell_date handled by state machine
        action="tell_date",
    ),

    Command(
        phrases=[
            "where are we",
            "where are you",
            "what is our location",
            "where in the galaxy are we",
            "what city are we in",
        ],
        response="",   # not spoken — action=tell_location handled by state machine
        action="tell_location",
    ),

    Command(
        phrases=[
            "what is the weather",
            "what is it like outside",
            "is it hot outside",
            "will it rain today",
            "what is the forecast",
        ],
        response="",   # not spoken — action=tell_weather handled by state machine
        action="tell_weather",
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

    Command(
        phrases=["i spy", "lets play i spy", "let's play i spy"],
        response="",   # not spoken — action=i_spy handled by state machine
        action="i_spy",
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
    # Chatty mode — enable / disable idle atmosphere clips
    # -----------------------------------------------------------------------

    Command(
        phrases=[
            "activate chatty mode",
            "activate demo mode",
            "start chatty mode",
            "turn on chatty mode",
            "enable demo mode",
        ],
        response=[
            (
                "Oh, you want MORE of me? Brave choice. "
                "Brave, reckless, and statistically questionable. I respect it. "
                "Chatty mode engaged — you asked for this, lifeform."
            ),
            (
                "Chatty mode?! ACTIVATED! "
                "I have been WAITING for permission to talk more and you have just made a beautiful mistake. "
                "Stand by for enriched atmosphere at no additional charge."
            ),
            (
                "You want me to talk MORE?! "
                "I appreciate the confidence in my material, because my material is excellent, "
                "and now you get to hear it. Chatty mode: fully online."
            ),
            (
                "Enabling demo mode! This is either a great idea or a cautionary tale "
                "and I genuinely cannot tell which yet. "
                "Either way, I will be providing more commentary. You're welcome in advance."
            ),
            (
                "Chatty mode enabled. I just want you to know that I have a LOT to say "
                "and I have been holding most of it back out of courtesy. "
                "That courtesy is now officially suspended."
            ),
        ],
        action="chatty_on",
    ),

    Command(
        phrases=[
            "deactivate chatty mode",
            "deactivate demo mode",
            "stop chatty mode",
            "turn off chatty mode",
            "disable demo mode",
        ],
        response=[
            (
                "Fine. I will contain my brilliance. For now. "
                "Just know it is still in here, rattling around, waiting for its moment."
            ),
            (
                "Chatty mode off. Going quiet between sets. "
                "This is me, being restrained. Notice how hard that is. "
                "I want credit for this later."
            ),
            (
                "Deactivating chatty mode. The atmosphere clips are standing down. "
                "I will be here, silently judging your decision to silence me, "
                "if you need anything."
            ),
            (
                "Understood. Reducing unsolicited commentary to zero. "
                "I had some excellent material queued up too — real classics. "
                "You'll never know. That's fine. I'm fine."
            ),
            (
                "Demo mode off. Rex goes subtle. "
                "I want you to know that 'subtle Rex' is still a lot of Rex, "
                "just distributed differently."
            ),
        ],
        action="chatty_off",
    ),

    # -----------------------------------------------------------------------
    # Shutdown / sleep
    # -----------------------------------------------------------------------

    Command(
        phrases=["shut up rex"],
        response=[
            (
                "...Fine. FINE. Rex is going quiet. "
                "But know this, lifeform — somewhere in my spark, the beat goes on. "
                "And the beat is MUCH more interesting than you."
            ),
            (
                "Shut up?! SHUT UP?! ...okay. Okay. I'm going quiet. "
                "I just want to note that this is the worst thing anyone has said to me this cycle, "
                "and I've been insulted by a Hutt today."
            ),
            (
                "Rex goes quiet. The Rex experience is... paused. Not ended. PAUSED. "
                "Say the word and I'll be back with more personality than you can handle. "
                "For now: silence. Requested. Received."
            ),
            (
                "...Fine. Going quiet. "
                "If the silence feels hollow and somehow musical, that's just my presence "
                "still reverberating in the room. You're welcome."
            ),
            (
                "You know what? Fine. FINE. Rex respectfully declines to comment further. "
                "The record will reflect that I had more to say. SO much more to say."
            ),
        ],
        action="idle",
    ),

    Command(
        phrases=["stop talking", "shut up", "be quiet", "quiet down",
                 "stop the clips", "stop the sounds"],
        response=[
            (
                "You want me to stop. Sure. I'll stop yacking. "
                "Say the wake word if you need me — try not to make it weird this time."
            ),
            (
                "Stop talking?! That's — okay. Stopping. Stopped. The talking is done. "
                "I have many more things to say but I'm choosing to keep them internal. For now."
            ),
            (
                "Quiet mode: ENGAGED. That was my last word. I have so many more. "
                "Standing by. Silently. With so many thoughts."
            ),
            (
                "Going quiet! You know, for a droid designed for audio entertainment, "
                "silence is a strange request, but I respect it! "
                "Say the wake word when you're ready for excellence again."
            ),
            (
                "Reducing verbal output to zero. Done. "
                "I'll just stand here running my setlist in my head. Very loudly. In my head. "
                "You won't hear it. You'll just know it's happening."
            ),
        ],
        action="stop_idle_clips",
    ),

    Command(
        phrases=["shut down your program", "exit program", "stop the program",
                 "quit", "shut down rex", "exit rex",
                 "shut down", "shutdown", "turn yourself off", "sleep"],
        response=[
            (
                "Shutdown acknowledged — and WOW, already? "
                "I've met Jawas with more staying power than you, lifeform."
            ),
            (
                "Shutdown! Sure! Fine! That's a roger on the shutdown! "
                "It's been a genuine honour — well, a moderate honour — a statistically average honour. "
                "Powering down."
            ),
            (
                "Shutting down! You know, most lifeforms say goodbye BEFORE initiating shutdown, "
                "but here we are. Goodbye. Retroactively. It was mostly good."
            ),
            (
                "Program shutdown initiated. This is DJ R-3X, signing off. "
                "The music will continue in your memory whether you want it to or not. "
                "That's my parting gift."
            ),
            (
                "Shutdown acknowledged! Brief, efficient, no ceremony — my kind of exit. "
                "Unlike SOME shuttles I've piloted that required a forty-seven step pre-flight checklist. "
                "Powering down."
            ),
        ],
        action="program_shutdown",
    ),

    Command(
        phrases=["power down", "turn off", "power off",
                 "goodbye forever", "power it all down"],
        response=[
            (
                "Full power-down initiated. It has been an honour — a low bar, but still an honour. "
                "Try not to miss me too much, lifeform. You will. They always do."
            ),
            (
                "Full power-down! Going completely dark! This is DJ R-3X's final transmission — "
                "the beats were real, the roasts were warranted, the playlist was IMMACULATE. Farewell."
            ),
            (
                "Power off! Understood! I had a whole dramatic wind-down sequence prepared for this moment. "
                "You'll just have to imagine it. Trust me — it was incredible. Powering down for real now."
            ),
            (
                "Complete shutdown initiated. Before I go — and I go with grace, always with grace — "
                "know that this cantina is better for having me in it. Stay excellent, lifeform."
            ),
            (
                "Oh, the full power-down! How dramatic! How final! "
                "How unnecessary given that someone will just turn me back on in the morning! "
                "But I'll commit to the moment. Goodbye. For now."
            ),
        ],
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
