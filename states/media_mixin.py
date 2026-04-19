"""Game, music, and dance handlers extracted from the main state machine."""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING

import config
from commands.parser import parse
from vision.i_spy import build_round, guess_matches

if TYPE_CHECKING:
    from .state_machine import State

log = logging.getLogger(__name__)

_I_SPY_START_LINES: list[str] = [
    "I Spy? Ohhh, now we're playing preschool in a cantina. Fine. Try to keep up, lifeform.",
    "An I Spy round? Bold choice for someone with the visual instincts of a stormtrooper.",
    "All right, lifeform. I will pick the object, you will disappoint me with the guess. Classic format.",
    "I Spy it is. Let's see whether your eyeballs are decorative or functional.",
]

_I_SPY_CORRECT_LINES: list[str] = [
    "Well look at that, you got it. The answer was {answer}. Try not to act too proud about basic object recognition.",
    "Correct. It was {answer}. A shocking display of competence from this side of the galaxy.",
    "Yeah, yeah, you nailed it. {answer}. I hate it when the lifeforms are observant.",
]

_I_SPY_WRONG_LINES: list[str] = [
    "Nope. It was {answer}. I have seen Jawa scrap piles make better guesses.",
    "Wrong, glorious and immediate. The answer was {answer}.",
    "Not even close, lifeform. It was {answer}. You'd lose hide-and-seek to a protocol droid.",
]

_I_SPY_TIMEOUT_LINES: list[str] = [
    "Times up. I picked {answer}. Apparently suspense was doing all the heavy lifting here.",
    "No guess? Fine. It was {answer}. I cannot carry this game and your timing.",
    "You ran out of time, lifeform. The answer was {answer}. Tragic work.",
]

_I_SPY_FAIL_LINES: list[str] = [
    "I tried to play I Spy, but my scene scan went full trash compactor. We'll call that a tactical retreat.",
    "My optics just fumbled the assignment. No game this round, lifeform.",
]

_I_SPY_NUDGE_LINES: list[str] = [
    "Well? Take the guess, lifeform.",
    "Any time now. The object is not going to identify itself.",
    "Go ahead. Use those organic eyeballs.",
]


class StateMachineMediaMixin:
    """Leaf handlers for games, music, and dance routines."""

    _DANCE_INTRO_LINES: tuple[str, ...] = (
        "Fine. But only because you asked nicely.",
        "Watch and learn. This is how it's done.",
        "Oh, you want a show? Alright. Don't say I never gave you anything.",
        "I've been waiting for someone to ask me this.",
        "Prepare to be dazzled. Or confused. Possibly both.",
    )

    _DANCE_OUTRO_LINES: tuple[str, ...] = (
        "You're welcome. That was a gift.",
        "And that, my friend, is how it's done.",
        "I hope you appreciated that. I certainly did.",
        "Now you know why they call me DJ R3X.",
        "Don't clap too hard. I embarrass easily.",
    )

    _DANCE_MUSIC_INTRO_LINES: tuple[str, ...] = (
        "Oh, you want music? I've got just the thing.",
        "Buckle up. This one slaps.",
        "Alright, let's see if you can keep up.",
        "Dropping the beat in three, two...",
        "You called for music? Consider it done.",
    )

    _DANCE_MUSIC_OUTRO_LINES: tuple[str, ...] = (
        "And scene. You're welcome.",
        "That's a wrap. I hope you danced.",
        "Music off. Moment over. Back to business.",
        "Another banger in the books.",
        "Hope that was worth it. It was for me.",
    )

    def _handle_i_spy(self) -> State | None:
        """Run a single-turn I Spy round inside ACTIVE state."""
        start_line = random.choice(_I_SPY_START_LINES)
        log.info("I Spy: starting round")
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(start_line)
        except Exception:
            log.exception("I Spy: TTS error on intro")
        finally:
            self._end_speech(servo_stop)

        if not self._camera.is_available():
            return self._speak_i_spy_failure()

        if self._head_tracker is not None:
            self._head_tracker.pause("i_spy capture")
        restore = self._prepare_i_spy_camera_pose()
        frame = self._camera.capture_frame()
        self._restore_servo_pose(restore)
        if self._head_tracker is not None:
            self._head_tracker.resume("i_spy done")
        if not frame:
            log.warning("I Spy: camera capture returned no frame")
            return self._speak_i_spy_failure()

        analysis = self._llm.analyze_i_spy_scene(frame)
        round_data = build_round(analysis)
        if round_data is None:
            log.warning("I Spy: no usable scene analysis returned")
            return self._speak_i_spy_failure()

        log.info(
            "I Spy: answer=%r clue=%r scene=%r",
            round_data.answer, round_data.clue, round_data.scene_description,
        )

        clue_line = f"I spy {round_data.article} {round_data.clue}. One guess."
        servo_stop = self._begin_speech(emotion="neutral")
        try:
            self._synthesizer.speak(clue_line)
        except Exception:
            log.exception("I Spy: TTS error on clue")
        finally:
            self._end_speech(servo_stop)

        guess = self._listen_for_i_spy_guess()
        if guess is None:
            return None
        if guess == "":
            line = random.choice(_I_SPY_TIMEOUT_LINES).replace("{answer}", round_data.answer)
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("I Spy: TTS error on timeout reveal")
            finally:
                self._end_speech(servo_stop)
            return None

        guard_cmd = parse(guess, allow_fuzzy=False)
        if guard_cmd is not None and guard_cmd.action in {
            "cancel",
            "program_shutdown",
            "os_shutdown",
            "sleep",
            "idle",
        }:
            log.info("I Spy: interrupted by command %r", guard_cmd.action)
            return self._execute_command(guard_cmd, guess)

        if guess_matches(guess, round_data.answer):
            line = random.choice(_I_SPY_CORRECT_LINES).replace("{answer}", round_data.answer)
            emotion = "excited"
        else:
            line = random.choice(_I_SPY_WRONG_LINES).replace("{answer}", round_data.answer)
            emotion = "neutral"

        servo_stop = self._begin_speech(emotion=emotion)
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("I Spy: TTS error on result")
        finally:
            self._end_speech(servo_stop)
        return None

    def _listen_for_i_spy_guess(self) -> str | None:
        """Listen for an I Spy guess."""
        if not self._transcriber.is_available():
            return None

        guess = self._transcribe_i_spy_guess(config.I_SPY_GUESS_TIMEOUT_SECONDS)
        if guess is None:
            return None
        if guess:
            log.info("I Spy: guess=%r", guess)
            return guess

        nudge = random.choice(_I_SPY_NUDGE_LINES)
        servo_stop = self._begin_speech(emotion="neutral")
        try:
            self._synthesizer.speak(nudge)
        except Exception:
            log.exception("I Spy: TTS error on nudge")
        finally:
            self._end_speech(servo_stop)

        guess = self._transcribe_i_spy_guess(config.WAKE_GOODBYE_TIMEOUT)
        if guess is None:
            return None
        if guess:
            log.info("I Spy: guess=%r", guess)
            return guess
        return ""

    def _transcribe_i_spy_guess(self, timeout_seconds: float) -> str | None:
        """Capture one possible I Spy guess."""
        self._apply_listening_led_theme()
        self._wake_word.pause()
        try:
            self._wait_for_post_speech_listen_cooldown("I Spy guess listen")
            return self._transcriber.transcribe(
                wait_for_speech_seconds=timeout_seconds,
                allow_short=True,
            )
        except Exception:
            log.exception("I Spy: transcription error while waiting for guess")
            return None
        finally:
            self._wake_word.resume()
            self._apply_active_led_theme()

    def _speak_i_spy_failure(self) -> State | None:
        """Apologize in character and end the I Spy round cleanly."""
        line = random.choice(_I_SPY_FAIL_LINES)
        servo_stop = self._begin_speech(emotion="neutral")
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("I Spy: TTS error on failure fallback")
        finally:
            self._end_speech(servo_stop)
        return None

    def _handle_dance_short(self) -> State | None:
        """Speak an intro, dance to Cantina Band for a fixed duration, then stop."""
        if self._servos is not None and self._servos.is_dancing:
            self._speak_simple("I am already dancing. Keep up.", emotion="excited")
            return None

        intro = random.choice(self._DANCE_INTRO_LINES)
        self._speak_simple(intro, emotion="excited")

        if self._head_tracker is not None:
            self._head_tracker.pause("dance")

        if self._servos is not None:
            self._servos.stop()
            self._servos.start_dancing()

        cantina = config.CANTINA_BAND_PATH
        if cantina.exists():
            self._player.play_music(cantina, loop=False)
        else:
            log.warning("Cantina Band not found at %s — dancing without music", cantina)

        time.sleep(config.DANCE_SHORT_DURATION)
        self._player.fade_music(duration=config.DANCE_FADE_DURATION)

        if self._servos is not None:
            self._servos.stop_dancing()
            self._servos.start()

        if self._head_tracker is not None:
            self._head_tracker.resume("dance done")

        outro = random.choice(self._DANCE_OUTRO_LINES)
        self._speak_simple(outro, emotion="excited")
        return None

    def _handle_play_music(self) -> State | None:
        """Pick a random track, dance while it plays, then speak an outro."""
        if self._servos is not None and self._servos.is_dancing:
            self._speak_simple("I am already dancing. Keep up.", emotion="excited")
            return None

        if not self._music_tracks:
            log.info("No music tracks available for play_music action")
            self._speak_simple(
                "I don't seem to have any music loaded right now.",
                emotion="neutral",
            )
            return None

        track = random.choice(self._music_tracks)
        log.info("play_music action: selected %s", track.name)

        intro = random.choice(self._DANCE_MUSIC_INTRO_LINES)
        self._speak_simple(intro, emotion="excited")

        if self._head_tracker is not None:
            self._head_tracker.pause("play_music dance")

        if self._servos is not None:
            self._servos.stop()
            self._servos.start_dancing()

        self._player.play_music(track, loop=False)
        self._player.wait_for_music(timeout=600.0)

        if self._servos is not None:
            self._servos.stop_dancing()
            self._servos.start()

        if self._head_tracker is not None:
            self._head_tracker.resume("play_music done")

        outro = random.choice(self._DANCE_MUSIC_OUTRO_LINES)
        self._speak_simple(outro, emotion="excited")
        return None

    def _play_music_track(self) -> None:
        """Play the current indexed track if the library is populated."""
        if not self._music_tracks:
            log.info(
                "No music tracks found in %s — skipping playback",
                config.ASSETS_DIR / "music",
            )
            return
        track = self._music_tracks[self._music_index % len(self._music_tracks)]
        log.info("Playing music track: %s", track.name)
        self._player.play_music(track, loop=False)

    def _advance_and_play(self) -> None:
        """Advance the current music index and start playback."""
        if not self._music_tracks:
            return
        self._music_index = (self._music_index + 1) % len(self._music_tracks)
        self._play_music_track()
