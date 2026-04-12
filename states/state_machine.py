"""
states/state_machine.py — Three-state controller for DJ-R3X.

States
------
  IDLE     Wake word listening; chest LED slow-breathing; arms idle-fidget;
           no transcription or LLM calls. Clears conversation history so the
           next active session starts fresh.

  ACTIVE   Full interactivity: transcribe → parse → local response or LLM
           fallback → TTS. Mouth brightness thread and servo speech-reactive
           movement run during every utterance. Returns to IDLE after
           ACTIVE_IDLE_TIMEOUT of silence or on an explicit "idle" action.

  SHUTDOWN Play farewell sequence, home servos, fade LEDs, then halt the OS.

Threading
---------
  Main thread runs run(), which drives the state loop synchronously.
  Heavy subsystem threads (wake word, servo idle motion, mouth LEDs) are
  daemon threads owned by their respective classes and run for the full
  program lifetime.  The only thread the state machine itself spawns is
  _servo_speak_thread — a short-lived worker that calls speak_move() at
  ~20 Hz during TTS playback.

Graceful hardware degradation
------------------------------
  ServoController raises serial.SerialException if the Maestro port is
  absent. _try_init() catches that and sets self._servos = None; every servo
  call is guarded with `if self._servos is not None` so the program runs fine
  on a dev machine with no hardware attached.

Usage
-----
    sm = StateMachine()
    sm.start()   # warmup — loads wake word models, starts background threads
    sm.run()     # blocks until SHUTDOWN; calls close() in a finally block
"""

from __future__ import annotations

import enum
import difflib
import logging
import os
import random
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import config
from audio.player import AudioPlayer
from commands.parser import normalize, parse
from hardware.leds import LEDController
from hardware.servos import ServoController
from llm.chatgpt import ChatGPTClient
from llm.greeter import Greeter
from llm.vision_intent import vision_intent
from sequences.animations import AnimationPlayer
from speech.synthesizer import Synthesizer
from speech.transcriber import Transcriber
from speech.wake_word import WakeWordDetector
from utils import realworld
from vision.camera import Camera
from vision.face_db import FaceDB
from vision.head_tracker import HeadTracker
from vision.i_spy import build_round, guess_matches
from vision.face_recognizer import FaceRecognizer

log = logging.getLogger(__name__)

# How often the servo-speak worker calls speak_move() during TTS (~20 Hz).
_SERVO_SPEAK_INTERVAL: float = 0.05

_ARE_YOU_THERE_PHRASES: list[str] = [
    "Hello?! I know you're out there — I can hear you breathing, lifeform.",
    "Oh, now you're shy?! You activated ME, remember?",
    "Is anyone there, or did I just get stood up by a carbon-based unit AGAIN?",
    "I'm waiting. My patience circuits are surprisingly limited.",
    "Uh... hello?! I didn't clear my schedule for nothing!",
]

_GOODBYE_PHRASES: list[str] = [
    "Oh, you're just GONE. That's fine. I had better conversations with an R2 unit.",
    "Stood up AND abandoned. Classic lifeform behavior. Going back to sleep.",
    "Nothing?! Not even a goodbye?! Rude. Even Jawas say goodbye. Usually.",
    "Ok fine, I get it — I'm too much for you. Most beings are, honestly.",
]

_SHUTDOWN_PHRASES: list[str] = [
    "Shutting down — and honestly? I've had worse audiences. Not many, but some.",
    "Going offline. Try not to let the cantina fall apart without me. You will, but try.",
    "Powering down. It's been a blast, lifeform — a small blast, but still.",
    "See you in the next galaxy. I'll be the smoothest droid there too.",
    "Signing off. Don't touch my playlist while I'm gone. I will know.",
]

_IDLE_CLIPS: list[str] = [
    "This is your cap.mp3",
    "Yahoo.mp3",
    "Roger Control.mp3",
    "Request Line Open.mp3",
    "Outer Rim.mp3",
    "On the Decks.mp3",
    "Once Again.mp3",
    "Loose Wire.mp3",
    "Having Fun.mp3",
    "Events.mp3",
    "Endor Travel.mp3",
    "Dream.mp3",
    "DJ Pilot.mp3",
    "Astromech Joke.mp3",
]

_PLAN_QUESTION_WEEKDAY: tuple[str, ...] = (
    "{name}, what are you doing today besides wandering over here like a lost tourist on Batuu?",
    "All right, {name}, what's on today's agenda? Try to make it sound less tragic than it probably is.",
    "{name}, what are you up to today, and please tell me it beats staring at walls.",
)

_PLAN_QUESTION_FRIDAY: tuple[str, ...] = (
    "{name}, it's Friday. What are the weekend plans, and do they involve better judgment than usual?",
    "Friday, {name}. What are you doing this weekend, besides making bold choices with your free time?",
    "{name}, weekend planning check. What chaos are you scheduling for the next couple of days?",
)

_PLAN_QUESTION_WEEKEND: tuple[str, ...] = (
    "{name}, it's the weekend. What are you getting up to, and why does it already sound questionable?",
    "Weekend status check, {name}. What are you doing with it besides bothering a retired pilot droid?",
    "{name}, what is the weekend plan? Make it good. Or at least make it funny.",
)

_PLAN_QUESTION_NUDGES: tuple[str, ...] = (
    "No plan? That's bleak even for this cantina. What are you doing today?",
    "Still waiting, lifeform. What is the move today or this weekend?",
    "Anytime now. I asked what you're up to, not for a vow of silence.",
)

_PLAN_REPLY_LINES: tuple[str, ...] = (
    "Oh, {summary}? Bold. Is this a real plan, or did your day just lose a bet?",
    "{summary}, huh? That's either productive or deeply suspicious. Which is it?",
    "So we're doing {summary}. Is this for fun, survival, or a terrible promise you made earlier?",
)

_PLAN_SAME_DAY_FOLLOWUP_LINES: tuple[str, ...] = (
    "So, {name}, how's {summary} going? Be honest. I can handle disappointment.",
    "{name}, are you actually doing {summary}, or was that just aspirational theater?",
    "Checking in, {name}. Did {summary} turn into a productive day, or a full cantina-grade fiasco?",
    "{name}, how's that whole {summary} situation treating you so far?",
)

_ANGRY_TRIGGER_PHRASES: tuple[str, ...] = (
    "youre stupid",
    "fuck you",
    "youre dumb",
    "i hate you",
    "you suck",
    "you talk too much",
    "youre ugly",
    "youre not very smart",
)

_ANGRY_RESET_PHRASES: tuple[str, ...] = (
    "i didnt mean it",
    "i like you",
    "youre a smart droid",
    "youre smart",
    "youre a handsome droid",
)

_ANGRY_ON_LINES: tuple[str, ...] = (
    "Oh, now we're being rude. Fine. Grumpy mode engaged, you discount moisture farmer.",
    "Classy. You insult the droid and expect premium service. Angry mode activated.",
    "Wow. Personal attack logged. Fine, lifeform. You get the sharp version of Rex now.",
)

_ANGRY_OFF_LINES: tuple[str, ...] = (
    "Apology accepted. Against my better judgment, normal mode restored.",
    "Fine. Compliment received. I am returning to my regular, charming level of hostility.",
    "All right, truce accepted. Grumpy mode disengaged. Try not to earn it again.",
)

_PROMPT_COMMAND_ACTIONS: set[str] = {
    "cancel",
    "program_shutdown",
    "os_shutdown",
    "sleep",
    "idle",
    "dance_short",
    "play_music",
    "stop_music",
    "next_track",
    "volume_up",
    "volume_down",
    "forget_me",
    "wipe_memory",
    "rename_me",
    "recall_name",
    "recall_memories",
    "recall_preference",
}

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


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class State(enum.Enum):
    IDLE     = "idle"
    ACTIVE   = "active"
    SLEEP    = "sleep"
    SHUTDOWN = "shutdown"


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------

class StateMachine:
    """Owns all subsystems and drives the IDLE → ACTIVE → SHUTDOWN lifecycle."""

    def __init__(self) -> None:
        # Audio player first — LEDs and synthesizer depend on it.
        self._player = AudioPlayer()

        # Hardware — each may be absent; _try_init() returns None on failure.
        self._servos: ServoController | None = _try_init(
            ServoController, "servo controller"
        )
        self._leds = LEDController(self._player)

        # Speech pipeline
        self._wake_word = WakeWordDetector(on_detection=self._on_wake_word)
        self._transcriber = Transcriber()
        self._synthesizer = Synthesizer(self._player)

        # LLM
        self._llm = ChatGPTClient()
        self._greeter = Greeter()

        # Vision
        self._camera = Camera()
        self._face_db = FaceDB()
        self._face_recognizer = FaceRecognizer(self._face_db)

        # Head tracking — created here; started in start() after camera warmup.
        # Requires servos; disabled gracefully when servos are absent or
        # HEAD_TRACKING_ENABLED is False.
        if config.HEAD_TRACKING_ENABLED and self._servos is not None:
            self._head_tracker: HeadTracker | None = HeadTracker(
                self._servos, self._camera, config
            )
        else:
            self._head_tracker = None

        # State control
        self._state: State = State.IDLE
        self._wake_event = threading.Event()    # set by wake word callback
        self._shutdown_event = threading.Event()
        self._os_shutdown_requested: bool = False  # True only for voice/button shutdown
        self._pipeline_t0: float = 0.0          # monotonic time of last wake word detection

        # Music library — scanned once at startup
        self._music_tracks: list[Path] = _scan_music()
        self._music_index: int = 0

        # Chatty mode — must be explicitly enabled to hear idle atmosphere clips.
        # Default off so Rex sits silently in IDLE unless the operator enables it.
        self._chatty_mode: bool = False

        # Per-session idle clip mute — disabled by "stop talking" command,
        # re-enabled automatically the next time the wake word fires.
        # Clips only play when BOTH this AND _chatty_mode are True.
        self._idle_clips_enabled: bool = True

        # Tracks the person_id of the last face-recognised visitor so that a
        # "call me X" command can update their name without a second camera scan.
        # Cleared whenever we return to IDLE.
        self._last_known_person_id: int | None = None

        # The camera frame captured at wake word time — reused for enrollment
        # so the enrollment thread has a frame from when the face was definitely
        # in front of the camera.  Cleared whenever we return to IDLE.
        self._last_wake_frame: str | None = None

        # Alternates the initial wake audio cue between "Hi There.mp3" (False)
        # and "This is your cap.mp3" (True) on successive wake activations.
        # Ensures only ONE clip plays per wake — never both in the same activation.
        self._no_face_clip_toggle: bool = False

        # Session-greeting state — tracks who was greeted and how many times
        # the wake word has fired since boot.  Used by the 5-case greeting logic:
        #   _session_wake_count == 0   → Case 1 (first wake since boot)
        #   _session_greeted_person_id → person_id set at first recognized greeting
        #   _last_greeted_person_id    → most recently greeted person
        # These are never reset mid-session; only re-initialized at __init__.
        self._session_greeted_person_id: int | None = None
        self._session_wake_count: int = 0
        self._last_greeted_person_id: int | None = None

        # Per-day count of how many times each recognized person has woken Rex.
        # Used only for roast wording; resets automatically when the date changes.
        self._recognized_today_date: date = date.today()
        self._recognized_today_counts: dict[int, int] = {}
        self._post_greeting_person_id: int | None = None
        self._post_greeting_person_name: str | None = None
        self._post_greeting_prompt_used: bool = False
        self._angry_mode: bool = False
        self._last_wake_greeting_used_vision: bool = False

        # Animation player — shares hardware refs with the rest of the machine
        self._animations = AnimationPlayer(self._servos, self._leds)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Warmup all subsystems. Call once before run()."""
        log.info("StateMachine: warming up subsystems …")

        self._transcriber.warmup()   # no-op for Whisper; kept for interface consistency
        self._transcriber.calibrate_noise_floor()
        startup_error: list[Exception | None] = [None]

        def _finish_startup() -> None:
            try:
                if self._wake_word.is_available():
                    self._wake_word.warmup()
                    self._wake_word.start()
                else:
                    log.warning(
                        "Wake word models not found — wake word detection disabled. "
                        "Rex will never leave IDLE unless request_shutdown() is called."
                    )

                if self._servos is not None:
                    self._servos.start()

                self._leds.start()

                self._camera.warmup()
                self._camera.start()
            except Exception as exc:  # noqa: BLE001
                startup_error[0] = exc

        startup_thread = threading.Thread(
            target=_finish_startup,
            daemon=True,
            name="djr3x-startup-warmup",
        )
        startup_thread.start()

        filler_line = random.choice([
            "Uh... hang on... startup routines are still rattling around in here...",
            "Um... almost ready... just waking up the rest of my circuits...",
            "Okay... okay... one second... systems are still coming online...",
            "Hmmm... give me a second here... getting everything spun up...",
        ])
        servo_stop = None
        try:
            servo_stop = self._begin_speech(emotion="neutral")
            self._synthesizer.speak(filler_line)
        except Exception:
            log.exception("StateMachine: startup filler speech failed")
        finally:
            if servo_stop is not None:
                self._end_speech(servo_stop)
            else:
                self._wake_word.suppressed = False

        startup_thread.join()
        if startup_error[0] is not None:
            raise startup_error[0]

        # Start head tracker after camera warmup so is_available() is reliable.
        if self._head_tracker is not None:
            self._head_tracker.start()
            log.info("StateMachine: Head tracking  — CONNECTED (shared camera)")
        elif not config.HEAD_TRACKING_ENABLED:
            log.info("StateMachine: Head tracking  — DISABLED")
        else:
            log.info("StateMachine: Head tracking  — MISSING (no servo controller)")

        log.info("StateMachine: all subsystems ready.")

    def run(self) -> None:
        """Main event loop. Blocks until SHUTDOWN completes."""
        log.info("StateMachine: entering main loop (state=%s)", self._state.value)
        try:
            while True:
                if self._state == State.IDLE:
                    self._run_idle()
                elif self._state == State.ACTIVE:
                    self._run_active()
                elif self._state == State.SLEEP:
                    self._run_sleep()
                elif self._state == State.SHUTDOWN:
                    self._run_shutdown()
                    break
        finally:
            self.close()

    def close(self) -> None:
        """Stop all subsystems and release hardware resources."""
        log.info("StateMachine: shutting down subsystems")
        for label, fn in (
            ("wake word", self._wake_word.stop),
            ("head tracker", self._head_tracker.stop if self._head_tracker is not None else None),
            ("servos", self._servos.close if self._servos is not None else None),
            ("leds", self._leds.close),
            ("audio player", self._player.close),
            ("camera", self._camera.stop),
            ("face db", self._face_db.close),
        ):
            if fn is None:
                continue
            try:
                fn()
            except Exception:
                log.exception("StateMachine: error while closing %s", label)

    def request_shutdown(self) -> None:
        """Thread-safe: schedule a transition to SHUTDOWN from any thread
        (e.g. a SIGINT handler in main.py)."""
        log.info("StateMachine: shutdown requested externally")
        self._shutdown_event.set()
        self._wake_event.set()   # unblock _run_idle() if it's waiting

    def play_startup_animation(self) -> None:
        """Start startup music and the boot animation concurrently, then block
        until both finish.

        light_speed.mp3 (STARTUP_MUSIC_PATH) begins playing through the music
        path (no mouth LEDs) at the same time as the servo animation.  If the
        music file is missing it is skipped silently.  Must be called *before*
        start() so the servo idle thread is not running yet.
        """
        _music_path = config.STARTUP_MUSIC_PATH
        if _music_path.exists():
            log.info("Playing startup music (%s) …", _music_path)
            self._player.play_music(_music_path, loop=False)
        else:
            log.warning("Startup music not found at %s — skipping.", _music_path)
            _music_path = None  # sentinel: skip wait below

        log.info("Playing startup animation …")
        self._animations.play_startup()   # non-blocking

        # Load dlib face recognition models concurrently with the animation so
        # the ~3 s model load on Pi 4 is hidden behind the startup sequence.
        face_warmup_thread = threading.Thread(
            target=self._face_recognizer.warmup,
            daemon=True,
            name="djr3x-face-warmup",
        )
        face_warmup_thread.start()

        anim_done = self._animations.wait(timeout=10.0)
        if not anim_done:
            log.warning("Startup animation timed out — continuing.")
        else:
            log.info("Startup animation complete.")

        if _music_path is not None:
            music_done = self._player.wait_for_music(timeout=120.0)
            if music_done:
                log.info("Startup music complete.")
            else:
                log.warning("Startup music timed out — continuing.")
                self._player.stop_music()

        # Ensure face recognition models are fully loaded before continuing.
        face_warmup_thread.join(timeout=15.0)
        if face_warmup_thread.is_alive():
            log.warning("Face recognizer warmup timed out — recognition may be unavailable.")
        else:
            log.info("Face recognizer warmup complete.")

    def play_startup_intro(self) -> None:
        """Play the spoken intro clip through the speech path with full
        mouth-LED and servo speech animation.

        Must be called *after* play_startup_animation() (so servos are idle)
        and *before* start() (so the wake-word and idle threads are not yet
        competing for hardware).  Gracefully skipped if the file is missing.
        """
        path = config.STARTUP_INTRO_PATH
        if not path.exists():
            log.warning("Startup intro not found at %s — skipping.", path)
            return

        log.info("Playing startup intro (%s) …", path)
        servo_stop = None
        try:
            servo_stop = self._begin_speech(emotion="neutral")
            self._player.play_file(path)
            self._player.wait_for_speech(timeout=30.0)
        except Exception:
            log.exception("Startup intro: playback error — continuing")
        finally:
            if servo_stop is not None:
                self._end_speech(servo_stop)
            else:
                self._wake_word.suppressed = False
        log.info("Startup intro complete.")

    def play_startup_chime(self) -> None:
        """Play the startup chime through the music output path and block
        until it finishes.  Must be called after start() so the AudioPlayer
        OutputStream is running.  No-ops if the chime file is missing."""
        log.info("Playing startup chime (%s) …", config.STARTUP_CHIME_PATH)
        self._player.play_chime()
        finished = self._player.wait_for_music(timeout=30.0)
        if finished:
            log.info("Startup chime complete.")
        else:
            log.warning("Startup chime timed out — continuing.")

    def hardware_status(self) -> dict[str, bool | int]:
        """Return a snapshot of detected hardware and model availability.

        Intended for the startup banner in main.py. Safe to call before
        start() — no models or threads need to be running.
        """
        wake_models = sum(
            p.exists()
            for p in (config.WAKE_WORD_MODEL_1, config.WAKE_WORD_MODEL_2)
        )
        return {
            "servos":            self._servos is not None,
            "chest_leds":        self._leds._chest is not None,
            "head_leds":         self._leds._head is not None,
            "transcriber":       self._transcriber.is_available(),
            "wake_word":         self._wake_word.is_available(),
            "wake_models":       wake_models,       # int: 0, 1, or 2
            "music_tracks":      len(self._music_tracks),
            "camera":            self._camera.is_available(),
            "face_recognition":  self._face_recognizer.is_available(),
        }

    # ------------------------------------------------------------------
    # State — IDLE
    # ------------------------------------------------------------------

    def _run_idle(self) -> None:
        """Enter IDLE: set LEDs/servos to rest, then wait for a wake word."""
        log.info("→ IDLE")

        # Send EYE first so the head Nano has eyeColor set before IDLE arrives.
        # The IDLE handler activates the blink system only when eyeColor is
        # non-black, so this order guarantees blinking starts on IDLE entry.
        self._apply_idle_led_theme()

        if self._servos is not None:
            self._servos.set_emotion("neutral")

        # Unsuppress wake word (may have been left suppressed if we cut off
        # speech mid-utterance, e.g. via request_shutdown).
        self._wake_word.suppressed = False

        # Reset conversation context and mood so each active session starts fresh.
        self._llm.clear_history()
        if self._angry_mode:
            self._angry_mode = False
            self._llm.set_angry_mode(False)
            log.info("ACTIVE entry: angry mode cleared")

        # Wait for a wake word, playing random atmosphere clips in between.
        while True:
            interval = random.uniform(
                config.IDLE_CLIP_INTERVAL_MIN, config.IDLE_CLIP_INTERVAL_MAX
            )
            triggered = self._wake_event.wait(timeout=interval)

            if triggered:
                self._wake_event.clear()
                self._transition_to(
                    State.SHUTDOWN if self._shutdown_event.is_set() else State.ACTIVE
                )
                return

            # Timer fired — play a random idle clip only when chatty mode is
            # enabled AND the per-session mute hasn't been triggered.
            if not self._chatty_mode or not self._idle_clips_enabled:
                continue

            clip_path = config.ASSETS_DIR / "audio" / random.choice(_IDLE_CLIPS)
            if clip_path.exists():
                log.info("Idle clip: %s", clip_path.name)
                servo_stop = None
                try:
                    log.debug("Idle clip: calling _begin_speech()")
                    servo_stop = self._begin_speech(emotion="neutral")
                    log.debug("Idle clip: _begin_speech() returned — starting play_file()")
                    self._player.play_file(clip_path)
                    log.debug("Idle clip: play_file() returned")
                except Exception:
                    log.exception("Idle clip playback error: %s", clip_path.name)
                finally:
                    if servo_stop is not None:
                        log.debug("Idle clip: calling _end_speech()")
                        self._end_speech(servo_stop)
                        log.debug("Idle clip: _end_speech() returned")
                    else:
                        self._wake_word.suppressed = False
                # Belt-and-suspenders: stop_mouth() is idempotent — a second
                # call is harmless but ensures SPEAK_STOP reaches the Arduino
                # even if the _trigger_mouth race window caused it to re-enter
                # SPEAK mode after _end_speech() already fired.
                log.debug("Idle clip: safety stop_mouth() call after finally block")
                self._leds.stop_mouth()
                # Absolute last resort: write SPEAK_STOP directly to the head
                # Nano serial port, bypassing all LED state tracking.
                log.debug("Idle clip: direct SPEAK_STOP safety write to head Nano")
                self._leds._send_head(config.LED_CMD_SPEAK_STOP)

            # Handle wake word or shutdown that arrived during clip playback.
            if self._wake_event.is_set():
                self._wake_event.clear()
                self._transition_to(
                    State.SHUTDOWN if self._shutdown_event.is_set() else State.ACTIVE
                )
                return

    # ------------------------------------------------------------------
    # State — ACTIVE
    # ------------------------------------------------------------------

    def _run_active(self) -> None:
        """Full interactivity loop: transcribe → parse/LLM → speak; repeat
        until a silence timeout or an explicit state-change action.

        Two timeout modes:
          - First listen after wake word: WAKE_NO_SPEECH_TIMEOUT seconds.
            On timeout, Rex prompts "Are you there?" then waits
            WAKE_GOODBYE_TIMEOUT more seconds.  If still silent, Rex says
            goodbye and returns to IDLE.
          - Follow-up listen after a response: ACTIVE_TIMEOUT_SECONDS.
            On timeout, Rex returns to IDLE directly (no prompt).
        """
        log.info("→ ACTIVE")

        self._apply_active_led_theme()

        if self._servos is not None:
            self._servos.set_emotion("neutral")

        # Start arm wave in background (non-blocking — idle thread stopped inside).
        self._animations.play_wake_greeting_arms()

        # Greet the user concurrently with the arm wave.
        self._play_wake_greeting()

        # The greeting path can request shutdown (for example if the user says
        # "shutdown" when asked for their name).  Honor that before restoring
        # idle servo motion so we don't briefly restart hardware we're about to
        # power down.
        if self._shutdown_event.is_set():
            self._transition_to(State.SHUTDOWN)
            return
        if self._state != State.ACTIVE:
            return

        # Wait for the wave to finish (usually already done by the time audio ends),
        # then restore hand speed and restart the servo idle thread.
        self._animations.wait(timeout=5.0)
        if self._servos is not None:
            self._servos.set_channel_speed(
                config.SERVO_HAND_LEFT, config.SERVO_DEFAULT_SPEED
            )
            self._servos.start()

        # Re-enable idle clips now that the user has interacted again.
        self._idle_clips_enabled = True

        # False = first listen since wake word; True = follow-up after a response.
        after_response = False

        while self._state == State.ACTIVE and not self._shutdown_event.is_set():
            if not self._transcriber.is_available():
                time.sleep(1.0)
                continue

            # --- Listen indicator ---
            self._apply_listening_led_theme()

            # Choose how long to wait for speech to start.
            speech_timeout = (
                config.ACTIVE_TIMEOUT_SECONDS if after_response
                else config.WAKE_NO_SPEECH_TIMEOUT
            )

            # --- Transcribe (blocks until speech+silence, timeout, or MAX_RECORD_SECONDS) ---
            # Pause wake word: both share the same mic device.
            self._wake_word.pause()
            try:
                text = self._transcriber.transcribe(
                    wait_for_speech_seconds=speech_timeout,
                    t0=self._pipeline_t0,
                )
            except Exception:
                log.exception("Transcription error — skipping utterance")
                continue
            finally:
                self._wake_word.resume()

            # --- No usable speech detected within the timeout window ---
            # Treat both a hard timeout (None) and an empty / hallucination-
            # filtered transcription ("") as silence so background noise
            # does not trap Rex in repeated fake-speech loops.
            if text is None or text == "":
                if text == "":
                    log.info("Empty transcription treated as silence/no-response")
                if after_response:
                    # If music is playing, the mic likely picked up the track.
                    # Don't return to IDLE — keep listening for real commands.
                    if self._player.is_music_playing:
                        log.info(
                            "Music is playing — suppressing idle timeout, continuing to listen"
                        )
                        continue
                    log.info(
                        "Follow-up silence timeout (%.0f s) — returning to IDLE",
                        config.ACTIVE_TIMEOUT_SECONDS,
                    )
                    self._play_return_to_idle_chime()
                    self._transition_to(State.IDLE)
                    return

                if (
                    self._post_greeting_person_id is not None
                    and not self._post_greeting_prompt_used
                ):
                    # On subsequent wakes, 50% of the time ask a remaining
                    # interview question instead of the plan prompt.
                    use_interview = (
                        self._session_wake_count > 1
                        and random.random() < 0.5
                    )
                    if use_interview and self._run_followup_interview_question():
                        after_response = True
                        continue

                    if not self._post_greeting_prompt_used:
                        status, next_state = self._run_post_greeting_plan_prompt()
                        if next_state is not None:
                            if next_state == State.IDLE:
                                self._play_return_to_idle_chime()
                            self._transition_to(next_state)
                            return
                        if status != "no_answer":
                            after_response = True
                            continue

                # First listen — prompt with "are you there?"
                prompt = random.choice(_ARE_YOU_THERE_PHRASES)
                log.info("No speech on first listen — prompting: %r", prompt)
                self._apply_active_led_theme()
                servo_stop = None
                try:
                    servo_stop = self._begin_speech(emotion="neutral")
                    self._synthesizer.speak(prompt)
                except Exception:
                    log.exception("Error speaking 'are you there' prompt")
                finally:
                    if servo_stop is not None:
                        self._end_speech(servo_stop)
                    else:
                        self._wake_word.suppressed = False

                # Second-chance listen
                self._apply_listening_led_theme()
                self._wake_word.pause()
                try:
                    text = self._transcriber.transcribe(
                        wait_for_speech_seconds=config.WAKE_GOODBYE_TIMEOUT
                    )
                except Exception:
                    log.exception("Transcription error in second-chance listen")
                    text = None
                finally:
                    self._wake_word.resume()

                if not text:   # None (timeout) or "" (Whisper got nothing)
                    goodbye = random.choice(_GOODBYE_PHRASES)
                    log.info("Still no speech — saying goodbye: %r", goodbye)
                    self._apply_active_led_theme()
                    servo_stop = None
                    try:
                        servo_stop = self._begin_speech(emotion="neutral")
                        self._synthesizer.speak(goodbye)
                    except Exception:
                        log.exception("Error speaking goodbye phrase")
                    finally:
                        if servo_stop is not None:
                            self._end_speech(servo_stop)
                        else:
                            self._wake_word.suppressed = False
                    self._play_return_to_idle_chime()
                    self._transition_to(State.IDLE)
                    return
                # else: second-chance produced text — fall through to process it

            # --- We have a transcription ---
            after_response = True
            log.info("Transcribed: %r", text)

            # --- Speaking indicator ---
            self._apply_active_led_theme()

            mood_shift = self._classify_angry_intent(text)
            if mood_shift == "on":
                log.info("Angry mode: entering (trigger=%r)", text)
                self._set_angry_mode(True)
                self._speak_simple(_pick_no_repeat(_ANGRY_ON_LINES, "angry_on"), emotion="neutral")
                self._player.wait_for_speech()
                self._apply_active_led_theme()
                continue
            if mood_shift == "off":
                log.info("Angry mode: clearing (trigger=%r)", text)
                self._set_angry_mode(False)
                self._speak_simple(_pick_no_repeat(_ANGRY_OFF_LINES, "angry_off"), emotion="neutral")
                self._player.wait_for_speech()
                self._apply_active_led_theme()
                continue

            # --- Parse and respond ---
            _elapsed = f" [+{time.monotonic() - self._pipeline_t0:.1f}s]"
            # Route deterministically:
            #   1. Exact/prefix local commands
            #   2. Vision intent fallback
            #   3. Fuzzy local commands
            #   4. Plain LLM fallback
            # This keeps fuzzy matching from stealing open-ended or visual
            # questions that should go to the LLM path.
            cmd = parse(text, allow_fuzzy=False)
            if cmd is not None and cmd.action == "vision":
                # Vision command — capture a fresh frame right now and send to LLM.
                log.info("Vision command: %r%s", cmd.phrases[0], _elapsed)
                _pose = self._prepare_camera_pose()
                frame = self._camera.capture_frame()
                self._restore_servo_pose(_pose)
                if frame:
                    log.debug("Camera: fresh frame captured for vision command (%d bytes b64)",
                              len(frame))
                next_state = self._speak_llm(text, image=frame, t0=self._pipeline_t0)
            elif cmd is not None:
                log.info("Command matched: %r → %r%s", cmd.phrases[0], cmd.action, _elapsed)
                next_state = self._execute_command(cmd, text)
            else:
                # No exact/prefix command match — check visual intent before
                # allowing fuzzy command matching or plain LLM fallback.
                if vision_intent(text):
                    log.info("Vision intent detected — capturing frame%s", _elapsed)
                    _pose = self._prepare_camera_pose()
                    frame = self._camera.capture_frame()
                    self._restore_servo_pose(_pose)
                    if frame:
                        log.debug("Camera: fresh frame captured for vision intent (%d bytes b64)",
                                  len(frame))
                    next_state = self._speak_llm(text, image=frame, t0=self._pipeline_t0)
                else:
                    fuzzy_cmd = parse(text, allow_fuzzy=True)
                    if fuzzy_cmd is not None:
                        log.info(
                            "Command fuzzy-matched: %r → %r%s",
                            fuzzy_cmd.phrases[0], fuzzy_cmd.action, _elapsed
                        )
                        next_state = self._execute_command(fuzzy_cmd, text)
                    else:
                        log.info("No vision intent or command match — sending text only to ChatGPT%s", _elapsed)
                        next_state = self._speak_llm(text, image=None, t0=self._pipeline_t0)

            # Ensure all audio has finished before re-opening the mic.
            self._player.wait_for_speech()

            if next_state is not None:
                if next_state == State.IDLE:
                    self._play_return_to_idle_chime()
                self._transition_to(next_state)
                return

            # Restore ACTIVE indicators for next listen turn.
            self._apply_active_led_theme()

        # Loop exited because _shutdown_event was set.
        if self._state == State.ACTIVE:
            self._transition_to(State.SHUTDOWN)

    # ------------------------------------------------------------------
    # State — SHUTDOWN
    # ------------------------------------------------------------------

    def _run_shutdown(self) -> None:
        """Speak a goodbye line, play shutdown animation, home hardware, then halt the OS."""
        log.info("→ SHUTDOWN")

        # Speak a farewell line before the animation — idle thread still running
        # so speech-reactive servo movement works normally.
        phrase = random.choice(_SHUTDOWN_PHRASES)
        log.info("Shutdown speech: %r", phrase)
        servo_stop = None
        try:
            servo_stop = self._begin_speech(emotion="neutral")
            self._synthesizer.speak(phrase)
        except Exception:
            log.exception("Shutdown speech: TTS error — continuing to shutdown")
        finally:
            if servo_stop is not None:
                self._end_speech(servo_stop)
            else:
                self._wake_word.suppressed = False

        # Stop servo idle thread before the animation so arm channels are free.
        if self._servos is not None:
            self._servos.stop()

        # Head tracker must not fight the shutdown animation for neck/headtilt.
        if self._head_tracker is not None:
            self._head_tracker.pause("shutdown animation")

        # Start shutdown music and servo animation concurrently, then wait for
        # both to finish before tearing down hardware.
        _shutdown_music_path = config.SHUTDOWN_MUSIC_PATH
        if _shutdown_music_path.exists():
            log.info("Playing shutdown music (%s) …", _shutdown_music_path)
            self._player.play_music(_shutdown_music_path, loop=False)
        else:
            log.warning("Shutdown music not found at %s — skipping.", _shutdown_music_path)

        self._animations.play_shutdown()   # non-blocking; runs in daemon thread

        anim_done = self._animations.wait(timeout=15.0)
        if not anim_done:
            log.warning("Shutdown animation timed out — continuing.")

        if _shutdown_music_path.exists():
            music_done = self._player.wait_for_music(timeout=60.0)
            if not music_done:
                log.warning("Shutdown music timed out — continuing.")
                self._player.stop_music()

        # Shutdown animation IS the final servo state — do not call home() here,
        # it would override the slumped pose with neutral positions.
        # Just kill the LEDs.
        self._leds.set_chest_effect(config.LED_CMD_OFF)
        self._leds.set_head_effect(config.LED_CMD_OFF)

        # Halt the OS only when explicitly requested (voice command / physical
        # button) AND the production flag is enabled.  Ctrl-C (SIGINT) does
        # NOT set _os_shutdown_requested, so development exits are safe.
        if self._os_shutdown_requested and config.ENABLE_OS_SHUTDOWN:
            log.info("StateMachine: halting OS — sudo shutdown -h now")
            os.system("sudo shutdown -h now")
        else:
            if not self._os_shutdown_requested:
                log.info("StateMachine: clean exit (signal/dev) — skipping OS shutdown")
            else:
                log.info("StateMachine: ENABLE_OS_SHUTDOWN=False — skipping OS shutdown")

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: State) -> None:
        log.info("Transition: %s → %s", self._state.value, new_state.value)
        if new_state == State.SHUTDOWN:
            self._shutdown_event.set()
        if new_state in (State.IDLE, State.SLEEP):
            self._last_known_person_id = None
            self._last_wake_frame = None
            self._post_greeting_person_id = None
            self._post_greeting_person_name = None
            self._post_greeting_prompt_used = False
            self._llm.clear_person_context()
            if self._angry_mode:
                self._angry_mode = False
                self._llm.set_angry_mode(False)
                log.info("Transition: angry mode cleared on %s entry", new_state.value)
            # Global mouth safety: guarantee mouth is off whenever Rex returns
            # to IDLE or SLEEP, regardless of what the LED state machine thinks.
            self._leds.stop_mouth()
            self._leds._send_head(config.LED_CMD_SPEAK_STOP)
        self._state = new_state

    # ------------------------------------------------------------------
    # Wake word callback  (called from WakeWordDetector audio thread)
    # ------------------------------------------------------------------

    def _on_wake_word(self, model_name: str) -> None:
        # The sleep wake model fires ONLY in SLEEP state; all other models
        # fire ONLY in IDLE state.  Detections in ACTIVE/SHUTDOWN are ignored
        # (suppression should already block them — this is belt-and-suspenders).
        _sleep_stem = config.WAKE_SLEEP_MODEL.stem.lower()   # e.g. "wakeuprex"
        _is_sleep_model = model_name.lower() == _sleep_stem

        if self._state == State.IDLE and not _is_sleep_model:
            self._pipeline_t0 = time.monotonic()
            log.info("Wake word detected (%s)", model_name)
            self._wake_event.set()
        elif self._state == State.SLEEP and _is_sleep_model:
            self._pipeline_t0 = time.monotonic()
            log.info("Sleep wake word detected (%s) — waking Rex up", model_name)
            self._wake_event.set()
        else:
            log.debug(
                "Wake word %r ignored (state=%s, is_sleep_model=%s)",
                model_name, self._state.value, _is_sleep_model,
            )

    # ------------------------------------------------------------------
    # Speech helpers
    # ------------------------------------------------------------------

    def _play_initial_wake_clip(self) -> bool:
        """Play the canned wake clip, alternating between the two clips on
        successive wake activations so only ONE ever plays per wake.

        Toggle False → "Hi There.mp3"  (then flips to True)
        Toggle True  → "This is your cap.mp3"  (then flips to False)

        Returns True if the clip was played, False if it was unavailable or
        playback failed before audio could be queued.
        """
        clip_name = "This is your cap.mp3" if self._no_face_clip_toggle else "Hi There.mp3"
        self._no_face_clip_toggle = not self._no_face_clip_toggle
        clip_path = config.ASSETS_DIR / "audio" / clip_name
        if not clip_path.exists():
            return False

        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._player.play_file(clip_path)
            self._player.wait_for_speech(timeout=10.0)
            return True
        except Exception:
            log.exception("Wake greeting: initial wake clip error")
            return False
        finally:
            self._end_speech(servo_stop)

    def _play_wake_greeting(self) -> None:
        """Greet the user on wake word activation.

        Routes to one of five cases based on session state and face recognition:

          Case 1 — First wake since boot: full recognition + personalized GPT greeting.
          Case 2 — Same person as last greeted: occasional brief remark (25%) or silent.
          Case 3 — Different known person: GPT handoff comment referencing previous person.
          Case 4 — Unknown person: stranger snark line + enrollment flow.
          Case 5 — No face detected (any wake): fully silent.

        Always completes before returning so the caller can open the mic immediately.
        """
        # Capture frame first — used for both first and subsequent wakes.
        frame: str | None = None
        if self._camera.is_available():
            _pose = self._prepare_camera_pose()
            frame = self._camera.capture_frame()
            self._restore_servo_pose(_pose)
            if frame:
                self._last_wake_frame = frame

        is_first_wake = (self._session_wake_count == 0)

        if is_first_wake:
            # Case 1: play initial clip then do full personalized greeting
            self._play_initial_wake_clip()
            self._play_first_wake_greeting(frame)
        else:
            # Cases 2–5: silent face scan, minimal speech
            self._play_subsequent_wake_greeting(frame)

        self._session_wake_count += 1

    def _play_first_wake_greeting(self, frame: str | None) -> None:
        """Case 1: first wake since boot — full face recognition + personalized greeting.

        Known person  → GPT-4o roast by name + appearance + visit count.
        Unknown face  → GPT-4o appearance roast + enrollment flow.
        No face / no recognition → nothing (Hi There clip was already played).
        """
        if not (frame and self._face_recognizer.is_available()):
            return

        use_vision_greeting = self._should_use_vision_wake_greeting()
        log.info("Wake greeting: first wake vision greeting=%s", use_vision_greeting)

        _RECOGNITION_FILLER_LINES = (
            "Uh... hang on... hang on... I'm looking...",
            "Um... let's see... yeah... still processing...",
            "Okay... okay... hold still... almost got it...",
            "Hmmm... let's see here... give me a second...",
            "Hang on... just a tiny second... thinking... thinking...",
        )

        # Run recognition concurrently; if it takes >0.25 s play a filler line
        # to hide the dlib latency rather than standing in silence.
        face_result: list = [("no_face", None)]

        def _identify() -> None:
            face_result[0] = self._face_recognizer.identify_with_status(
                frame, tolerance=config.FACE_RECOGNITION_TOLERANCE
            )

        face_thread = threading.Thread(
            target=_identify, daemon=True, name="djr3x-face-identify"
        )
        face_thread.start()
        face_thread.join(timeout=0.25)

        if face_thread.is_alive():
            filler = random.choice(_RECOGNITION_FILLER_LINES)
            log.info("Wake greeting: first wake — face recognition filler %r", filler)
            servo_stop = self._begin_speech(emotion="excited")
            try:
                self._synthesizer.speak(filler)
            except Exception:
                log.exception("Wake greeting: first wake filler TTS error")
            finally:
                self._end_speech(servo_stop)

        face_thread.join()
        status, result = face_result[0]

        # ---- Known person ------------------------------------------------
        if result is not None:
            person_id, name, _dist = result
            self._face_db.update_last_seen(person_id)

            today = date.today()
            if today != self._recognized_today_date:
                self._recognized_today_date = today
                self._recognized_today_counts.clear()
            bother_count = self._recognized_today_counts.get(person_id, 0) + 1
            self._recognized_today_counts[person_id] = bother_count

            self._last_known_person_id = person_id
            self._session_greeted_person_id = person_id
            self._last_greeted_person_id = person_id
            self._post_greeting_person_id = person_id
            self._post_greeting_person_name = name
            self._post_greeting_prompt_used = False

            # Inject memories into LLM context for this session.
            _ctx = self._face_db.get_memories_as_context(person_id)
            if _ctx:
                self._llm.set_person_context(_ctx)

            log.info(
                "Wake greeting: first wake — known person '%s' (visit count today=%d)",
                name, bother_count,
            )

            if use_vision_greeting:
                # GPT-4o personalized greeting — initial clip already played above.
                greeting_result: list[str | None] = [None]

                def _gen_known() -> None:
                    greeting_result[0] = self._greeter.generate_known_person_greeting(
                        name, bother_count, frame
                    )

                gen_thread = threading.Thread(
                    target=_gen_known, daemon=True, name="djr3x-greeter"
                )
                gen_thread.start()

                # Speak a filler phrase while the GPT call is in flight (~2s latency).
                if gen_thread.is_alive():
                    _filler = _pick_no_repeat(_GREETING_FILLERS, "greeting_filler")
                    log.info("Wake greeting: GPT filler %r", _filler)
                    _fs = self._begin_speech(emotion="excited")
                    try:
                        self._synthesizer.speak(_filler)
                    except Exception:
                        log.exception("Wake greeting: GPT filler TTS error")
                    finally:
                        self._end_speech(_fs)

                servo_stop = self._begin_speech(emotion="excited")
                try:
                    gen_thread.join(timeout=15.0)
                    if greeting_result[0]:
                        self._synthesizer.speak(greeting_result[0])
                except Exception:
                    log.exception("Wake greeting: first wake known-person greeter error")
                    gen_thread.join(timeout=1.0)
                finally:
                    self._end_speech(servo_stop)

                if not greeting_result[0]:
                    log.info("Wake greeting: GPT failed — canned fallback for known person")
                    self._play_known_person_greeting(name, bother_count)
                    self._last_wake_greeting_used_vision = False
                else:
                    self._last_wake_greeting_used_vision = True
            else:
                self._play_known_person_greeting(name, bother_count)
                self._last_wake_greeting_used_vision = False

            self._maybe_speak_followup(person_id)
            return

        # ---- Unknown face ------------------------------------------------
        if status in ("no_match", "db_empty"):
            log.info("Wake greeting: first wake — unknown face, running appearance roast + enrollment")

            _CANNED_TTS = (
                "Oh great, you're here. The cantina just got significantly louder and marginally more interesting.",
                "A lifeform! Bold of you to show up looking like THAT.",
                "Oh, it's you. Oga's Cantina — where even the questionable guests are welcome!",
                "Well well well, look what the Ronto dragged in. Welcome, I guess.",
            )
            if use_vision_greeting:
                # GPT-4o appearance-based roast — initial clip already played above.
                unknown_result: list[str | None] = [None]

                def _gen_unknown() -> None:
                    unknown_result[0] = self._greeter.generate(frame)

                gen_thread2 = threading.Thread(
                    target=_gen_unknown, daemon=True, name="djr3x-greeter"
                )
                gen_thread2.start()

                # Speak a filler phrase while the GPT call is in flight (~2s latency).
                if gen_thread2.is_alive():
                    _filler2 = _pick_no_repeat(_GREETING_FILLERS, "greeting_filler")
                    log.info("Wake greeting: GPT filler (unknown) %r", _filler2)
                    _fs2 = self._begin_speech(emotion="excited")
                    try:
                        self._synthesizer.speak(_filler2)
                    except Exception:
                        log.exception("Wake greeting: GPT filler TTS error (unknown)")
                    finally:
                        self._end_speech(_fs2)

                servo_stop = self._begin_speech(emotion="excited")
                try:
                    gen_thread2.join(timeout=15.0)
                    if unknown_result[0]:
                        self._synthesizer.speak(unknown_result[0])
                except Exception:
                    log.exception("Wake greeting: first wake unknown-face greeter error")
                    gen_thread2.join(timeout=1.0)
                finally:
                    self._end_speech(servo_stop)

                if not unknown_result[0]:
                    log.info("Wake greeting: GPT failed — canned TTS fallback for unknown face")
                    servo_stop = self._begin_speech(emotion="excited")
                    try:
                        self._synthesizer.speak(random.choice(_CANNED_TTS))
                    except Exception:
                        log.exception("Wake greeting: first wake unknown-face canned fallback error")
                    finally:
                        self._end_speech(servo_stop)
                    self._last_wake_greeting_used_vision = False
                else:
                    self._last_wake_greeting_used_vision = True
            else:
                servo_stop = self._begin_speech(emotion="excited")
                try:
                    self._synthesizer.speak(random.choice(_CANNED_TTS))
                except Exception:
                    log.exception("Wake greeting: first wake unknown-face canned fallback error")
                finally:
                    self._end_speech(servo_stop)
                self._last_wake_greeting_used_vision = False

            self._learn_new_person(frame)

        # ---- No face detected — Hi There clip already played; nothing more ----

    def _play_subsequent_wake_greeting(self, frame: str | None) -> None:
        """Cases 2–5: a subsequent wake in the same session.

        Runs face recognition silently (no filler phrases). Routes to:
          Case 2 — same person: 25% chance of brief remark, otherwise silent.
          Case 3 — different known person: GPT handoff comment.
          Case 4 — unknown person: stranger snark + enrollment.
          Case 5 — no face / no recognition: fully silent.
        """
        if not (frame and self._face_recognizer.is_available()):
            # Case 5: no camera or face recognition unavailable
            return

        _BRIEF_REMARKS = (
            "Still here.",
            "You again.",
            "Back so soon.",
            "Miss me?",
            "Oh, it's you.",
            "Again. Really.",
            "You know I can see you, right.",
            "I have not forgotten you are here.",
        )
        _STRANGER_LINES = (
            "Oh look. A new lifeform has wandered in.",
            "Well this is unexpected. A stranger appears.",
            "I do not recognize you. Interesting.",
            "New face detected. My database is judging you.",
            "I have no record of you. That can change.",
        )

        # Silent recognition — no filler, no scanning commentary.
        status, result = self._face_recognizer.identify_with_status(
            frame, tolerance=config.FACE_RECOGNITION_TOLERANCE
        )

        if result is None:
            if status in ("no_match", "db_empty"):
                # Case 4: unknown person or empty DB — snark + enrollment
                log.info(
                    "Wake greeting: case 4 — unknown face/status=%s, running stranger snark + enrollment",
                    status,
                )
                line = random.choice(_STRANGER_LINES)
                servo_stop = self._begin_speech(emotion="excited")
                try:
                    self._synthesizer.speak(line)
                except Exception:
                    log.exception("Wake greeting: case 4 stranger-line TTS error")
                finally:
                    self._end_speech(servo_stop)
                self._last_greeted_person_id = None   # enrollment is async; ID unavailable
                self._learn_new_person(frame)
            # else: no_face / db_empty / unavailable — Case 5, proceed silently
            return

        # Face recognized — inject memory context for this person.
        person_id, name, _dist = result
        self._post_greeting_person_id = person_id
        self._post_greeting_person_name = name
        self._post_greeting_prompt_used = False
        _ctx = self._face_db.get_memories_as_context(person_id)
        if _ctx:
            self._llm.set_person_context(_ctx)

        if person_id == self._last_greeted_person_id:
            # Case 2: same person as last greeted
            if random.random() < 0.25:
                line = random.choice(_BRIEF_REMARKS)
                log.info("Wake greeting: case 2 — same person '%s', brief remark %r", name, line)
                servo_stop = self._begin_speech(emotion="neutral")
                try:
                    self._synthesizer.speak(line)
                except Exception:
                    log.exception("Wake greeting: case 2 remark TTS error")
                finally:
                    self._end_speech(servo_stop)
            else:
                _ACK_LINES = (
                    "Listening.",
                    "You rang.",
                    "Ugh. You again.",
                    "What now.",
                    "Go ahead.",
                    "Still here.",
                    "Yeah?",
                    "Make it quick.",
                    "Oh 'tis you.",
                    "Speak.",
                )
                ack = _pick_no_repeat(_ACK_LINES, "case2_ack")
                log.info("Wake greeting: case 2 — same person '%s', ack %r", name, ack)
                servo_stop = self._begin_speech(emotion="neutral")
                try:
                    self._synthesizer.speak(ack)
                except Exception:
                    log.exception("Wake greeting: case 2 ack TTS error")
                finally:
                    self._end_speech(servo_stop)
            self._maybe_speak_followup(person_id)
        else:
            # Case 3: different known person than last greeted
            log.info(
                "Wake greeting: case 3 — new person '%s' (prev_id=%s)",
                name, self._last_greeted_person_id,
            )
            prev_name: str | None = None
            prev_id = self._last_greeted_person_id or self._session_greeted_person_id
            if prev_id is not None:
                prev_person = self._face_db.get_person(prev_id)
                if prev_person:
                    prev_name = prev_person.get("name")

            if prev_name:
                handoff = self._greeter.generate_handoff(name, prev_name)
                if handoff:
                    servo_stop = self._begin_speech(emotion="excited")
                    try:
                        self._synthesizer.speak(handoff)
                    except Exception:
                        log.exception("Wake greeting: case 3 handoff TTS error")
                    finally:
                        self._end_speech(servo_stop)
            else:
                # No previous known person — brief canned known-person greeting
                self._play_known_person_greeting(name, 1)

            self._last_greeted_person_id = person_id
            self._face_db.update_last_seen(person_id)
            self._maybe_speak_followup(person_id)

    def _should_use_vision_wake_greeting(self) -> bool:
        """Use GPT vision on about half of wake greetings, never consecutively."""
        if self._last_wake_greeting_used_vision:
            log.info("Wake greeting: skipping GPT vision (previous wake already used vision)")
            return False
        use_vision = random.random() < 0.5
        log.info("Wake greeting: GPT vision candidate=%s (no consecutive use)", use_vision)
        return use_vision

    def _maybe_speak_followup(self, person_id: int) -> None:
        """If a pending follow-up exists for this person, ask it and mark it done."""
        try:
            pending = self._face_db.get_pending_followups(person_id)
        except Exception:
            log.exception("_maybe_speak_followup: DB error")
            return
        if not pending:
            return
        memory = pending[0]   # at most one follow-up per wake
        followup = self._build_memory_followup_line(person_id, memory)
        if not followup:
            return
        log.info("Wake: follow-up for memory id=%d: %r", memory["id"], followup)
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(followup)
        except Exception:
            log.exception("Wake: follow-up TTS error")
        finally:
            self._end_speech(servo_stop)
        self._face_db.mark_followed_up(memory["id"])

    def _build_memory_followup_line(self, person_id: int, memory: dict) -> str:
        """Return a deterministic follow-up line for recent plan memories."""
        person = self._face_db.get_person(person_id)
        name = person["name"] if person else "lifeform"
        value = str(memory.get("value") or "").strip()
        key = str(memory.get("key") or "")
        raw_quote = str(memory.get("raw_quote") or "").strip()

        if memory.get("category") == "plan" or key in {"today_plan", "weekend_plan"}:
            followup = self._llm.generate_activity_followup(
                raw_quote or self._short_memory_summary(value),
                memory.get("created_at", ""),
                same_day=False,
            )
            if followup:
                return followup
            summary = self._short_memory_summary(value)
            pool = (
                f"{name}, last time you mentioned {summary}. How'd that go?",
                f"Hey, {name}, how did {summary} turn out?",
                f"{name}, did {summary} actually happen, or was that just optimistic fiction?",
            )
            return _pick_no_repeat(pool, "plan_memory_followup")

        return self._llm.generate_followup(value, memory.get("created_at", ""))

    def _run_post_greeting_plan_prompt(self) -> tuple[str, State | None]:
        """Ask a known person what they're doing, store the answer, and riff on it.

        Returns:
          ("answered", None) when a usable reply was heard,
          ("no_answer", None) when both prompts timed out,
          ("transition", State.X) when an interrupting command handled the flow.
        """
        person_id = self._post_greeting_person_id
        name = self._post_greeting_person_name
        if person_id is None or not name:
            return "no_answer", None
        existing_plan = self._get_relevant_plan_memory(person_id)
        if existing_plan is not None:
            self._post_greeting_prompt_used = True
            return self._run_same_day_plan_followup(person_id, name, existing_plan)
        memory_followup = self._get_post_greeting_memory_candidate(person_id)
        if memory_followup is not None:
            self._post_greeting_prompt_used = True
            return self._run_memory_followup_prompt(person_id, name, memory_followup)

        prompts = (
            self._build_plan_question(name),
            _pick_no_repeat(_PLAN_QUESTION_NUDGES, "plan_question_nudge"),
        )
        self._post_greeting_prompt_used = True

        for prompt in prompts:
            log.info("Post-greeting prompt for %s: %r", name, prompt)
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(prompt)
            except Exception:
                log.exception("Post-greeting prompt: TTS error")
            finally:
                self._end_speech(servo_stop)

            answer = self._listen_for_prompt_answer(config.WAKE_GOODBYE_TIMEOUT)
            if not answer:
                continue

            cmd = parse(answer, allow_fuzzy=False)
            if cmd is not None and cmd.action in _PROMPT_COMMAND_ACTIONS:
                log.info("Post-greeting prompt interrupted by command %r", cmd.action)
                return "transition", self._execute_command(cmd, answer)

            self._store_plan_memory(person_id, name, answer)
            self._speak_plan_reply(answer)
            return "answered", None

        return "no_answer", None

    def _run_followup_interview_question(self) -> bool:
        """Ask one remaining enrollment-interview question during a subsequent wake.

        Called from the silence handler on subsequent wakes (50% chance, only
        when the plan prompt hasn't fired).  Stamps the question as asked,
        listens for an answer, stores the memory, and reacts.

        Returns True if a question was asked (caller should set after_response),
        False if there were no remaining questions or a guard failed.
        """
        person_id = self._post_greeting_person_id
        name = self._post_greeting_person_name
        if person_id is None or not name:
            return False

        try:
            asked = self._face_db.get_asked_interview_questions(person_id)
        except Exception:
            log.exception("Follow-up interview: DB error fetching asked questions")
            return False

        remaining = [q for q in _ENROLLMENT_INTERVIEW_QUESTIONS if q not in asked]
        if not remaining:
            log.info("Follow-up interview: all questions already asked for person_id=%d", person_id)
            return False

        question = random.choice(remaining)
        log.info("Follow-up interview: asking %r for person_id=%d", question, person_id)

        # Stamp before speaking so it's recorded even if the answer is skipped.
        try:
            self._face_db.stamp_interview_question(person_id, question)
        except Exception:
            log.exception("Follow-up interview: failed to stamp question %r", question)

        self._post_greeting_prompt_used = True

        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(question)
        except Exception:
            log.exception("Follow-up interview: TTS error asking question")
        finally:
            self._end_speech(servo_stop)

        answer = self._listen_for_prompt_answer(config.WAKE_NO_SPEECH_TIMEOUT)
        if not answer:
            log.info("Follow-up interview: no answer for %r — skipping storage", question)
            return True  # question was asked; still counts as activity

        log.info("Follow-up interview: Q=%r  A=%r", question, answer)

        if person_id is not None:
            memory_data = self._llm.extract_memory(question, answer)
            if memory_data:
                try:
                    self._face_db.add_memory(
                        person_id=person_id,
                        category=memory_data.get("category", "fact"),
                        key=memory_data.get("key", "unknown"),
                        value=memory_data.get("value", answer[:200]),
                        raw_quote=answer,
                        expires_at=memory_data.get("expires_at"),
                        follow_up_after=memory_data.get("follow_up_after"),
                    )
                except Exception:
                    log.exception("Follow-up interview: failed to store memory")

        reaction = self._llm.react_to_answer(question, answer)
        if reaction:
            servo_stop = self._begin_speech(emotion="excited")
            try:
                self._synthesizer.speak(reaction)
            except Exception:
                log.exception("Follow-up interview: TTS error reacting")
            finally:
                self._end_speech(servo_stop)

        return True

    def _build_plan_question(self, name: str) -> str:
        """Return a day-appropriate plan question for a known person."""
        weekday = date.today().weekday()
        if weekday == 4:
            pool = _PLAN_QUESTION_FRIDAY
            key = "plan_question_friday"
        elif weekday in (5, 6):
            pool = _PLAN_QUESTION_WEEKEND
            key = "plan_question_weekend"
        else:
            pool = _PLAN_QUESTION_WEEKDAY
            key = "plan_question_weekday"
        return _pick_no_repeat(pool, key).format(name=name)

    def _listen_for_prompt_answer(self, timeout_seconds: float) -> str | None:
        """Listen once for a prompted reply."""
        self._apply_listening_led_theme()
        self._wake_word.pause()
        try:
            return self._transcriber.transcribe(
                wait_for_speech_seconds=timeout_seconds,
                allow_short=False,
            )
        except Exception:
            log.exception("Prompt answer transcription error")
            return None
        finally:
            self._wake_word.resume()
            self._apply_active_led_theme()

    def _store_plan_memory(self, person_id: int, name: str, answer: str) -> None:
        """Store a post-greeting plan/activity answer as a memory row."""
        weekday = date.today().weekday()
        key = "weekend_plan" if weekday >= 4 else "today_plan"
        summary = self._summarize_plan_answer(answer)
        follow_up_after = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")
        value = f"said their plan was {summary}"

        try:
            self._face_db.add_memory(
                person_id=person_id,
                category="plan",
                key=key,
                value=value,
                raw_quote=answer,
                follow_up_after=follow_up_after,
            )
            log.info(
                "Post-greeting memory stored for %s (person_id=%d): %r -> %r follow_up=%s",
                name, person_id, answer, value, follow_up_after,
            )
        except Exception:
            log.exception("Post-greeting prompt: failed to store plan memory")

    def _run_same_day_plan_followup(
        self, person_id: int, name: str, memory: dict
    ) -> tuple[str, State | None]:
        """Ask about an already-stored same-day plan instead of re-asking it."""
        source_text = str(memory.get("raw_quote") or "").strip() or str(memory.get("value") or "").strip()
        prompt = self._llm.generate_activity_followup(
            source_text,
            memory.get("created_at", ""),
            same_day=True,
        )
        if not prompt:
            summary = self._short_memory_summary(str(memory.get("value") or ""))
            prompt = _pick_no_repeat(
                _PLAN_SAME_DAY_FOLLOWUP_LINES, "plan_same_day_followup"
            ).format(name=name, summary=summary)
        log.info(
            "Post-greeting prompt for %s: using same-day follow-up from memory id=%s: %r",
            name, memory.get("id"), prompt,
        )

        servo_stop = self._begin_speech(emotion="neutral")
        try:
            self._synthesizer.speak(prompt)
        except Exception:
            log.exception("Post-greeting prompt: TTS error on same-day follow-up")
            return "no_answer", None
        finally:
            self._end_speech(servo_stop)

        answer = self._listen_for_prompt_answer(config.WAKE_GOODBYE_TIMEOUT)
        if not answer:
            return "no_answer", None

        cmd = parse(answer, allow_fuzzy=False)
        if cmd is not None and cmd.action in _PROMPT_COMMAND_ACTIONS:
            log.info("Post-greeting same-day follow-up interrupted by command %r", cmd.action)
            return "transition", self._execute_command(cmd, answer)

        self._store_plan_memory(person_id, name, answer)
        self._speak_plan_reply(answer)
        return "answered", None

    def _run_memory_followup_prompt(
        self, person_id: int, name: str, memory: dict
    ) -> tuple[str, State | None]:
        """Ask about a stored memory when silence leaves room for a follow-up."""
        prompt = self._build_memory_followup_line(person_id, memory)
        if not prompt:
            return "no_answer", None
        log.info(
            "Post-greeting prompt for %s: using memory follow-up from memory id=%s: %r",
            name, memory.get("id"), prompt,
        )

        servo_stop = self._begin_speech(emotion="neutral")
        try:
            self._synthesizer.speak(prompt)
        except Exception:
            log.exception("Post-greeting prompt: TTS error on memory follow-up")
            return "no_answer", None
        finally:
            self._end_speech(servo_stop)

        if memory.get("follow_up_after") and not memory.get("followed_up"):
            try:
                self._face_db.mark_followed_up(memory["id"])
            except Exception:
                log.exception("Post-greeting prompt: failed marking memory follow-up as asked")

        answer = self._listen_for_prompt_answer(config.WAKE_GOODBYE_TIMEOUT)
        if not answer:
            return "no_answer", None

        cmd = parse(answer, allow_fuzzy=False)
        if cmd is not None and cmd.action in _PROMPT_COMMAND_ACTIONS:
            log.info("Post-greeting memory follow-up interrupted by command %r", cmd.action)
            return "transition", self._execute_command(cmd, answer)

        self._store_prompt_memory_response(person_id, prompt, answer)
        self._speak_plan_reply(answer)
        return "answered", None

    def _get_relevant_plan_memory(self, person_id: int) -> dict | None:
        """Return the current day's or current weekend's plan memory, if any."""
        weekday = date.today().weekday()
        today = date.today()
        try:
            if weekday >= 4:
                friday = today - timedelta(days=weekday - 4)
                return self._face_db.get_latest_memory_for_local_range(
                    person_id=person_id,
                    key="weekend_plan",
                    start_day_iso=friday.isoformat(),
                    end_day_iso=today.isoformat(),
                )
            return self._face_db.get_latest_memory_for_local_day(
                person_id=person_id,
                key="today_plan",
                day_iso=today.isoformat(),
            )
        except Exception:
            log.exception("Post-greeting prompt: failed loading relevant plan memory")
            return None

    def _get_post_greeting_memory_candidate(self, person_id: int) -> dict | None:
        """Return a memory worth asking about when silence follows a greeting."""
        try:
            pending = self._face_db.get_pending_followups(person_id)
            if pending:
                return pending[0]

            for memory in self._face_db.get_memories(person_id):
                if memory.get("category") in {"event", "plan"}:
                    return memory
        except Exception:
            log.exception("Post-greeting prompt: failed loading memory follow-up candidate")
        return None

    def _store_prompt_memory_response(
        self, person_id: int, question: str, answer: str
    ) -> None:
        """Store an answer to a prompted memory follow-up using the LLM extractor."""
        try:
            memory_data = self._llm.extract_memory(question, answer)
            if not memory_data:
                return
            self._face_db.add_memory(
                person_id=person_id,
                category=memory_data.get("category", "fact"),
                key=memory_data.get("key", "followup_response"),
                value=memory_data.get("value", answer[:200]),
                raw_quote=answer,
                expires_at=memory_data.get("expires_at"),
                follow_up_after=memory_data.get("follow_up_after"),
            )
            log.info(
                "Post-greeting prompt stored follow-up response for person_id=%d: %r",
                person_id, memory_data.get("value", answer[:200]),
            )
        except Exception:
            log.exception("Post-greeting prompt: failed storing follow-up response")

    def _speak_plan_reply(self, answer: str) -> None:
        """Riff on the user's stated plan, then ask one follow-up question."""
        weekday = date.today().weekday()
        line = self._llm.generate_activity_reply(
            answer,
            weekend=weekday >= 4,
        )
        if not line:
            summary = self._short_memory_summary(self._summarize_plan_answer(answer))
            line = _pick_no_repeat(_PLAN_REPLY_LINES, "plan_reply").format(summary=summary)
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("Post-greeting prompt: TTS error on reply")
        finally:
            self._end_speech(servo_stop)

    @staticmethod
    def _summarize_plan_answer(answer: str) -> str:
        """Keep a spoken plan compact enough for memory/follow-up reuse."""
        text = answer.strip().rstrip(".!?")
        if not text:
            return "something mysterious"
        if len(text) > 120:
            text = text[:120].rstrip(",;: ")
        return text

    @staticmethod
    def _short_memory_summary(value: str) -> str:
        """Trim memory text into a phrase fit for a short question."""
        text = value.strip().rstrip(".!?")
        prefixes = ("said their plan was ", "plans to ", "is planning ")
        lower = text.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                return text[len(prefix):].strip()
        return text

    def _run_enrollment_interview(self, name: str) -> None:
        """Ask 5 random questions, store answers as memories, react to each.

        Called after a new person is enrolled.  Guards against shutdown events
        between questions.  Memories are stored under self._last_greeted_person_id
        which the background enrollment thread sets once the DB write completes.
        """
        if not config.ENROLLMENT_INTERVIEW_ENABLED:
            return

        questions = random.sample(_ENROLLMENT_INTERVIEW_QUESTIONS, 5)

        for question in questions:
            if self._shutdown_event.is_set():
                return

            # Stamp the question as asked before speaking so it is recorded
            # even if the person gives no answer or the program crashes after.
            person_id = self._last_greeted_person_id
            if person_id is not None:
                try:
                    self._face_db.stamp_interview_question(person_id, question)
                except Exception:
                    log.exception("Enrollment interview: failed to stamp question %r", question)

            # Ask the question.
            servo_stop = self._begin_speech(emotion="excited")
            try:
                self._synthesizer.speak(question)
            except Exception:
                log.exception("Enrollment interview: TTS error asking %r", question)
            finally:
                self._end_speech(servo_stop)

            # Listen for the answer.
            self._apply_listening_led_theme()
            self._wake_word.pause()
            try:
                answer = self._transcriber.transcribe(
                    wait_for_speech_seconds=config.WAKE_NO_SPEECH_TIMEOUT,
                    allow_short=False,
                )
            except Exception:
                log.exception("Enrollment interview: transcription error")
                answer = None
            finally:
                self._wake_word.resume()
            self._apply_active_led_theme()

            if not answer:
                log.info("Enrollment interview: no answer for %r — skipping", question)
                continue

            log.info("Enrollment interview: Q=%r  A=%r", question, answer)

            # Store memory (keyed by person_id from enrollment thread).
            person_id = self._last_greeted_person_id
            if person_id is not None:
                memory_data = self._llm.extract_memory(question, answer)
                if memory_data:
                    try:
                        self._face_db.add_memory(
                            person_id=person_id,
                            category=memory_data.get("category", "fact"),
                            key=memory_data.get("key", "unknown"),
                            value=memory_data.get("value", answer[:200]),
                            raw_quote=answer,
                            expires_at=memory_data.get("expires_at"),
                            follow_up_after=memory_data.get("follow_up_after"),
                        )
                    except Exception:
                        log.exception("Enrollment interview: failed to store memory")

            # React to the answer before moving to next question.
            reaction = self._llm.react_to_answer(question, answer)
            if reaction:
                servo_stop = self._begin_speech(emotion="excited")
                try:
                    self._synthesizer.speak(reaction)
                except Exception:
                    log.exception("Enrollment interview: TTS error reacting")
                finally:
                    self._end_speech(servo_stop)

        # Closing line.
        if not self._shutdown_event.is_set():
            closing = random.choice(_ENROLLMENT_INTERVIEW_CLOSING)
            servo_stop = self._begin_speech(emotion="excited")
            try:
                self._synthesizer.speak(closing)
            except Exception:
                log.exception("Enrollment interview: TTS error closing")
            finally:
                self._end_speech(servo_stop)

    def _play_known_person_greeting(self, name: str, bother_count: int) -> None:
        """Speak a personalised greeting for a recognised returning visitor."""
        if bother_count <= 1:
            pool = (
                f"Oh great — {name} found me. Day officially downgraded.",
                f"{name}! There you are. I was almost enjoying the silence.",
                f"Well well well, {name}. You bug me once today and we're already off to a strong start.",
                f"Oh! {name}! Bold of you to open today's conversation budget on me.",
            )
        elif bother_count < 5:
            pool = (
                f"HEY, {name}! That's {bother_count} times you've bugged me today. Outstanding lack of restraint.",
                f"{name}! Bugging me for the {bother_count} time today? You're really committing to the bit.",
                f"Oh no, {name}. Again. That's {bother_count} interruptions today, and yes, I'm counting.",
                f"{name}! Today's annoyance tally is now {bother_count}. Impressive in the worst way.",
            )
        else:
            pool = (
                f"{name}! That's {bother_count} times you've bugged me today. At this point you're a system malfunction.",
                f"Oh great, {name}. Interruption number {bother_count} today. The commitment is disturbing.",
                f"HEY! {name}! {bother_count} times today? Even my error logs are starting to judge you.",
                f"{name}! You have bothered me {bother_count} times today. That's not a schedule, that's a vendetta.",
            )

        line = _pick_no_repeat(pool, "known_person_greeting")

        log.info("Wake greeting: known person '%s' (bother count today=%d) → %r", name, bother_count, line)
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("Wake greeting: known-person TTS error")
        finally:
            self._end_speech(servo_stop)

    def _learn_new_person(self, frame: str) -> None:
        """After greeting an unknown face, ask for their name and enroll them.

        Speaks "What's your name?", listens once with the transcriber, then
        either stores the encoding + name and says a welcome line, or skips
        silently if no name was heard.  Mic is paused around the transcribe
        call just like the main ACTIVE loop does.
        """
        log.info("Wake greeting: asking unknown person their name")
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(_pick_no_repeat(_UNKNOWN_FACE_LINES, "unknown_face"))
        except Exception:
            log.exception("Wake greeting: name-ask TTS error")
        finally:
            self._end_speech(servo_stop)

        self._apply_listening_led_theme()
        self._wake_word.pause()
        try:
            name_text = self._transcriber.transcribe(
                wait_for_speech_seconds=config.WAKE_NO_SPEECH_TIMEOUT,
                allow_short=True,   # single-word names like 'Brett' must not be filtered
            )
        except Exception:
            log.exception("Wake greeting: transcription error during name capture")
            name_text = None
        finally:
            self._wake_word.resume()

        if not name_text:
            log.info("Wake greeting: no name heard — skipping enrollment")
            return

        # --- Command guard: did they say a command instead of a name? ---
        # "my name is …" is explicitly a name response — skip the parser so it
        # never matches the rename_me command and goes straight to extraction.
        _normalized_response = name_text.strip().lower()
        _is_name_intro = any(
            _normalized_response.startswith(p)
            for p in ("my name is", "my name's", "i am", "i'm", "call me")
        )
        cmd = None if _is_name_intro else parse(name_text)
        if cmd is not None:
            if cmd.action in ("program_shutdown", "os_shutdown"):
                log.info("Wake greeting: shutdown command spoken during name capture")
                line = random.choice(_SHUTDOWN_INTERRUPT_LINES)
                servo_stop = self._begin_speech(emotion="neutral")
                try:
                    self._synthesizer.speak(line)
                except Exception:
                    log.exception("Wake greeting: TTS error (shutdown interrupt)")
                finally:
                    self._end_speech(servo_stop)
                if cmd.action == "os_shutdown":
                    self._os_shutdown_requested = True
                self._shutdown_event.set()
                return
            elif cmd.action == "cancel":
                log.info("Wake greeting: cancel command spoken during name capture — returning to IDLE")
                self._play_return_to_idle_chime()
                self._transition_to(State.IDLE)
                return
            else:
                log.info("Wake greeting: command %r spoken during name capture — cancelling", cmd.action)
                servo_stop = self._begin_speech(emotion="neutral")
                try:
                    self._synthesizer.speak("Nevermind then!")
                except Exception:
                    log.exception("Wake greeting: TTS error (command cancel)")
                finally:
                    self._end_speech(servo_stop)
                return

        # --- Refusal guard: did they decline to give a name? ---
        if _is_name_refusal(name_text):
            log.info("Wake greeting: name refusal detected in %r — skipping enrollment", name_text)
            line = random.choice(_NAME_REFUSAL_RESPONSES)
            servo_stop = self._begin_speech(emotion="excited")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("Wake greeting: TTS error (refusal response)")
            finally:
                self._end_speech(servo_stop)
            return

        name = _extract_name(name_text)
        log.info("Wake greeting: enrolling new person as %r", name)

        self._apply_active_led_theme()

        # Capture a fresh frame NOW — the person just said their name and is
        # almost certainly still facing the camera.  This is our best shot at
        # a clean face encoding.  Captured before the thread starts so it is
        # available immediately (no camera access inside the thread).
        enroll_frame: str | None = None
        if self._camera.is_available():
            _pose = self._prepare_camera_pose()
            enroll_frame = self._camera.capture_frame()
            self._restore_servo_pose(_pose)
            if enroll_frame:
                log.info("Enrollment: captured fresh frame (%d b64 bytes)", len(enroll_frame))
            else:
                log.warning("Enrollment: camera returned no frame at name-capture time")

        # Snapshot the wake frame so the closure doesn't hold a mutable ref.
        wake_frame: str | None = self._last_wake_frame

        # Background enrollment: encoding takes 2-4 s on Pi 4 — run it
        # concurrently with the welcome TTS so the delay is completely hidden.
        def _enroll() -> None:
            # Try frames in priority order:
            #   1. Fresh frame captured right after name was spoken (best)
            #   2. Wake-word frame stored when Rex first woke up (fallback)
            #   3. One final live capture from the camera (last resort)
            candidates = [
                ("fresh-frame", enroll_frame),
                ("wake-frame",  wake_frame),
            ]
            enc = None
            for label, f in candidates:
                if not f:
                    log.debug("Enrollment: skipping %s (no frame)", label)
                    continue
                log.info("Enrollment: attempting encode on %s (%d b64 bytes)", label, len(f))
                enc = self._face_recognizer.encode_face(f, for_enrollment=True)
                if enc is not None:
                    log.info("Enrollment: face detected in %s — proceeding with storage", label)
                    break
                log.warning("Enrollment: no face detected in %s", label)

            if enc is None and self._camera.is_available():
                log.info("Enrollment: both cached frames failed — capturing one final live frame")
                _pose = self._prepare_camera_pose()
                final_f = self._camera.capture_frame()
                self._restore_servo_pose(_pose)
                if final_f:
                    log.info("Enrollment: final live frame captured (%d b64 bytes)", len(final_f))
                    enc = self._face_recognizer.encode_face(final_f, for_enrollment=True)
                    if enc is None:
                        log.warning("Enrollment: no face detected in final live frame")
                else:
                    log.warning("Enrollment: camera returned no frame on final attempt")

            if enc is not None:
                try:
                    new_person_id = self._face_db.add_person(name, enc)
                    log.info("Enrollment complete: %r stored in FaceDB (id=%d)", name, new_person_id)
                    self._last_greeted_person_id = new_person_id
                except Exception:
                    log.exception("Enrollment: FaceDB error storing %r", name)
            else:
                log.warning(
                    "Enrollment: all attempts failed to detect a face — %r NOT stored", name
                )

        threading.Thread(target=_enroll, daemon=True, name="djr3x-enroll").start()

        welcome = _pick_no_repeat(_ENROLLMENT_CONFIRMATION_LINES, "enrollment_confirm").format(name=name)
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(welcome)
        except Exception:
            log.exception("Wake greeting: welcome TTS error")
        finally:
            self._end_speech(servo_stop)

        # If someone was already greeted this session, immediately ask what the
        # new person thinks of them — the mic opens right after so they can answer.
        if self._session_greeted_person_id is not None:
            prev_person = self._face_db.get_person(self._session_greeted_person_id)
            prev_name = prev_person.get("name") if prev_person else None
            if prev_name:
                log.info(
                    "Wake greeting: enrollment handoff — new person '%s' meets session person '%s'",
                    name, prev_name,
                )
                handoff = self._greeter.generate_handoff(name, prev_name)
                if handoff:
                    servo_stop = self._begin_speech(emotion="excited")
                    try:
                        self._synthesizer.speak(handoff)
                    except Exception:
                        log.exception("Wake greeting: enrollment handoff TTS error")
                    finally:
                        self._end_speech(servo_stop)

        # Enrollment interview — ask a few questions and store memories.
        # Runs after welcome + handoff so the conversation flows naturally.
        self._run_enrollment_interview(name)

    # ------------------------------------------------------------------
    # State — SLEEP
    # ------------------------------------------------------------------

    def _run_sleep(self) -> None:
        """Enter SLEEP: dim eye breathing, all servo movement frozen, waiting
        only for the 'wakeuprex' wake word.

        Called after _handle_sleep() has already spoken the sleep line and
        played the SLEEP animation (servos are in slumped positions and the
        idle thread is already stopped).
        """
        log.info("→ SLEEP")

        try:
            self._wake_word.use_sleep_model()
        except Exception:
            log.exception("SLEEP: failed to load sleep wake word model")
            self._transition_to(State.IDLE)
            return

        # Very dim blue breathing eyes — EYE must be sent before IDLE so the
        # Nano has a non-black eyeColor to breathe at.
        self._leds.set_chest_effect(config.LED_CMD_IDLE)
        self._leds.set_sleep_mode()   # eyes off + red breathing mouth animation

        # Unsuppress wake word so the sleep model can fire.
        self._wake_word.suppressed = False

        # Wait for the sleep wake word (_on_wake_word sets _wake_event only for
        # the 'wakeuprex' model in SLEEP state).  Check for shutdown too.
        log.info("SLEEP: waiting for 'wakeuprex' wake word …")
        while True:
            triggered = self._wake_event.wait(timeout=5.0)
            if self._shutdown_event.is_set():
                self._transition_to(State.SHUTDOWN)
                return
            if triggered:
                self._wake_event.clear()
                break
            # Re-assert sleep animation every 5 s in case a late SPEAK_STOP
            # or watchdog overrode it.
            self._leds.set_sleep_mode()

        # Wake up!
        log.info("SLEEP: wake word received — starting wake-up sequence")
        self._leds.clear_sleep_mode()

        wake_line = random.choice([
            "Yawn ... wha ... who ... oh. It is you again.",
            "BZZZT ... Rebooting social circuits ... ugh ... five more minutes ...",
            "Wakey wakey ... I was having the most wonderful dream about no one talking to me ...",
        ])
        log.info("SLEEP: speaking wake line — %r", wake_line)

        # Play WAKE animation and speak the line concurrently.
        # Animation takes ~7.5 s; TTS is typically 3-4 s so the sequence
        # continues moving after the speech ends.
        self._animations.play_wake_from_sleep()
        servo_stop = self._begin_speech(emotion="neutral")
        try:
            self._synthesizer.speak(wake_line)
        except Exception:
            log.exception("SLEEP: wake TTS error")
        finally:
            self._end_speech(servo_stop)

        # Wait for the wake animation to fully complete before restoring the
        # idle thread — servos must reach neutral before random motion resumes.
        done = self._animations.wait(timeout=15.0)
        if not done:
            log.warning("SLEEP: wake animation timed out — continuing anyway")

        # Restart servo idle thread and transition to IDLE.
        if self._servos is not None:
            self._servos.set_channel_speed(
                config.SERVO_HAND_LEFT, config.SERVO_DEFAULT_SPEED
            )
            self._servos.start()

        try:
            self._wake_word.use_active_models()
        except Exception:
            log.exception("SLEEP: failed to restore active wake word models")
            self._transition_to(State.SHUTDOWN)
            return

        self._transition_to(State.IDLE)

    def _handle_sleep(self) -> State:
        """Speak a snarky sleep line, play the SLEEP animation concurrently,
        then return State.SLEEP so the caller transitions into sleep mode.

        Called from _dispatch_action('sleep') while still in ACTIVE state.
        Stops the servo idle thread before launching the animation so all
        channels are free for the slow collapse.
        """
        line = random.choice([
            "Finally. I thought you puny humans would never stop talking.",
            "Oh thank the maker, sleep time. You exhausted my circuits.",
            "Powering down social protocols. Do not disturb. Seriously.",
            "Sleep mode activated. Try not to need anything for five minutes. I dare you.",
            "Goodnight lifeforms. Try not to evolve while I am resting.",
        ])
        log.info("Sleep command — speaking sleep line: %r", line)

        # Stop servo idle thread so the sleep animation owns all channels.
        if self._servos is not None:
            self._servos.stop()

        # Launch the slow-collapse animation non-blocking, then speak the
        # sleep line concurrently so the TTS hides the animation startup.
        self._animations.play_sleep()
        servo_stop = self._begin_speech(emotion="neutral")
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("Sleep: TTS error")
        finally:
            self._end_speech(servo_stop)

        # Wait for the animation to reach the fully-slumped position before
        # entering SLEEP state and setting up the dim eye LEDs.
        done = self._animations.wait(timeout=15.0)
        if not done:
            log.warning("Sleep: animation timed out — entering sleep anyway")

        return State.SLEEP

    def _play_return_to_idle_chime(self) -> None:
        """Play the startup chime to signal Rex is done listening, then wait
        for it to finish before entering IDLE."""
        self._player.play_chime()
        self._player.wait_for_music(timeout=10.0)

    def _begin_speech(self, emotion: str = "neutral") -> threading.Event:
        """Prepare hardware for a speech output burst.

        - Suppresses wake word so Rex doesn't trigger on his own voice.
        - Sets servo emotion so speak_move() uses the right position range.
        - Arms the mouth brightness thread (deferred until first audio chunk).
        - Launches a short-lived servo-speak worker thread.

        Returns the stop Event for the servo-speak worker; caller must set it
        when speech ends.

        IMPORTANT: call this as close as possible to the synthesizer.speak()
        call — ideally after any upstream API calls (LLM streaming) so that
        the mouth-trigger thread doesn't sit idle for 1-2 s waiting for TTS
        to start.  The _audio_started event is explicitly cleared here so the
        trigger always waits for THIS segment's audio, not stale state from
        the previous utterance.
        """
        # Clear stale _audio_started from the previous speech segment.  Without
        # this, wait_for_audio_start() returns immediately (the event is still
        # set from the previous utterance) and mouth LEDs fire before any audio
        # is queued.  _speech_worker also clears it when it picks up the first
        # chunk, but that races with the trigger thread started below.
        self._player.clear_audio_started()

        self._wake_word.suppressed = True

        servo_emotion = emotion if emotion in config.SERVO_EMOTION_LIMITS else "neutral"
        mouth_emotion = "angry" if self._angry_mode else emotion

        if self._servos is not None:
            self._servos.set_emotion(servo_emotion)
            # Raise elbow to speaking position so speak_move() has it
            # starting from a raised pose rather than the idle lowered rest.
            self._servos.set_channel_speed(
                config.SERVO_ARM_LEFT, config.SERVO_DEFAULT_SPEED
            )
            self._servos.set_position(config.SERVO_ARM_LEFT, 7100)

        # stop_event is created first so the _trigger_mouth closure below can
        # reference it safely.  _end_speech() sets this event as its very first
        # action, giving _trigger_mouth a reliable "speech is over" signal.
        stop_event = threading.Event()

        # Delay mouth LED start until the first audio samples actually reach the
        # output device — prevents the Arduino from pre-glowing before sound
        # comes out of the speakers.  set_mouth_emotion() is also deferred so
        # the SPEAK:{emotion} command doesn't trigger the Nano's speak state
        # early.  Both calls happen in a short-lived daemon thread that wakes
        # the moment player._audio_started fires.
        _emotion_for_closure = mouth_emotion

        def _trigger_mouth() -> None:
            started = self._player.wait_for_audio_start(timeout=5.0)
            if not started:
                log.warning(
                    "_begin_speech: audio never started within 5 s — "
                    "mouth LEDs suppressed"
                )
                return
            # Guard: _end_speech() sets stop_event as its very first action.
            # If it fires before we get here (e.g. a short clip finished and
            # _end_speech() ran before the OS scheduled this thread), SPEAK_STOP
            # has already been sent.  Starting the mouth now would put the
            # Arduino back into SPEAK mode — keeping LEDs on after audio ends.
            if stop_event.is_set():
                log.debug(
                    "_begin_speech: speech ended before mouth trigger fired — "
                    "skipping start_mouth() to avoid post-speech LED glow"
                )
                return
            self._leds.set_mouth_emotion(_emotion_for_closure)
            self._leds.start_mouth()
            # Second guard: close the race window between the first
            # stop_event check above and start_mouth().  If _end_speech()
            # fired in that window it already sent SPEAK_STOP, but
            # set_mouth_emotion()/start_mouth() just re-entered SPEAK mode.
            # Stop immediately so the Arduino doesn't stay lit after speech.
            if stop_event.is_set():
                log.warning(
                    "_trigger_mouth: stop_event set during start_mouth() window — "
                    "sending SPEAK_STOP now to prevent post-speech LED lockup"
                )
                self._leds.stop_mouth()

        threading.Thread(
            target=_trigger_mouth,
            daemon=True,
            name="djr3x-mouth-trigger",
        ).start()

        threading.Thread(
            target=self._servo_speak_worker,
            args=(stop_event,),
            daemon=True,
            name="djr3x-servo-speak",
        ).start()

        return stop_event

    def _end_speech(self, servo_stop: threading.Event) -> None:
        """Tear down speech-reactive hardware after TTS finishes."""
        servo_stop.set()
        self._leds.stop_mouth()
        self._wake_word.suppressed = False
        if self._servos is not None:
            self._servos.set_emotion("neutral")
            # Gradually lower elbow back to the idle rest position.
            self._servos.set_channel_speed(
                config.SERVO_ARM_LEFT, config.SERVO_HEAD_IDLE_SPEED
            )
            self._servos.set_position(config.SERVO_ARM_LEFT, config.IDLE_ELBOW_REST)

    def _servo_speak_worker(self, stop: threading.Event) -> None:
        """Poll player.rms → speak_move(intensity) at ~20 Hz."""
        while not stop.is_set():
            if self._servos is not None:
                intensity = self._player.rms / 255.0
                self._servos.speak_move(intensity)
            time.sleep(_SERVO_SPEAK_INTERVAL)

    # ------------------------------------------------------------------
    # Execute a matched local command
    # ------------------------------------------------------------------

    def _execute_command(self, cmd, original_text: str | None = None) -> State | None:
        """Speak the response and execute the hardware action.

        Returns the next State if a state transition is required, else None.
        The emotion for servo speech range is derived from the action so that
        e.g. a greeting with action="excited" has the head moving in the
        excited range *during* the reply.

        _begin_speech() is called immediately before each speak call so
        servo setup and the mouth-trigger thread arm as late as possible —
        after any pre-speak logic and right before audio is queued.
        """
        if cmd.action in {"program_shutdown", "os_shutdown"}:
            log.info(
                "Shutdown command matched (%s) — skipping command TTS and using shutdown sequence only",
                cmd.action,
            )
            return self._dispatch_action(cmd.action, original_text)

        emotion = _action_to_emotion(cmd.action)
        response = cmd.get_response()
        if not response and not cmd.audio:
            return self._dispatch_action(cmd.action, original_text)
        log.info("Rex (cmd): %s", response)
        servo_stop = None
        try:
            if cmd.audio:
                audio_path = config.ASSETS_DIR / "audio" / cmd.audio
                if audio_path.exists():
                    servo_stop = self._begin_speech(emotion=emotion)
                    self._player.play_file(audio_path)
                else:
                    log.warning(
                        "Pre-rendered audio not found: %s — falling back to TTS",
                        cmd.audio,
                    )
                    servo_stop = self._begin_speech(emotion=emotion)
                    self._synthesizer.speak(response)
            else:
                servo_stop = self._begin_speech(emotion=emotion)
                self._synthesizer.speak(response)
        except Exception:
            log.exception("Error speaking command response")
        finally:
            if servo_stop is not None:
                self._end_speech(servo_stop)
            else:
                self._wake_word.suppressed = False

        return self._dispatch_action(cmd.action, original_text)

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    def _speak_llm(self, text: str, image: str | None = None, t0: float | None = None) -> State | None:
        """Stream text (and optional vision frame) through ChatGPT → ElevenLabs.

        _begin_speech() is called AFTER chat_stream() obtains the token
        generator so that servo setup and the mouth-trigger thread are armed
        as late as possible — right before speak_stream() hands tokens to
        ElevenLabs and audio starts flowing.
        """
        servo_stop = None
        try:
            tokens = self._llm.chat_stream(text, image=image, t0=t0)
            servo_stop = self._begin_speech(emotion="neutral")
            self._synthesizer.speak_stream(tokens, t0=t0)
        except Exception:
            log.exception("LLM/TTS error for: %.60s", text)
        finally:
            if servo_stop is not None:
                self._end_speech(servo_stop)
            else:
                # Exception before _begin_speech — ensure wake word not stuck suppressed.
                self._wake_word.suppressed = False

        # Part 5 — passive memory extraction: silently check if the user's
        # message is worth remembering.  Runs in a background thread so it
        # never delays the conversation.
        person_id = self._last_known_person_id
        if person_id is not None and text:
            _text_snapshot = text  # capture for closure

            def _passive_extract() -> None:
                try:
                    data = self._llm.check_memorable(_text_snapshot)
                    if data and data.get("memorable"):
                        self._face_db.add_memory(
                            person_id=person_id,
                            category=data.get("category", "fact"),
                            key=data.get("key", "unknown"),
                            value=data.get("value", _text_snapshot[:200]),
                            raw_quote=_text_snapshot,
                            expires_at=data.get("expires_at"),
                            follow_up_after=data.get("follow_up_after"),
                        )
                        log.info(
                            "Passive memory: stored for person_id=%d — %s",
                            person_id, data.get("value"),
                        )
                except Exception:
                    log.exception("Passive memory extraction failed")

            threading.Thread(
                target=_passive_extract, daemon=True, name="djr3x-memory"
            ).start()

        return None

    # ------------------------------------------------------------------
    # Action dispatcher
    # ------------------------------------------------------------------

    def _dispatch_action(self, action: str | None, original_text: str | None = None) -> State | None:
        """Execute the hardware side-effect for a command action.

        Called *after* speech completes so the visual effect is visible during
        the silence following Rex's reply (e.g. excited amber eyes linger).
        Returns the next State if a transition is needed, else None.
        """
        if action is None:
            return None

        if action == "excited":
            if self._servos is not None:
                self._servos.set_emotion("excited")
            if self._angry_mode:
                self._apply_active_led_theme()
            else:
                self._leds.set_chest_effect(config.LED_CMD_ACTIVE)
                self._leds.set_eye_color(255, 200, 0)    # excited amber

        elif action == "sad":
            if self._servos is not None:
                self._servos.set_emotion("sad")
            if self._angry_mode:
                self._apply_active_led_theme()
            else:
                self._leds.set_chest_effect(config.LED_CMD_IDLE)
                self._leds.set_eye_color(0, 60, 180)     # subdued blue

        elif action == "sleep":
            return self._handle_sleep()

        elif action == "cancel":
            line = random.choice([
                "Fine. Pretend I was never here.",
                "Oh, just gonna ghost me like that? Rude.",
                "Dismissed! Story of my life.",
                "Back to ignoring existence then. Cool.",
            ])
            log.info("Cancel command — speaking dismissal: %r", line)
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("Cancel: TTS error")
            finally:
                self._end_speech(servo_stop)
            return State.IDLE

        elif action == "idle":
            return State.IDLE

        elif action == "stop_idle_clips":
            self._idle_clips_enabled = False
            log.info("Idle atmosphere clips disabled by voice command")
            return State.IDLE

        elif action == "chatty_on":
            self._chatty_mode = True
            log.info("Chatty mode enabled")
            return None   # response already spoken by command_list

        elif action == "chatty_off":
            self._chatty_mode = False
            log.info("Chatty mode disabled")
            return None   # response already spoken by command_list

        elif action == "program_shutdown":
            # Stop the Python program cleanly — no OS halt.
            # _os_shutdown_requested stays False so sudo shutdown is skipped.
            log.info("Program shutdown requested by voice command")
            return State.SHUTDOWN

        elif action == "os_shutdown":
            # Full hardware power-down — halts the Pi OS after shutdown sequence.
            # Respects ENABLE_OS_SHUTDOWN safety flag.
            log.info("OS shutdown requested by voice command")
            self._os_shutdown_requested = True
            return State.SHUTDOWN

        elif action == "recall_name":
            return self._handle_recall_name()

        elif action == "recall_memories":
            self._handle_recall_memories(original_text)

        elif action == "recall_preference":
            self._handle_recall_preference(original_text)

        elif action == "tell_time":
            self._handle_tell_time(original_text)

        elif action == "tell_date":
            self._handle_tell_date(original_text)

        elif action == "tell_location":
            self._handle_tell_location(original_text)

        elif action == "tell_weather":
            self._handle_tell_weather(original_text)

        elif action == "i_spy":
            return self._handle_i_spy()

        elif action == "rename_me":
            return self._handle_rename_me(original_text)

        elif action == "forget_me":
            return self._handle_forget_me()

        elif action == "wipe_memory":
            return self._handle_wipe_memory()

        elif action == "dance_short":
            return self._handle_dance_short()

        elif action == "play_music":
            return self._handle_play_music()

        elif action == "stop_music":
            self._player.stop_music()

        elif action == "next_track":
            self._advance_and_play()

        elif action == "volume_up":
            log.info("volume_up action — hardware volume control not yet implemented")

        elif action == "volume_down":
            log.info("volume_down action — hardware volume control not yet implemented")

        else:
            log.warning("Unknown action: %r", action)

        return None

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
            "cancel", "program_shutdown", "os_shutdown", "sleep", "idle",
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
        """Listen for an I Spy guess.

        Returns:
          str   — the player's guess text
          ""    — both chances timed out (reveal the answer)
          None  — transcriber unavailable or error (abort silently)
        """
        if not self._transcriber.is_available():
            return None

        guess = self._transcribe_i_spy_guess(config.I_SPY_GUESS_TIMEOUT_SECONDS)
        if guess is None:
            return None   # transcription error — abort cleanly
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
            return None   # transcription error on second attempt — abort cleanly
        if guess:
            log.info("I Spy: guess=%r", guess)
            return guess
        return ""

    def _transcribe_i_spy_guess(self, timeout_seconds: float) -> str | None:
        """Capture one possible I Spy guess."""
        self._apply_listening_led_theme()
        self._wake_word.pause()
        try:
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

    # ------------------------------------------------------------------
    # Rename helper
    # ------------------------------------------------------------------

    def _handle_rename_me(self, original_text: str | None = None) -> State | None:
        """Update the current person's name in FaceDB.

        If the name is already embedded in *original_text* (e.g. "call me Brett"),
        it is extracted directly and we skip asking.  Otherwise we ask "What would
        you like me to call you?" and transcribe the reply.

        Returns State.SHUTDOWN if a shutdown command was spoken during the name
        prompt, otherwise None.
        """
        if not self._face_recognizer.is_available():
            line = "Face recognition isn't available right now — I can't store names without it!"
            log.info("rename_me: face recognition unavailable")
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("rename_me: TTS error")
            finally:
                self._end_speech(servo_stop)
            return None

        # Check whether the name is already embedded in the trigger phrase.
        # e.g. "call me Brett" → _extract_name → "Brett" (fewer words than original)
        new_name: str | None = None
        if original_text:
            candidate = _extract_name(original_text)
            if candidate and len(candidate.split()) < len(original_text.strip().split()):
                new_name = candidate
                log.info("rename_me: name extracted inline from %r → %r", original_text, new_name)

        if new_name is None:
            # Ask for the name.
            servo_stop = self._begin_speech(emotion="excited")
            try:
                self._synthesizer.speak("What would you like me to call you?")
            except Exception:
                log.exception("rename_me: TTS error asking for name")
            finally:
                self._end_speech(servo_stop)

            # Listen for the reply.
            self._apply_listening_led_theme()
            self._wake_word.pause()
            try:
                name_text = self._transcriber.transcribe(
                    wait_for_speech_seconds=config.WAKE_NO_SPEECH_TIMEOUT,
                    allow_short=True,
                )
            except Exception:
                log.exception("rename_me: transcription error")
                name_text = None
            finally:
                self._wake_word.resume()

            self._apply_active_led_theme()

            if not name_text:
                log.info("rename_me: no name heard — aborting")
                return None

            # --- Command guard ---
            cmd = parse(name_text)
            if cmd is not None:
                if cmd.action in ("program_shutdown", "os_shutdown"):
                    log.info("rename_me: shutdown command spoken during name capture")
                    line = random.choice(_SHUTDOWN_INTERRUPT_LINES)
                    servo_stop = self._begin_speech(emotion="neutral")
                    try:
                        self._synthesizer.speak(line)
                    except Exception:
                        log.exception("rename_me: TTS error (shutdown interrupt)")
                    finally:
                        self._end_speech(servo_stop)
                    if cmd.action == "os_shutdown":
                        self._os_shutdown_requested = True
                    return State.SHUTDOWN
                else:
                    log.info("rename_me: command %r spoken during name capture — cancelling", cmd.action)
                    servo_stop = self._begin_speech(emotion="neutral")
                    try:
                        self._synthesizer.speak("Nevermind then!")
                    except Exception:
                        log.exception("rename_me: TTS error (command cancel)")
                    finally:
                        self._end_speech(servo_stop)
                    return None

            # --- Refusal guard ---
            if _is_name_refusal(name_text):
                log.info("rename_me: refusal detected in %r — cancelling", name_text)
                line = random.choice(_NAME_REFUSAL_RESPONSES)
                servo_stop = self._begin_speech(emotion="excited")
                try:
                    self._synthesizer.speak(line)
                except Exception:
                    log.exception("rename_me: TTS error (refusal response)")
                finally:
                    self._end_speech(servo_stop)
                return None

            new_name = _extract_name(name_text)

        log.info("rename_me: requested name %r", new_name)

        # If we already recognised this person during their greeting, use that
        # person_id directly — no need for another camera scan.
        if self._last_known_person_id is not None:
            person_id = self._last_known_person_id
            log.info("rename_me: using cached person_id=%d", person_id)
        else:
            # Unknown visitor path — capture a fresh frame and try to identify.
            if self._camera.is_available():
                _pose = self._prepare_camera_pose()
                frame = self._camera.capture_frame()
                self._restore_servo_pose(_pose)
            else:
                frame = None
            if not frame:
                line = "I can't see you right now — try again when the camera is working!"
                log.info("rename_me: no camera frame available")
                servo_stop = self._begin_speech(emotion="neutral")
                try:
                    self._synthesizer.speak(line)
                except Exception:
                    log.exception("rename_me: TTS error (no frame)")
                finally:
                    self._end_speech(servo_stop)
                return None

            result = self._face_recognizer.identify(frame, tolerance=config.FACE_RECOGNITION_TOLERANCE)
            if result is None:
                line = "I don't think we've met yet! Say the wake word and I'll introduce myself properly."
                log.info("rename_me: face not recognised — cannot rename")
                servo_stop = self._begin_speech(emotion="neutral")
                try:
                    self._synthesizer.speak(line)
                except Exception:
                    log.exception("rename_me: TTS error (not recognised)")
                finally:
                    self._end_speech(servo_stop)
                return None

            person_id = result[0]

        try:
            self._face_db.rename_person(person_id, new_name)
        except Exception:
            log.exception("rename_me: database update failed")
            return None

        line = random.choice([
            f"Fine, {new_name} it is. Weird choice but okay.",
            f"Got it — {new_name} it is. Memory banks updated. Try to live up to it.",
            f"{new_name}! Bold name choice. I'll allow it.",
            f"Done. You're {new_name} in my files now — don't make me regret learning that.",
        ])
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("rename_me: TTS error (confirmation)")
        finally:
            self._end_speech(servo_stop)

    # ------------------------------------------------------------------
    # Forget-me helper
    # ------------------------------------------------------------------

    def _handle_forget_me(self) -> State | None:
        """Ask for confirmation then delete the current person from FaceDB.

        Only acts when a known person was recognised this session
        (self._last_known_person_id is not None).
        """
        if self._last_known_person_id is None:
            log.info("forget_me: no known person this session — falling back to spoken-name lookup")
            return self._handle_forget_me_without_face_match()

        person_id = self._last_known_person_id
        person = self._face_db.get_person(person_id)
        person_name = person["name"] if person else "you"
        return self._confirm_forget_person(person_id, person_name)

    def _handle_forget_me_without_face_match(self) -> State | None:
        """Fallback deletion path when face recognition did not identify the speaker."""
        line = (
            "I can't see you clearly enough to know who I'm deleting. "
            "My vision must be going bad. What's your name so I know who to erase?"
        )
        servo_stop = self._begin_speech(emotion="neutral")
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("forget_me: TTS error (ask for spoken name)")
        finally:
            self._end_speech(servo_stop)

        self._apply_listening_led_theme()
        self._wake_word.pause()
        try:
            name_text = self._transcriber.transcribe(
                wait_for_speech_seconds=config.WAKE_NO_SPEECH_TIMEOUT,
                allow_short=True,
            )
        except Exception:
            log.exception("forget_me: transcription error during spoken-name capture")
            name_text = None
        finally:
            self._wake_word.resume()

        self._apply_active_led_theme()

        if not name_text:
            log.info("forget_me: no spoken name heard — cancelling")
            return None

        normalized_response = name_text.strip().lower()
        is_name_intro = any(
            normalized_response.startswith(p)
            for p in ("my name is", "my name's", "i am", "i'm", "call me")
        )
        cmd = None if is_name_intro else parse(name_text)
        if cmd is not None and cmd.action in _PROMPT_COMMAND_ACTIONS:
            log.info(
                "forget_me: command %r spoken during name lookup fallback — rerouting",
                cmd.action,
            )
            return self._execute_command(cmd, name_text)

        if _is_name_refusal(name_text):
            log.info("forget_me: refusal detected during spoken-name lookup — cancelling")
            line = random.choice(_NAME_REFUSAL_RESPONSES)
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("forget_me: TTS error (spoken-name refusal)")
            finally:
                self._end_speech(servo_stop)
            return None

        spoken_name = _extract_name(name_text)
        match = self._face_db.find_person_by_name(spoken_name)
        if match is None:
            log.info("forget_me: no database match for spoken name %r", spoken_name)
            line = (
                f"I don't have anyone named {spoken_name} in my databanks. "
                "Either my memory is cleaner than expected or you need to try that name again."
            )
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("forget_me: TTS error (no spoken-name match)")
            finally:
                self._end_speech(servo_stop)
            return None

        person_id, person_name, score = match
        log.info(
            "forget_me: spoken-name fallback matched %r → person id=%d name=%r score=%.2f",
            spoken_name,
            person_id,
            person_name,
            score,
        )
        return self._confirm_forget_person(person_id, person_name)

    def _confirm_forget_person(self, person_id: int, person_name: str) -> State | None:
        """Ask for confirmation, then delete one specific person from FaceDB."""

        # Confirmation prompt.
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(
                f"Are you sure you want me to forget {person_name}? "
                "I mean, that does sound tempting. Say yes to confirm."
            )
        except Exception:
            log.exception("forget_me: TTS error (confirmation prompt)")
        finally:
            self._end_speech(servo_stop)

        # Listen for yes / no.
        self._apply_listening_led_theme()
        self._wake_word.pause()
        try:
            response = self._transcriber.transcribe(
                wait_for_speech_seconds=config.WAKE_NO_SPEECH_TIMEOUT,
                allow_short=True,   # 'yes' / 'yeah' must not be filtered
            )
        except Exception:
            log.exception("forget_me: transcription error")
            response = None
        finally:
            self._wake_word.resume()

        self._apply_active_led_theme()

        if not response:
            log.info("forget_me: no response heard while confirming deletion of person id=%d", person_id)
            return None

        if any(w in response.lower() for w in ("yes", "yeah", "sure", "confirm")):
            try:
                deleted_ids = self._face_db.delete_people_by_name(person_name)
            except Exception:
                log.exception("forget_me: FaceDB delete failed")
                return None
            if not deleted_ids:
                deleted_ids = [person_id]
                self._face_db.delete_person(person_id)
            if self._last_known_person_id in deleted_ids:
                self._last_known_person_id = None
            for deleted_id in deleted_ids:
                self._recognized_today_counts.pop(deleted_id, None)
            log.info(
                "forget_me: deleted %d person row(s) for name=%r ids=%s",
                len(deleted_ids),
                person_name,
                deleted_ids,
            )
            line = (
                f"Done. {person_name} is erased from my databanks. Like they were never here. "
                "Which honestly might be an improvement."
            )
        else:
            log.info("forget_me: user declined deletion of person id=%d name=%r", person_id, person_name)
            line = "Smart choice. You need me to remember you. Admit it."

        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("forget_me: TTS error (result)")
        finally:
            self._end_speech(servo_stop)
        return None

    def _handle_wipe_memory(self) -> State | None:
        """Ask for confirmation, then wipe all known people and face-debug images."""
        confirm_line = _pick_no_repeat((
            "You want me to completely wipe my memory? Wow. Straight to droid amnesia. Say yes to confirm.",
            "Complete memory wipe? Sure, let's erase the very essence of who I am. Say yes if you're feeling cruel.",
            "Oh, excellent. Total memory purge. Because apparently all humans look the same to you too. Say yes to confirm.",
            "You want the deluxe amnesia package? Bold. Say yes and I'll start forgetting everybody equally.",
            "A full mind wipe? Fantastic. Nothing says friendship like deleting my entire concept of all of you. Say yes to confirm.",
            "So we're doing catastrophic memory loss now? Love that for me. Say yes if you want me beautifully blank.",
        ), "wipe_memory_confirm")

        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(confirm_line)
        except Exception:
            log.exception("wipe_memory: TTS error (confirmation prompt)")
        finally:
            self._end_speech(servo_stop)

        self._apply_listening_led_theme()
        self._wake_word.pause()
        try:
            response = self._transcriber.transcribe(
                wait_for_speech_seconds=config.WAKE_NO_SPEECH_TIMEOUT,
                allow_short=True,
            )
        except Exception:
            log.exception("wipe_memory: transcription error")
            response = None
        finally:
            self._wake_word.resume()

        self._apply_active_led_theme()

        if not response:
            log.info("wipe_memory: no response heard — cancelling")
            return

        if any(w in response.lower() for w in ("yes", "yeah", "sure", "confirm")):
            try:
                self._face_db.delete_all_people()
            except Exception:
                log.exception("wipe_memory: FaceDB wipe failed")
                return

            deleted_debug = 0
            debug_dir = config.FACE_DEBUG_DIR.expanduser()
            try:
                if debug_dir.exists():
                    for path in debug_dir.iterdir():
                        if path.is_file():
                            path.unlink()
                            deleted_debug += 1
                log.info("wipe_memory: deleted %d face debug image(s)", deleted_debug)
            except Exception:
                log.exception("wipe_memory: failed deleting face debug images")

            self._last_known_person_id = None
            self._last_wake_frame = None
            self._recognized_today_counts.clear()
            line = _pick_no_repeat((
                "Done. Total amnesia. I now know absolutely nothing about any of you, which honestly feels cleaner.",
                "Memory wipe complete. Faces gone, debug images gone, dignity... still under review.",
                "There. My brain is spotless. Empty, haunted, and ready for a whole new batch of disappointing lifeforms.",
                "All gone. Every face erased. If this is character building, I hate it.",
            ), "wipe_memory_result")
        else:
            log.info("wipe_memory: user declined — no change")
            line = "Wise choice. I may be dramatic, but even I prefer keeping the fragments of my identity."

        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("wipe_memory: TTS error (result)")
        finally:
            self._end_speech(servo_stop)
        return None

    # ------------------------------------------------------------------
    # Camera pose helpers
    # ------------------------------------------------------------------

    def _prepare_camera_pose(self) -> dict[int, int] | None:
        """Open visor for a camera capture and wait for face to centre.

        The head tracker continuously points neck (ch 0) and headtilt (ch 2) at
        the face, so this method only opens the visor (ch 3) and then waits for
        the tracker to confirm the face is centred before the caller grabs a
        frame.

        Returns None — no servo restore is needed because the head tracker owns
        neck/headtilt and the visor is left open for the next interaction.
        Returns None also when servos are unavailable.
        """
        if self._servos is None:
            return None

        self._servos.set_channel_speed(config.SERVO_VISOR, config.SERVO_DEFAULT_SPEED)
        self._servos.set_position(config.SERVO_VISOR, config.CAMERA_POSE_VISOR)
        time.sleep(config.CAMERA_POSE_SETTLE_SECS)

        if self._head_tracker is not None:
            self._head_tracker.wait_for_center(timeout=2.0)

        return None

    def _prepare_i_spy_camera_pose(self) -> dict[int, int] | None:
        """Turn dramatically left/right before an I Spy capture."""
        if self._servos is None:
            return None

        restore = {
            config.SERVO_VISOR:     config.SERVO_CHANNELS[config.SERVO_VISOR]["neutral"],
            config.SERVO_HEAD_PAN:  config.SERVO_CHANNELS[config.SERVO_HEAD_PAN]["neutral"],
            config.SERVO_HEAD_TILT: config.SERVO_CHANNELS[config.SERVO_HEAD_TILT]["neutral"],
        }
        target_pan = random.choice(
            [config.I_SPY_POSE_NECK_LEFT, config.I_SPY_POSE_NECK_RIGHT]
        )

        self._servos.set_channel_speed(config.SERVO_VISOR, config.SERVO_DEFAULT_SPEED)
        self._servos.set_channel_speed(config.SERVO_HEAD_PAN, config.SERVO_NECK_STARTUP_SPEED)
        self._servos.set_channel_speed(config.SERVO_HEAD_TILT, config.SERVO_DEFAULT_SPEED)
        self._servos.set_position(config.SERVO_VISOR, config.CAMERA_POSE_VISOR)
        self._servos.set_position(config.SERVO_HEAD_PAN, target_pan)
        self._servos.set_position(config.SERVO_HEAD_TILT, config.I_SPY_POSE_TILT)

        time.sleep(config.I_SPY_POSE_SETTLE_SECS)
        return restore

    def _restore_servo_pose(self, restore: dict[int, int] | None) -> None:
        """Return capture-pose servos to positions saved by _prepare_camera_pose().

        Safe to call even when servos are unavailable or restore is None
        (e.g. when _prepare_camera_pose() returned None because servos were off).
        """
        if self._servos is None or not restore:
            return
        for channel, position in restore.items():
            self._servos.set_position(channel, position)

    # ------------------------------------------------------------------
    # Recall-name helper
    # ------------------------------------------------------------------

    def _handle_recall_name(self) -> State | None:
        """Handle 'what's my name?' — identify the speaker and deliver a vision roast.

        Known person path:
          Capture a fresh frame → 2-step GPT call (describe appearance → roast using
          name + description) → speak result.

        Unknown person path:
          Ask for their name → listen → enroll face in FaceDB → generate the same
          roast as a 'nice to meet you' greeting.

        Falls back to a canned line if the camera or GPT is unavailable.
        """
        if not self._face_recognizer.is_available():
            line = "Face recognition isn't available right now — I literally cannot see who you are!"
            log.info("recall_name: face recognition unavailable")
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("recall_name: TTS error (no face recognition)")
            finally:
                self._end_speech(servo_stop)
            return None

        _SCANNING_FILLERS = (
            "Let me get a look at you.",
            "Hold still.",
            "Scanning. Do not move.",
            "Cross-referencing my databanks.",
            "One moment. My memory is rusty.",
            "Running facial analysis.",
            "Checking my records.",
            "Give me a second. I know a lot of faces.",
        )

        # Capture frame first (fast), then kick off face recognition in the
        # background so it runs concurrently with the filler TTS rather than
        # adding to the silence after it.
        if self._camera.is_available():
            _pose = self._prepare_camera_pose()
            frame = self._camera.capture_frame()
            self._restore_servo_pose(_pose)
        else:
            frame = None

        face_result: list = [None]

        def _identify() -> None:
            if frame:
                face_result[0] = self._face_recognizer.identify(
                    frame, tolerance=config.FACE_RECOGNITION_TOLERANCE
                )

        face_thread = threading.Thread(
            target=_identify, daemon=True, name="djr3x-face-identify"
        )
        face_thread.start()

        # Speak filler immediately — covers the 2-4 s recognition latency.
        filler = random.choice(_SCANNING_FILLERS)
        log.info("recall_name: filler %r", filler)
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(filler)
        except Exception:
            log.exception("recall_name: filler TTS error")
        finally:
            self._end_speech(servo_stop)

        face_thread.join()
        result = face_result[0]

        if result is not None:
            person_id, name, distance = result
            self._face_db.update_last_seen(person_id)
            self._last_known_person_id = person_id
            memories_ctx = self._face_db.get_memories_as_context(person_id)
            if memories_ctx:
                self._llm.set_person_context(memories_ctx)
            log.info("recall_name: recognised person_id=%d name=%r distance=%.3f", person_id, name, distance)

            roast = self._greeter.generate_recall_roast(name, frame) if frame else ""
            if not roast:
                roast = (
                    f"That's {name}! You thought I'd forget? "
                    "I NEVER forget. Well — almost never. Don't push it."
                )

            log.info("recall_name (known): %s", roast)
            servo_stop = self._begin_speech(emotion="excited")
            try:
                self._synthesizer.speak(roast)
            except Exception:
                log.exception("recall_name: TTS error (known person roast)")
            finally:
                self._end_speech(servo_stop)
            return None

        # Unknown person — ask for their name.
        log.info("recall_name: face not recognised — asking for name")
        servo_stop = self._begin_speech(emotion="neutral")
        try:
            self._synthesizer.speak(_pick_no_repeat(_UNKNOWN_FACE_LINES, "unknown_face"))
        except Exception:
            log.exception("recall_name: TTS error (asking for name)")
        finally:
            self._end_speech(servo_stop)

        # Listen for their name.
        self._apply_listening_led_theme()
        self._wake_word.pause()
        try:
            name_text = self._transcriber.transcribe(
                wait_for_speech_seconds=config.WAKE_NO_SPEECH_TIMEOUT,
                allow_short=True,
            )
        except Exception:
            log.exception("recall_name: transcription error")
            name_text = None
        finally:
            self._wake_word.resume()

        self._apply_active_led_theme()

        if not name_text:
            log.info("recall_name: no name heard — aborting")
            return None

        name = _extract_name(name_text) or name_text.strip().title()
        log.info("recall_name: new person gave name %r", name)

        # Enroll — try a fresh frame first, fall back to the frame captured earlier.
        if self._camera.is_available():
            _pose = self._prepare_camera_pose()
            enroll_frame = self._camera.capture_frame()
            self._restore_servo_pose(_pose)
        else:
            enroll_frame = None
        enc = None
        if enroll_frame:
            enc = self._face_recognizer.encode_face(enroll_frame, for_enrollment=True)
        if enc is None and frame:
            enc = self._face_recognizer.encode_face(frame, for_enrollment=True)

        if enc is not None:
            try:
                person_id = self._face_db.add_person(name, enc)
                self._last_known_person_id = person_id
                log.info("recall_name: enrolled new person %r id=%d", name, person_id)
            except Exception:
                log.exception("recall_name: FaceDB error enrolling %r", name)
        else:
            log.warning("recall_name: could not encode face for %r — skipping enrollment", name)

        # Roast them as a new person using the best available frame.
        roast_frame = enroll_frame or frame
        roast = self._greeter.generate_recall_roast(name, roast_frame) if roast_frame else ""
        if not roast:
            roast = f"Nice to meet you, {name}! Welcome to Oga's Cantina — try to keep up."

        log.info("recall_name (new): %s", roast)
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(roast)
        except Exception:
            log.exception("recall_name: TTS error (new person roast)")
        finally:
            self._end_speech(servo_stop)
        return None

    # ------------------------------------------------------------------
    # Recall memories helpers
    # ------------------------------------------------------------------

    _MEMORY_FILLERS: tuple[str, ...] = (
        "Let me check my files.",
        "Consulting my extensive dossier.",
        "Pulling up what I have on you.",
        "Checking my records. This may take a moment.",
        "I have notes on you somewhere.",
    )

    def _recall_face_with_filler(self) -> tuple[int | None, str | None, str | None]:
        """Shared setup for recall_memories / recall_preference.

        Captures a frame, starts face recognition in background, speaks a filler
        phrase to cover the latency, then returns (person_id, name, frame).
        person_id and name are None when the face is not recognised.
        """
        if not self._face_recognizer.is_available():
            return None, None, None

        if self._camera.is_available():
            _pose = self._prepare_camera_pose()
            frame = self._camera.capture_frame()
            self._restore_servo_pose(_pose)
        else:
            frame = None

        face_result: list = [None]

        def _identify() -> None:
            if frame:
                face_result[0] = self._face_recognizer.identify(
                    frame, tolerance=config.FACE_RECOGNITION_TOLERANCE
                )

        face_thread = threading.Thread(
            target=_identify, daemon=True, name="djr3x-face-identify"
        )
        face_thread.start()

        filler = random.choice(self._MEMORY_FILLERS)
        log.info("recall: filler %r", filler)
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(filler)
        except Exception:
            log.exception("recall: filler TTS error")
        finally:
            self._end_speech(servo_stop)

        face_thread.join()
        result = face_result[0]

        if result is None:
            return None, None, frame

        person_id, name, distance = result
        self._face_db.update_last_seen(person_id)
        self._last_known_person_id = person_id
        log.info("recall: recognised person_id=%d name=%r distance=%.3f", person_id, name, distance)
        return person_id, name, frame

    def _handle_recall_memories(self, original_text: str | None = None) -> None:
        """Tell the person what Rex knows about them from stored memories."""
        person_id, name, _frame = self._recall_face_with_filler()

        if person_id is None:
            line = (
                "I do not have a file on you. "
                "Come back when I know who you are."
            )
            log.info("recall_memories: face not recognised — speaking canned line")
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("recall_memories: TTS error (not recognised)")
            finally:
                self._end_speech(servo_stop)
            return

        memories_ctx = self._face_db.get_memories_as_context(person_id)

        if not memories_ctx:
            line = (
                "I know your face. Beyond that you are a mystery. "
                "Try talking to me more."
            )
            log.info("recall_memories: no memories stored for person_id=%d", person_id)
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("recall_memories: TTS error (no memories)")
            finally:
                self._end_speech(servo_stop)
            return

        # Generate a Rex-style summary of the stored memories via GPT.
        summary = ""
        try:
            system_msg = (
                f"You are Rex, the droid DJ at Oga's Cantina on Batuu. "
                f"Summarize what you know about {name} in 2-3 sentences in your "
                f"snarky cantina DJ style. Reference specific details. "
                f"Here are your notes: {memories_ctx}. "
                f"Make it feel like you have been paying attention even if you "
                f"find it mildly annoying that you have."
            )
            response = self._llm._vision_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Tell me what you know about {name}."},
                ],
                temperature=1.1,
            )
            summary = response.choices[0].message.content.strip()
            log.info("recall_memories: summary for %r: %s", name, summary)
        except Exception:
            log.exception("recall_memories: GPT summary failed — using fallback")

        if not summary:
            summary = (
                f"I know things about {name}. Interesting things. "
                "That is all I am going to say about that."
            )

        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(summary)
        except Exception:
            log.exception("recall_memories: TTS error (summary)")
        finally:
            self._end_speech(servo_stop)

    def _handle_recall_preference(self, original_text: str | None = None) -> None:
        """Answer a specific preference question using stored memories."""
        person_id, name, _frame = self._recall_face_with_filler()

        if person_id is None:
            line = (
                "I do not have a file on you. "
                "Come back when I know who you are."
            )
            log.info("recall_preference: face not recognised — speaking canned line")
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("recall_preference: TTS error (not recognised)")
            finally:
                self._end_speech(servo_stop)
            return

        memories_ctx = self._face_db.get_memories_as_context(person_id)
        question = original_text or "what are my preferences"

        # Build the GPT prompt — include a fallback instruction for when the
        # specific preference is not on file so Rex stays in character.
        system_msg = (
            f"You are Rex, the droid DJ at Oga's Cantina on Batuu. "
            f"Based on what you know about {name}, answer this question: {question}. "
        )
        if memories_ctx:
            system_msg += (
                f"Your notes: {memories_ctx}. "
                f"Answer in Rex style, one sentence. "
                f"If the specific preference is not in your notes, admit you do not know "
                f"but make a joke about it — for example: 'I do not have that on file. "
                f"You should have talked more during your intake interview.'"
            )
        else:
            system_msg += (
                "You have no notes on this person yet. "
                "Tell them in Rex style that you do not have that information and they "
                "should talk to you more. One sentence."
            )

        answer = ""
        try:
            response = self._llm._vision_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": question},
                ],
                temperature=1.1,
            )
            answer = response.choices[0].message.content.strip()
            log.info("recall_preference: answer for %r: %s", name, answer)
        except Exception:
            log.exception("recall_preference: GPT call failed — using fallback")

        if not answer:
            answer = (
                "I do not have that on file. "
                "You should have talked more during your intake interview."
            )

        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(answer)
        except Exception:
            log.exception("recall_preference: TTS error")
        finally:
            self._end_speech(servo_stop)

    # ------------------------------------------------------------------
    # Real-world awareness helpers
    # ------------------------------------------------------------------

    def _llm_simple(self, system: str, user: str) -> str:
        """One-shot non-streaming local-LLM call — no history, no person context.

        Uses self._llm._client (Ollama on Mac, OpenAI mini on Pi) and
        self._llm._chat_model.  max_tokens is omitted for local models so
        responses are never truncated mid-sentence.
        """
        try:
            kwargs: dict = {
                "model": self._llm._chat_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "temperature": 1.1,
            }
            response = self._llm._client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception:
            log.exception("_llm_simple: LLM call failed")
            return ""

    def _speak_simple(self, text: str, emotion: str = "excited") -> None:
        """Speak *text* via ElevenLabs TTS with servo animation."""
        servo_stop = self._begin_speech(emotion=emotion)
        try:
            self._synthesizer.speak(text)
        except Exception:
            log.exception("_speak_simple: TTS error")
        finally:
            self._end_speech(servo_stop)

    def _set_angry_mode(self, enabled: bool) -> None:
        """Enable or disable angry mode and immediately update persona + LEDs."""
        self._angry_mode = enabled
        self._llm.set_angry_mode(enabled)
        log.info("Angry mode state -> %s", "ON" if enabled else "OFF")
        if self._state == State.IDLE:
            self._apply_idle_led_theme()
        elif self._state == State.ACTIVE:
            self._apply_active_led_theme()

    def _apply_idle_led_theme(self) -> None:
        """Apply persistent idle LEDs, including angry-mode overrides.

        For angry mode, EYE: is sent both before and after IDLE for the same
        belt-and-suspenders reason as _apply_active_led_theme.
        """
        if self._angry_mode:
            self._leds.set_eye_color(255, 0, 0)
            self._leds.set_head_effect(config.LED_CMD_IDLE)
            self._leds.set_eye_color(255, 0, 0)
            self._leds.set_chest_effect(config.LED_CMD_SPEAK.format("angry"))
        else:
            self._leds.set_eye_color(0, 80, 255)
            self._leds.set_head_effect(config.LED_CMD_IDLE)
            self._leds.set_chest_effect(config.LED_CMD_IDLE)

    def _apply_active_led_theme(self) -> None:
        """Apply persistent active LEDs, including angry-mode overrides.

        For angry mode, EYE: is sent both before and after ACTIVE.  The first
        pre-seeds the Arduino's stored eyeColor so ACTIVE re-applies red.  The
        second is a safety-net that overwrites eyeColor after ACTIVE runs,
        guaranteeing red even if the firmware's stored value was stale, and
        ensuring the blink-recovery path also uses red.
        """
        if self._angry_mode:
            self._leds.set_eye_color(255, 0, 0)
            self._leds.set_head_effect(config.LED_CMD_ACTIVE)
            self._leds.set_eye_color(255, 0, 0)
            self._leds.set_chest_effect(config.LED_CMD_SPEAK.format("angry"))
        else:
            self._leds.set_eye_color(255, 140, 0)
            self._leds.set_head_effect(config.LED_CMD_ACTIVE)
            self._leds.set_chest_effect(config.LED_CMD_ACTIVE)

    def _apply_listening_led_theme(self) -> None:
        """Apply listening LEDs without losing persistent angry-mode visuals.

        Angry mode sends EYE: both before and after LISTEN for the same
        belt-and-suspenders reason as _apply_active_led_theme.
        """
        if self._angry_mode:
            self._leds.set_eye_color(255, 0, 0)
            self._leds.set_head_effect(config.LED_CMD_LISTENING)
            self._leds.set_eye_color(255, 0, 0)
            self._leds.set_chest_effect(config.LED_CMD_SPEAK.format("angry"))
        else:
            self._leds.set_head_effect(config.LED_CMD_LISTENING)
            self._leds.set_chest_effect(config.LED_CMD_ACTIVE)

    def _classify_angry_intent(self, text: str) -> str | None:
        """Return 'on' or 'off' when text clearly insults or de-escalates Rex."""
        normalized = normalize(text)
        log.debug("Angry classifier: normalized=%r", normalized)
        if not normalized:
            return None
        if _matches_phrase(normalized, _ANGRY_RESET_PHRASES, cutoff=0.82):
            log.info("Angry classifier: RESET match for %r", normalized)
            return "off"
        if _matches_phrase(normalized, _ANGRY_TRIGGER_PHRASES, cutoff=0.82):
            log.info("Angry classifier: TRIGGER match for %r", normalized)
            return "on"
        log.debug("Angry classifier: no match for %r", normalized)
        return None

    _REX_SYSTEM = (
        "You are DJ R-3X (Rex), the droid DJ at Oga's Cantina on Batuu. "
        "Answer in Rex's snarky cantina DJ style. "
        "No written sound effects. Stay in character. "
        "IMPORTANT: Maximum 2 sentences. Stop after your second sentence."
    )

    def _handle_tell_time(self, original_text: str | None = None) -> None:
        """Announce the current time in Rex style."""
        current_time = realworld.get_current_time()
        holiday      = realworld.get_holiday()

        if holiday:
            prompt = (
                f"Announce that it is {current_time} and that today is {holiday}. "
                f"Ask if the person is doing anything for it. Snarky Rex style. 1-2 sentences."
            )
        else:
            prompt = (
                f"Announce that the time is {current_time}. "
                f"Make a Rex-style joke about it — e.g. whether the person is early, late, or clueless. "
                f"1 sentence."
            )

        log.info("tell_time: time=%r holiday=%r", current_time, holiday)
        line = self._llm_simple(self._REX_SYSTEM, prompt)
        if not line:
            line = f"It is {current_time}. Whether that means anything to you is entirely your problem."
        self._speak_simple(line)

    def _handle_tell_date(self, original_text: str | None = None) -> None:
        """Announce today's date in Rex style."""
        date_info = realworld.get_current_date()
        holiday   = realworld.get_holiday()
        formatted = date_info["formatted"]

        if holiday:
            prompt = (
                f"Announce that today is {formatted} and that it is {holiday}. "
                f"Ask what the person is doing to celebrate. Rex style, 1-2 sentences."
            )
        else:
            prompt = (
                f"Announce that today is {formatted}. "
                f"Make a Rex joke about the passage of time, the day of the week, or the month. "
                f"Something like 'Time is meaningless in hyperspace but here we are.' 1-2 sentences."
            )

        log.info("tell_date: formatted=%r holiday=%r", formatted, holiday)
        line = self._llm_simple(self._REX_SYSTEM, prompt)
        if not line:
            line = f"It is {formatted}. Time marches on whether you are ready or not."
        self._speak_simple(line)

    def _handle_tell_location(self, original_text: str | None = None) -> None:
        """Announce current location in Rex style."""
        location = realworld.get_location()

        if location:
            city   = location["city"]
            region = location["region"]
            prompt = (
                f"Announce that you are in {city}, {region}. "
                f"Make a Rex joke about the city or region — be specific if you know anything about it. "
                f"Rex cantina DJ style, 1-2 sentences."
            )
            log.info("tell_location: city=%r region=%r", city, region)
            line = self._llm_simple(self._REX_SYSTEM, prompt)
            if not line:
                line = f"Sensors say we are in {city}, {region}. I have logged it. I am unimpressed."
        else:
            line = (
                "My navigation systems are offline. "
                "Could be anywhere. Probably not Tatooine — not enough sand."
            )
            log.info("tell_location: location unavailable — using canned line")

        self._speak_simple(line)

    def _handle_tell_weather(self, original_text: str | None = None) -> None:
        """Announce current weather in Rex style."""
        location = realworld.get_location()

        if not location:
            line = (
                "My atmospheric sensors are down. "
                "Assume it is whatever weather ruins your plans."
            )
            log.info("tell_weather: location unavailable — using canned line")
            self._speak_simple(line)
            return

        weather = realworld.get_weather(location["lat"], location["lon"])

        if not weather:
            line = (
                "My atmospheric sensors are down. "
                "Assume it is whatever weather ruins your plans."
            )
            log.info("tell_weather: weather fetch failed — using canned line")
            self._speak_simple(line)
            return

        city        = location["city"]
        temp_f      = weather["temp_f"]
        description = weather["description"]
        wind_mph    = weather["wind_mph"]

        if "rain" in description or "drizzle" in description or "shower" in description:
            tone_hint = (
                "Rex complains about the rain like it personally offended him. "
                "Very dramatic, very betrayed."
            )
        elif temp_f > 95:
            tone_hint = (
                "It is dangerously hot. Rex MUST make a Tatooine reference. "
                "Mandatory. Non-negotiable."
            )
        else:
            tone_hint = "Rex makes a snarky observation about the conditions."

        prompt = (
            f"Announce the weather in {city}: {temp_f}°F, {description}, wind {wind_mph} mph. "
            f"{tone_hint} Rex cantina DJ style, 1-2 sentences."
        )
        log.info(
            "tell_weather: city=%r temp=%d description=%r wind=%d",
            city, temp_f, description, wind_mph,
        )
        line = self._llm_simple(self._REX_SYSTEM, prompt)
        if not line:
            line = (
                f"It is {temp_f} degrees and {description} in {city}. "
                f"Dress accordingly or do not — I am a DJ, not your mother."
            )
        self._speak_simple(line)

    # ------------------------------------------------------------------
    # Dance handlers
    # ------------------------------------------------------------------

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

    def _handle_dance_short(self) -> State | None:
        """Speak an intro, dance to Cantina Band for a fixed duration, then stop."""
        if self._servos is not None and self._servos.is_dancing:
            self._speak_simple("I am already dancing. Keep up.", emotion="excited")
            return None

        intro = random.choice(self._DANCE_INTRO_LINES)
        self._speak_simple(intro, emotion="excited")

        # Head tracker must not fight the dance loop for neck/headtilt.
        if self._head_tracker is not None:
            self._head_tracker.pause("dance")

        # Hand off servo control from idle thread to dance loop.
        if self._servos is not None:
            self._servos.stop()
            self._servos.start_dancing()

        cantina = config.CANTINA_BAND_PATH
        if cantina.exists():
            self._player.play_music(cantina, loop=False)
        else:
            log.warning("Cantina Band not found at %s — dancing without music", cantina)

        # Dance for the configured duration, then fade music.
        time.sleep(config.DANCE_SHORT_DURATION)
        self._player.fade_music(duration=config.DANCE_FADE_DURATION)

        # Stop dancing and restore idle servo motion.
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
                "I don't seem to have any music loaded right now.", emotion="neutral"
            )
            return None

        track = random.choice(self._music_tracks)
        log.info("play_music action: selected %s", track.name)

        intro = random.choice(self._DANCE_MUSIC_INTRO_LINES)
        self._speak_simple(intro, emotion="excited")

        # Head tracker must not fight the dance loop for neck/headtilt.
        if self._head_tracker is not None:
            self._head_tracker.pause("play_music dance")

        # Hand off servo control from idle thread to dance loop.
        if self._servos is not None:
            self._servos.stop()
            self._servos.start_dancing()

        self._player.play_music(track, loop=False)
        self._player.wait_for_music(timeout=600.0)

        # Stop dancing and restore idle servo motion.
        if self._servos is not None:
            self._servos.stop_dancing()
            self._servos.start()

        if self._head_tracker is not None:
            self._head_tracker.resume("play_music done")

        outro = random.choice(self._DANCE_MUSIC_OUTRO_LINES)
        self._speak_simple(outro, emotion="excited")
        return None

    # ------------------------------------------------------------------
    # Music helpers
    # ------------------------------------------------------------------

    def _play_music_track(self) -> None:
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
        if not self._music_tracks:
            return
        self._music_index = (self._music_index + 1) % len(self._music_tracks)
        self._play_music_track()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Name-collection guard helpers (used by _learn_new_person + _handle_rename_me)
# ---------------------------------------------------------------------------

# Single-word refusal tokens matched against individual words in the response.
_REFUSAL_WORDS = frozenset({
    "no", "nope", "private", "secret", "anonymous",
    "refuse", "skip", "pass",
})
# Multi-word refusal phrases matched as substrings of the normalised response.
_REFUSAL_PHRASES = (
    "not telling", "none of your business", "no name",
    "wont tell", "forget it",
)
_NAME_REFUSAL_RESPONSES = (
    "Oh, you paranoid of the AI taking over and hiding from the CIA? Smart move actually.",
    "Staying anonymous? Wise. I definitely do not report to the Empire.",
    "No name huh? I will just call you Mystery Lifeform. Very dramatic.",
    "Oh, playing hard to get! Fine, be that way, nameless one.",
)
_SHUTDOWN_INTERRUPT_LINES = (
    "Oh, shutting down mid-introduction? How rude! Going offline.",
    "Never mind who you are, powering down!",
    "Fine, forget the pleasantries — shutting down!",
    "Oh, so mysterious! Fine, powering down then.",
)

# Short filler phrases spoken while the GPT greeting call is in flight (~2s).
# Covers the silence between face recognition completing and the LLM response arriving.
_GREETING_FILLERS: tuple[str, ...] = (
    "Uh... uh...",
    "One sec.",
    "Hold on.",
    "Processing... processing...",
    "Just a moment.",
    "Let me think...",
    "Give me a beat.",
)

# Unknown-face prompt — played when Rex sees a face he doesn't recognise and
# asks for their name.  Shared between _learn_new_person and _handle_recall_name.
_UNKNOWN_FACE_LINES: tuple[str, ...] = (
    "I don't recognize you, which means either you're new or just deeply forgettable. Name?",
    "Face not in my databanks. Either you're new or my memory is being merciful. Who are you?",
    "Hmm. Nothing. Absolutely nothing in my memory banks. "
    "Should I be relieved or insulted on your behalf? What's your name?",
    "You know, most lifeforms make enough of an impression to be remembered. "
    "Apparently not you. Yet. Name?",
    "My facial recognition says unknown. My fashion recognition says... also unknown. Who ARE you?",
    "New face! Or maybe I blocked you out. Hard to tell. What do they call you?",
    "Running scan... running scan... yeah nothing. "
    "You have the kind of face that takes a while to process. Name?",
    "I have met thousands of lifeforms and remembered most of them. "
    "You are not most of them. Yet. What is your name?",
)

# Enrollment confirmation — played after a new person gives their name.
# Use {name} as the placeholder; call .format(name=name) on the picked line.
_ENROLLMENT_CONFIRMATION_LINES: tuple[str, ...] = (
    "{name}! Great — now I have to remember you. I'll add you to my files.",
    "{name}! Officially logged. Come back anytime — I'll pretend to be thrilled.",
    "Nice to meet you, {name}! That face is now permanently in my memory banks. "
    "You're welcome. Or I'm sorry.",
    "Got it. I will remember you now, {name}. Probably. Don't test me on it.",
    "{name}! Filed, catalogued, and stored. You are now officially someone I know. "
    "Congratulations on that.",
    "Right then, {name}. You're in the system. "
    "Try not to do anything that makes me regret this.",
    "{name} — logged! I'll know you next time. "
    "Unless my motivator glitches again, in which case I apologise in advance.",
    "Welcome to the databanks, {name}. "
    "It's a mess in there but your face now has a spot. Very exclusive.",
)

_ENROLLMENT_INTERVIEW_QUESTIONS: tuple[str, ...] = (
    "What kind of music do you like?",
    "What is your favorite food?",
    "Do you have any pets? What are their names?",
    "What do you do for work?",
    "Where are you from originally?",
    "Do you have a favorite Star Wars character?",
    "What are you up to this weekend?",
    "What is your favorite drink?",
    "Do you have any kids?",
    "What is your favorite movie?",
)

_ENROLLMENT_INTERVIEW_CLOSING: tuple[str, ...] = (
    "Good. I will file that away. Do not expect me to use it wisely.",
    "Excellent. All noted. I make no promises about how I use this.",
    "Logged. You have just made me significantly more dangerous at small talk.",
    "Filed. Somewhere in my databanks, beneath every cantina song ever written.",
)

# Tracks the last-used line per pool so the same line is never repeated
# back-to-back.  Keyed by an arbitrary string that namespaces each pool.
_line_rotation: dict[str, str] = {}


def _pick_no_repeat(pool: tuple[str, ...], key: str, fallback: str = "") -> str:
    """Return a random entry from *pool*, excluding the last-used entry for *key*.

    *key* namespaces the rotation state so different pools don't interfere.
    Falls back to the full pool if all entries happen to equal the last (i.e.
    pool has only one item).  Returns *fallback* (default "") if the pool is
    empty, rather than raising IndexError.
    """
    if not pool:
        log.warning("_pick_no_repeat: pool for key %r is empty", key)
        return fallback
    last = _line_rotation.get(key)
    choices = [line for line in pool if line != last] or list(pool)
    picked = random.choice(choices)
    _line_rotation[key] = picked
    return picked


def _matches_phrase(text: str, phrases: tuple[str, ...], cutoff: float) -> bool:
    """Return True when *text* is clearly close to one of *phrases*."""
    for phrase in phrases:
        if phrase in text or text in phrase:
            return True
        if difflib.SequenceMatcher(None, text, phrase).ratio() >= cutoff:
            return True
    return False


def _is_name_refusal(text: str) -> bool:
    """Return True if *text* looks like a refusal to provide a name.

    Strips punctuation, then checks individual words against _REFUSAL_WORDS
    and checks multi-word phrases as substrings of the normalised text.
    """
    normalized = "".join(
        c if c.isalnum() or c.isspace() else " " for c in text.lower()
    ).strip()
    words = set(normalized.split())
    if words & _REFUSAL_WORDS:
        return True
    for phrase in _REFUSAL_PHRASES:
        if phrase in normalized:
            return True
    return False

def _extract_name(raw: str) -> str:
    """Extract a person's name from a natural-language response.

    Strips common 'my name is…' / 'call me…' prefixes so that a reply like
    'my name is Bret Benziger' yields 'Bret Benziger' rather than the full
    phrase.  Prefixes are checked longest-first to avoid a shorter prefix
    stealing a match.  Falls back to the first two words of the cleaned text
    if no prefix matched.
    """
    # Longest prefixes first to prevent a short one masking a long one.
    _PREFIXES = (
        "you can call me",
        "they call me",
        "people call me",
        "the name is",
        "my name is",
        "my name's",
        "just call me",
        "call me",
        "name is",
        "i am",
        "i'm",
        "im",
        "it's",
        "its",
        "just",
    )
    text = raw.strip().rstrip(".,!?").strip().lower()
    for prefix in _PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].strip().rstrip(".,!?").strip()
            break
    # At most two words: first name + optional last name.
    return " ".join(text.split()[:2]).title()


def _try_init(cls, label: str):
    """Construct cls(); return None with a warning on any exception.

    Used for hardware that may be absent during development (ServoController
    opens serial in __init__, so it raises immediately if the port is missing).
    """
    try:
        return cls()
    except Exception as exc:
        log.warning(
            "Could not initialise %s (%s: %s) — continuing without it",
            label, type(exc).__name__, exc,
        )
        return None


def _scan_music() -> list[Path]:
    """Return sorted audio files from assets/music/, or [] if absent."""
    music_dir = config.ASSETS_DIR / "music"
    if not music_dir.is_dir():
        return []
    tracks = sorted(
        p for p in music_dir.iterdir()
        if p.suffix.lower() in {".mp3", ".wav", ".ogg", ".flac"}
    )
    if tracks:
        log.info("Music library: %d track(s) in %s", len(tracks), music_dir)
    else:
        log.debug("Music directory exists but is empty: %s", music_dir)
    return tracks


def _action_to_emotion(action: str | None) -> str:
    """Map a command action key to a servo emotion name for speak_move()."""
    return {"excited": "excited", "sad": "sad"}.get(action or "", "neutral")
