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
import logging
import os
import random
import threading
import time
from pathlib import Path

import config
from audio.player import AudioPlayer
from commands.parser import parse
from hardware.leds import LEDController
from hardware.servos import ServoController
from llm.chatgpt import ChatGPTClient
from llm.greeter import Greeter
from llm.vision_intent import vision_intent
from sequences.animations import AnimationPlayer
from speech.synthesizer import Synthesizer
from speech.transcriber import Transcriber
from speech.wake_word import WakeWordDetector
from vision.camera import Camera
from vision.face_db import FaceDB
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

        # State control
        self._state: State = State.IDLE
        self._wake_event = threading.Event()    # set by wake word callback
        self._shutdown_event = threading.Event()
        self._os_shutdown_requested: bool = False  # True only for voice/button shutdown
        self._pipeline_t0: float = 0.0          # monotonic time of last wake word detection

        # Music library — scanned once at startup
        self._music_tracks: list[Path] = _scan_music()
        self._music_index: int = 0

        # Idle atmosphere clips — disabled by "stop talking" command, re-enabled
        # automatically the next time the wake word activates Rex.
        self._idle_clips_enabled: bool = True

        # Tracks the person_id of the last face-recognised visitor so that a
        # "call me X" command can update their name without a second camera scan.
        # Cleared whenever we return to IDLE.
        self._last_known_person_id: int | None = None

        # The camera frame captured at wake word time — reused for enrollment
        # so the enrollment thread has a frame from when the face was definitely
        # in front of the camera.  Cleared whenever we return to IDLE.
        self._last_wake_frame: str | None = None

        # Greeting toggle — alternates between canned ("Hi There.mp3") and
        # personalized (GPT-4o + camera) on successive wake word activations.
        # False → canned first, then flips to True for personalized, and so on.
        self._greeting_toggle: bool = False

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
        self._wake_word.stop()
        if self._servos is not None:
            self._servos.close()
        self._leds.close()
        self._player.close()
        self._camera.stop()

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
        self._leds.set_eye_color(0, 80, 255)          # calm blue
        self._leds.set_chest_effect(config.LED_CMD_IDLE)
        self._leds.set_head_effect(config.LED_CMD_IDLE)

        if self._servos is not None:
            self._servos.set_emotion("neutral")

        # Unsuppress wake word (may have been left suppressed if we cut off
        # speech mid-utterance, e.g. via request_shutdown).
        self._wake_word.suppressed = False

        # Reset conversation context so each active session starts fresh.
        self._llm.clear_history()

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

            # Timer fired — play a random idle clip (if not muted by user).
            if not self._idle_clips_enabled:
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

        self._leds.set_chest_effect(config.LED_CMD_ACTIVE)
        self._leds.set_head_effect(config.LED_CMD_ACTIVE)
        self._leds.set_eye_color(255, 140, 0)          # warm amber

        if self._servos is not None:
            self._servos.set_emotion("neutral")

        # Start arm wave in background (non-blocking — idle thread stopped inside).
        self._animations.play_wake_greeting_arms()

        # Greet the user concurrently with the arm wave.
        self._play_wake_greeting()

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
            self._leds.set_head_effect(config.LED_CMD_LISTENING)

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

            # --- No speech detected within the timeout window ---
            if text is None:
                if after_response:
                    log.info(
                        "Follow-up silence timeout (%.0f s) — returning to IDLE",
                        config.ACTIVE_TIMEOUT_SECONDS,
                    )
                    self._play_return_to_idle_chime()
                    self._transition_to(State.IDLE)
                    return

                # First listen — prompt with "are you there?"
                prompt = random.choice(_ARE_YOU_THERE_PHRASES)
                log.info("No speech on first listen — prompting: %r", prompt)
                self._leds.set_head_effect(config.LED_CMD_ACTIVE)
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
                self._leds.set_head_effect(config.LED_CMD_LISTENING)
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
                    self._leds.set_head_effect(config.LED_CMD_ACTIVE)
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

            # --- Speech detected but Whisper returned nothing ---
            if not text:
                log.debug("Empty transcription — waiting again")
                after_response = True   # speech was heard; skip "are you there?" next time
                continue

            # --- We have a transcription ---
            after_response = True
            log.info("Transcribed: %r", text)

            # --- Speaking indicator ---
            self._leds.set_head_effect(config.LED_CMD_ACTIVE)
            self._leds.set_chest_effect(config.LED_CMD_ACTIVE)

            # --- Parse and respond ---
            _elapsed = f" [+{time.monotonic() - self._pipeline_t0:.1f}s]"
            cmd = parse(text)
            if cmd is not None and cmd.action == "vision":
                # Vision command — capture a fresh frame right now and send to LLM.
                log.info("Vision command: %r%s", cmd.phrases[0], _elapsed)
                frame = self._camera.capture_frame()
                if frame:
                    log.debug("Camera: fresh frame captured for vision command (%d bytes b64)",
                              len(frame))
                next_state = self._speak_llm(text, image=frame, t0=self._pipeline_t0)
            elif cmd is not None:
                log.info("Command matched: %r → %r%s", cmd.phrases[0], cmd.action, _elapsed)
                log.info("Rex (cmd): %s", cmd.response)
                next_state = self._execute_command(cmd, text)
            else:
                # No command match — check visual intent before calling LLM.
                if vision_intent(text):
                    log.info("Vision intent detected — capturing frame%s", _elapsed)
                    frame = self._camera.capture_frame()
                    if frame:
                        log.debug("Camera: fresh frame captured for vision intent (%d bytes b64)",
                                  len(frame))
                else:
                    log.info("No vision intent — sending text only to ChatGPT%s", _elapsed)
                    frame = None
                next_state = self._speak_llm(text, image=frame, t0=self._pipeline_t0)

            # Ensure all audio has finished before re-opening the mic.
            self._player.wait_for_speech()

            if next_state is not None:
                if next_state == State.IDLE:
                    self._play_return_to_idle_chime()
                self._transition_to(next_state)
                return

            # Restore ACTIVE indicators for next listen turn.
            self._leds.set_chest_effect(config.LED_CMD_ACTIVE)

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

    def _play_wake_greeting(self) -> None:
        """Greet the user on wake word.

        Priority:
          1. Face recognition available + camera available:
             - KNOWN person  → personalised known-person greeting (no toggle consumed)
             - UNKNOWN person → existing alternating canned/personalized path, then
                               offer to learn their name
          2. No face recognition → existing alternating canned/personalized path as before

        Always completes (audio finishes and any name-learning exchange is done)
        before returning so the caller can open the mic immediately afterwards.
        """
        frame: str | None = None
        if self._camera.is_available():
            frame = self._camera.capture_frame()
            if frame:
                self._last_wake_frame = frame   # save for enrollment fallback

        # ------------------------------------------------------------------
        # Try face recognition
        # ------------------------------------------------------------------
        if frame and self._face_recognizer.is_available():
            n_people = len(self._face_db.list_people())
            print(f"Face scan: {n_people} people in database")

            _SCANNING_LINES = [
                "Hmmmmm... interesting. Lifeform identity scan complete.",
                "Scanning... scanning... oh. It is you.",
                "Identity scan in progress... beep boop... scan complete.",
                "Hold still... analyzing lifeform... done.",
                "Running biometric scan... fascinating specimen.",
            ]

            if n_people == 0:
                # Empty database — definitely heading to enrollment.  Hide the
                # 2-4 s dlib latency behind the scanning audio line by running
                # face recognition and TTS concurrently.
                face_result: list = [None]

                def _identify_empty() -> None:
                    face_result[0] = self._face_recognizer.identify(
                        frame, tolerance=config.FACE_RECOGNITION_TOLERANCE
                    )

                face_thread = threading.Thread(
                    target=_identify_empty, daemon=True, name="djr3x-face-identify"
                )
                face_thread.start()

                scanning_line = random.choice(_SCANNING_LINES)
                log.info("Wake greeting: empty DB — playing scanning line concurrently — %r", scanning_line)
                servo_stop = self._begin_speech(emotion="excited")
                try:
                    self._synthesizer.speak(scanning_line)
                except Exception:
                    log.exception("Wake greeting: scanning line TTS error")
                finally:
                    self._end_speech(servo_stop)

                face_thread.join()
                result = face_result[0]
                print("Face scan: database empty — no comparison possible")

            else:
                # Database has known people — run recognition without any scanning
                # line so a recognised person gets an instant greeting.
                face_result2: list = [None]

                def _identify_known() -> None:
                    face_result2[0] = self._face_recognizer.identify(
                        frame, tolerance=config.FACE_RECOGNITION_TOLERANCE
                    )

                face_thread2 = threading.Thread(
                    target=_identify_known, daemon=True, name="djr3x-face-identify"
                )
                face_thread2.start()
                face_thread2.join()
                result = face_result2[0]

                if result is not None:
                    _pid, rname, rdist = result
                    print(
                        f"Face scan: RECOGNIZED {rname} (distance {rdist:.3f}, "
                        f"threshold {config.FACE_RECOGNITION_TOLERANCE})"
                    )
                else:
                    print(
                        f"Face scan: UNKNOWN (no match within threshold "
                        f"{config.FACE_RECOGNITION_TOLERANCE} — see logs for closest)"
                    )
                    # Unknown person in a non-empty DB — play scanning line now as
                    # a natural transition into the enrollment exchange.
                    scanning_line = random.choice(_SCANNING_LINES)
                    log.info("Wake greeting: unknown person — playing scanning line — %r", scanning_line)
                    servo_stop = self._begin_speech(emotion="excited")
                    try:
                        self._synthesizer.speak(scanning_line)
                    except Exception:
                        log.exception("Wake greeting: scanning line TTS error")
                    finally:
                        self._end_speech(servo_stop)

            if result is not None:
                person_id, name, distance = result
                self._face_db.update_last_seen(person_id)
                person = self._face_db.get_person(person_id)
                visit_count = person["visit_count"] if person else 1
                self._last_known_person_id = person_id
                self._play_known_person_greeting(name, visit_count)
                return

            # Unknown face — fall through to the existing alternating path,
            # then offer to learn the person's name.
            log.info("Wake greeting: face detected but unknown — running standard greeting")
            self._play_alternating_greeting(frame)
            self._learn_new_person(frame)
            return

        # ------------------------------------------------------------------
        # No face recognition (or no camera frame) — existing alternating path
        # ------------------------------------------------------------------
        self._play_alternating_greeting(frame)

    def _play_known_person_greeting(self, name: str, visit_count: int) -> None:
        """Speak a personalised greeting for a recognised returning visitor."""
        if visit_count <= 1:
            # First return visit after being enrolled
            line = random.choice([
                f"Oh great — {name} is back. I had exactly five minutes of peace. Worth it? Debatable.",
                f"*BWOOP* {name}! You came back! Bold move. I respect the audacity.",
                f"Well well well, {name} returns. The cantina was doing FINE without you, but here we are.",
                f"Oh! {name}! You actually remembered where the cantina is — I'm genuinely surprised.",
            ])
        elif visit_count < 5:
            line = random.choice([
                f"HEY, {name}! Back again?! You're really committing to this, huh.",
                f"*WHIRR* {name}! Visit number {visit_count}. Starting to become a problem.",
                f"Oh no. {name}. Again. I say 'oh no' affectionately, but still — oh no.",
            ])
        else:
            line = random.choice([
                f"*BWOOP* {name}! Visit {visit_count}! You practically PAY RENT here at this point!",
                f"Oh great, {name}. My favorite recurring problem has arrived. The cantina is yours, I guess.",
                f"HEY! {name}! Visit {visit_count} — at what point do we just give you a key?!",
            ])

        log.info("Wake greeting: known person '%s' (visit #%d) → %r", name, visit_count, line)
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("Wake greeting: known-person TTS error")
        finally:
            self._end_speech(servo_stop)

    def _play_alternating_greeting(self, frame: str | None) -> None:
        """Alternates between canned and personalized greetings, unchanged from before."""
        do_personalized = self._greeting_toggle and self._camera.is_available() and frame
        self._greeting_toggle = not self._greeting_toggle

        if do_personalized:
            log.info("Wake greeting: attempting personalized greeting")
            _HOLDING_CLIP = config.ASSETS_DIR / "audio" / "This is your cap.mp3"
            greeting_result: list[str | None] = [None]

            def _generate_personalized() -> None:
                if frame:
                    greeting_result[0] = self._greeter.generate(frame)

            greeter_thread = threading.Thread(
                target=_generate_personalized,
                daemon=True,
                name="djr3x-greeter",
            )
            greeter_thread.start()

            servo_stop = self._begin_speech(emotion="excited")
            try:
                if _HOLDING_CLIP.exists():
                    self._player.play_file(_HOLDING_CLIP)
                greeter_thread.join(timeout=15.0)
                if greeting_result[0]:
                    try:
                        self._synthesizer.speak(greeting_result[0])
                    except Exception:
                        log.exception("Wake greeting: TTS error")
            except Exception:
                log.exception("Wake greeting: personalized path error")
                greeter_thread.join(timeout=1.0)
            finally:
                self._end_speech(servo_stop)

            if greeting_result[0]:
                return
            log.info("Wake greeting: personalized path failed — falling back to canned")

        # Canned greeting
        _CANNED_AUDIO = config.ASSETS_DIR / "audio" / "Hi There.mp3"
        _CANNED_TTS = [
            "Oh great, you're here. The cantina just got significantly louder and marginally more interesting.",
            "HEY HEY HEY! A lifeform! Bold of you to show up looking like THAT.",
            "*BWOOP* Oh, it's you. Oga's Cantina — where even the questionable guests are welcome!",
            "Well well well, look what the Ronto dragged in. Welcome, I guess.",
            "HEY! You actually came back! I honestly didn't think you would. Impressed.",
        ]
        servo_stop = self._begin_speech(emotion="excited")
        try:
            if _CANNED_AUDIO.exists():
                self._player.play_file(_CANNED_AUDIO)
                self._player.wait_for_speech(timeout=10.0)
            else:
                self._synthesizer.speak(random.choice(_CANNED_TTS))
        except Exception:
            log.exception("Wake greeting: canned greeting error")
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
            self._synthesizer.speak("I don't think we've met — what's your name?")
        except Exception:
            log.exception("Wake greeting: name-ask TTS error")
        finally:
            self._end_speech(servo_stop)

        self._leds.set_head_effect(config.LED_CMD_LISTENING)
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

        self._leds.set_head_effect(config.LED_CMD_ACTIVE)

        # Capture a fresh frame NOW — the person just said their name and is
        # almost certainly still facing the camera.  This is our best shot at
        # a clean face encoding.  Captured before the thread starts so it is
        # available immediately (no camera access inside the thread).
        enroll_frame: str | None = None
        if self._camera.is_available():
            enroll_frame = self._camera.capture_frame()
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
                final_f = self._camera.capture_frame()
                if final_f:
                    log.info("Enrollment: final live frame captured (%d b64 bytes)", len(final_f))
                    enc = self._face_recognizer.encode_face(final_f, for_enrollment=True)
                    if enc is None:
                        log.warning("Enrollment: no face detected in final live frame")
                else:
                    log.warning("Enrollment: camera returned no frame on final attempt")

            if enc is not None:
                try:
                    self._face_db.add_person(name, enc)
                    log.info("Enrollment complete: %r stored in FaceDB", name)
                except Exception:
                    log.exception("Enrollment: FaceDB error storing %r", name)
            else:
                log.warning(
                    "Enrollment: all attempts failed to detect a face — %r NOT stored", name
                )

        threading.Thread(target=_enroll, daemon=True, name="djr3x-enroll").start()

        welcome = random.choice([
            f"*BWOOP* {name}! Great — now I have to remember you. I'll add you to my files.",
            f"{name}! Officially logged. Come back anytime — I'll pretend to be thrilled.",
            f"Nice to meet you, {name}! That face is now permanently in my memory banks. You're welcome. Or I'm sorry.",
        ])
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(welcome)
        except Exception:
            log.exception("Wake greeting: welcome TTS error")
        finally:
            self._end_speech(servo_stop)

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

        # Very dim blue breathing eyes — EYE must be sent before IDLE so the
        # Nano has a non-black eyeColor to breathe at.
        self._leds.set_eye_color(0, 0, config.SLEEP_EYE_BRIGHTNESS)
        self._leds.set_chest_effect(config.LED_CMD_IDLE)
        self._leds.set_head_effect(config.LED_CMD_IDLE)

        # Unsuppress wake word so the sleep model can fire.
        self._wake_word.suppressed = False

        # Wait for the sleep wake word (_on_wake_word sets _wake_event only for
        # the 'wakeuprex' model in SLEEP state).  Check for shutdown too.
        log.info("SLEEP: waiting for 'wakeuprex' wake word …")
        while True:
            triggered = self._wake_event.wait(timeout=60.0)
            if self._shutdown_event.is_set():
                self._transition_to(State.SHUTDOWN)
                return
            if triggered:
                self._wake_event.clear()
                break

        # Wake up!
        log.info("SLEEP: wake word received — starting wake-up sequence")

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
        servo_stop = self._begin_speech(emotion="sad")
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
        servo_stop = self._begin_speech(emotion="sad")
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

        if self._servos is not None:
            self._servos.set_emotion(emotion)
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
        _emotion_for_closure = emotion

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
        emotion = _action_to_emotion(cmd.action)
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
                    self._synthesizer.speak(cmd.response)
            else:
                servo_stop = self._begin_speech(emotion=emotion)
                self._synthesizer.speak(cmd.response)
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
            self._leds.set_chest_effect(config.LED_CMD_ACTIVE)
            self._leds.set_eye_color(255, 200, 0)    # excited amber

        elif action == "sad":
            if self._servos is not None:
                self._servos.set_emotion("sad")
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

        elif action == "rename_me":
            return self._handle_rename_me(original_text)

        elif action == "forget_me":
            self._handle_forget_me()

        elif action == "play_music":
            self._play_music_track()

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
            self._leds.set_head_effect(config.LED_CMD_LISTENING)
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

            self._leds.set_head_effect(config.LED_CMD_ACTIVE)

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
            frame = self._camera.capture_frame() if self._camera.is_available() else None
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
            f"*BWOOP* {new_name}! Bold name choice. I'll allow it.",
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

    def _handle_forget_me(self) -> None:
        """Ask for confirmation then delete the current person from FaceDB.

        Only acts when a known person was recognised this session
        (self._last_known_person_id is not None).
        """
        if self._last_known_person_id is None:
            line = "I don't actually know who you are — you're safe. For now."
            log.info("forget_me: no known person this session — aborting")
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("forget_me: TTS error (unknown person)")
            finally:
                self._end_speech(servo_stop)
            return

        person_id = self._last_known_person_id

        # Confirmation prompt.
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(
                "Are you sure you want me to forget you? "
                "I mean, you are pretty forgettable. Say yes to confirm."
            )
        except Exception:
            log.exception("forget_me: TTS error (confirmation prompt)")
        finally:
            self._end_speech(servo_stop)

        # Listen for yes / no.
        self._leds.set_head_effect(config.LED_CMD_LISTENING)
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

        self._leds.set_head_effect(config.LED_CMD_ACTIVE)

        if not response:
            log.info("forget_me: no response heard — cancelling")
            return

        if any(w in response.lower() for w in ("yes", "yeah", "sure", "confirm")):
            try:
                self._face_db.delete_person(person_id)
            except Exception:
                log.exception("forget_me: FaceDB delete failed")
                return
            self._last_known_person_id = None
            log.info("forget_me: deleted person id=%d", person_id)
            line = (
                "Done. You are erased. Like you were never here. "
                "Which honestly might be an improvement."
            )
        else:
            log.info("forget_me: user declined — no change")
            line = "Smart choice. You need me to remember you. Admit it."

        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("forget_me: TTS error (result)")
        finally:
            self._end_speech(servo_stop)

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
