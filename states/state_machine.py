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
    sm.start()   # warmup — loads Vosk and wake word models
    sm.run()     # blocks until SHUTDOWN; calls close() in a finally block
"""

from __future__ import annotations

import enum
import logging
import os
import threading
import time
from pathlib import Path

import config
from audio.player import AudioPlayer
from commands.parser import parse
from hardware.leds import LEDController
from hardware.servos import ServoController
from llm.chatgpt import ChatGPTClient
from speech.synthesizer import Synthesizer
from speech.transcriber import Transcriber
from speech.wake_word import WakeWordDetector

log = logging.getLogger(__name__)

# How often the servo-speak worker calls speak_move() during TTS (~20 Hz).
_SERVO_SPEAK_INTERVAL: float = 0.05


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class State(enum.Enum):
    IDLE     = "idle"
    ACTIVE   = "active"
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

        # State control
        self._state: State = State.IDLE
        self._wake_event = threading.Event()    # set by wake word callback
        self._shutdown_event = threading.Event()

        # Music library — scanned once at startup
        self._music_tracks: list[Path] = _scan_music()
        self._music_index: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Warmup all subsystems. Call once before run()."""
        log.info("StateMachine: warming up subsystems …")

        if self._transcriber.is_available():
            self._transcriber.warmup()
        else:
            log.warning(
                "Vosk model not found at %s — transcription disabled",
                config.VOSK_MODEL_PATH,
            )

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

    def request_shutdown(self) -> None:
        """Thread-safe: schedule a transition to SHUTDOWN from any thread
        (e.g. a SIGINT handler in main.py)."""
        log.info("StateMachine: shutdown requested externally")
        self._shutdown_event.set()
        self._wake_event.set()   # unblock _run_idle() if it's waiting

    # ------------------------------------------------------------------
    # State — IDLE
    # ------------------------------------------------------------------

    def _run_idle(self) -> None:
        """Enter IDLE: set LEDs/servos to rest, then wait for a wake word."""
        log.info("→ IDLE")

        self._leds.set_chest_effect(config.LED_CMD_IDLE)
        self._leds.set_head_effect(config.LED_CMD_IDLE)
        self._leds.set_eye_color(0, 80, 255)          # calm blue

        if self._servos is not None:
            self._servos.set_emotion("neutral")

        # Unsuppress wake word (may have been left suppressed if we cut off
        # speech mid-utterance, e.g. via request_shutdown).
        self._wake_word.suppressed = False

        # Reset conversation context so each active session starts fresh.
        self._llm.clear_history()

        # Block until a wake word fires (or shutdown is requested).
        self._wake_event.wait()
        self._wake_event.clear()

        if self._shutdown_event.is_set():
            self._transition_to(State.SHUTDOWN)
        else:
            self._transition_to(State.ACTIVE)

    # ------------------------------------------------------------------
    # State — ACTIVE
    # ------------------------------------------------------------------

    def _run_active(self) -> None:
        """Full interactivity loop: transcribe → parse/LLM → speak; repeat
        until ACTIVE_IDLE_TIMEOUT or an explicit state-change action."""
        log.info("→ ACTIVE")

        self._leds.set_chest_effect(config.LED_CMD_ACTIVE)
        self._leds.set_head_effect(config.LED_CMD_ACTIVE)
        self._leds.set_eye_color(255, 140, 0)          # warm amber

        if self._servos is not None:
            self._servos.set_emotion("neutral")

        last_interaction = time.monotonic()

        while self._state == State.ACTIVE:
            # --- Idle timeout ---
            elapsed = time.monotonic() - last_interaction
            if elapsed >= config.ACTIVE_IDLE_TIMEOUT:
                log.info("Active idle timeout (%.0f s) — returning to IDLE", elapsed)
                self._transition_to(State.IDLE)
                break

            if not self._transcriber.is_available():
                # No Vosk model: nothing to transcribe — just wait for timeout.
                time.sleep(1.0)
                continue

            # --- Listen indicator ---
            self._leds.set_head_effect(config.LED_CMD_LISTENING)

            # --- Transcribe (blocks until silence or MAX_RECORD_SECONDS) ---
            try:
                text = self._transcriber.transcribe()
            except Exception:
                log.exception("Transcription error — skipping utterance")
                continue

            if not text:
                log.debug("Empty transcription — waiting again")
                continue

            last_interaction = time.monotonic()
            log.info("Transcribed: %r", text)

            # --- Speaking indicator ---
            self._leds.set_head_effect(config.LED_CMD_SPEAKING)
            self._leds.set_chest_effect(config.LED_CMD_SPEAKING)

            # --- Parse and respond ---
            cmd = parse(text)
            if cmd is not None:
                log.info("Matched command: phrases[0]=%r action=%r", cmd.phrases[0], cmd.action)
                next_state = self._execute_command(cmd)
            else:
                log.info("No command match — routing to LLM")
                next_state = self._speak_llm(text)

            if next_state is not None:
                self._transition_to(next_state)
                break

            # Restore ACTIVE indicators for next listen turn
            self._leds.set_chest_effect(config.LED_CMD_ACTIVE)

    # ------------------------------------------------------------------
    # State — SHUTDOWN
    # ------------------------------------------------------------------

    def _run_shutdown(self) -> None:
        """Fade out, home hardware, then halt the OS."""
        log.info("→ SHUTDOWN")

        # LEDs off
        self._leds.set_chest_effect(config.LED_CMD_OFF)
        self._leds.set_head_effect(config.LED_CMD_OFF)

        # Servos to neutral home
        if self._servos is not None:
            self._servos.stop()    # stop idle-motion thread
            self._servos.home()

        # Halt — requires passwordless sudo (add to /etc/sudoers on the Pi)
        log.info("StateMachine: halting OS — sudo shutdown -h now")
        os.system("sudo shutdown -h now")

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: State) -> None:
        log.info("Transition: %s → %s", self._state.value, new_state.value)
        if new_state == State.SHUTDOWN:
            self._shutdown_event.set()
        self._state = new_state

    # ------------------------------------------------------------------
    # Wake word callback  (called from WakeWordDetector audio thread)
    # ------------------------------------------------------------------

    def _on_wake_word(self, model_name: str) -> None:
        if self._state == State.IDLE:
            log.info("Wake word detected (%s)", model_name)
            self._wake_event.set()
        # Ignore detections in ACTIVE/SHUTDOWN (suppression should already
        # block the callback, but this is a belt-and-suspenders guard).

    # ------------------------------------------------------------------
    # Speech helpers
    # ------------------------------------------------------------------

    def _begin_speech(self, emotion: str = "neutral") -> threading.Event:
        """Prepare hardware for a speech output burst.

        - Suppresses wake word so Rex doesn't trigger on his own voice.
        - Sets servo emotion so speak_move() uses the right position range.
        - Starts the mouth brightness thread (LEDController).
        - Launches a short-lived servo-speak worker thread.

        Returns the stop Event for the servo-speak worker; caller must set it
        when speech ends.
        """
        self._wake_word.suppressed = True

        if self._servos is not None:
            self._servos.set_emotion(emotion)

        self._leds.start_mouth()

        stop_event = threading.Event()
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

    def _execute_command(self, cmd) -> State | None:
        """Speak the response and execute the hardware action.

        Returns the next State if a state transition is required, else None.
        The emotion for servo speech range is derived from the action so that
        e.g. a greeting with action="excited" has the head moving in the
        excited range *during* the reply.
        """
        emotion = _action_to_emotion(cmd.action)
        servo_stop = self._begin_speech(emotion=emotion)

        try:
            if cmd.audio:
                audio_path = config.ASSETS_DIR / "audio" / cmd.audio
                if audio_path.exists():
                    self._player.play_file(audio_path)
                else:
                    log.warning(
                        "Pre-rendered audio not found: %s — falling back to TTS",
                        cmd.audio,
                    )
                    self._synthesizer.speak(cmd.response)
            else:
                self._synthesizer.speak(cmd.response)
        except Exception:
            log.exception("Error speaking command response")
        finally:
            self._end_speech(servo_stop)

        return self._dispatch_action(cmd.action)

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    def _speak_llm(self, text: str) -> State | None:
        """Stream text through ChatGPT → ElevenLabs. Returns None (no transition)."""
        servo_stop = self._begin_speech(emotion="neutral")
        try:
            tokens = self._llm.chat_stream(text)
            self._synthesizer.speak_stream(tokens)
        except Exception:
            log.exception("LLM/TTS error for: %.60s", text)
        finally:
            self._end_speech(servo_stop)
        return None

    # ------------------------------------------------------------------
    # Action dispatcher
    # ------------------------------------------------------------------

    def _dispatch_action(self, action: str | None) -> State | None:
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

        elif action == "idle":
            return State.IDLE

        elif action == "shutdown":
            return State.SHUTDOWN

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
