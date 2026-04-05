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

log = logging.getLogger(__name__)

# How often the servo-speak worker calls speak_move() during TTS (~20 Hz).
_SERVO_SPEAK_INTERVAL: float = 0.05

_ARE_YOU_THERE_PHRASES: list[str] = [
    "Are you there?",
    "Hello?",
    "Did you need something?",
    "I'm listening...",
    "Uh... hello?",
]

_GOODBYE_PHRASES: list[str] = [
    "Ok, nevermind!",
    "Alright, catch you later!",
    "I'll be here if you need me.",
    "Ok, going back to sleep!",
]

_SHUTDOWN_PHRASES: list[str] = [
    "Shutting down, catch you on the flip side!",
    "Going offline, keep the music alive!",
    "Powering down, it's been a blast lifeform!",
    "See you in the next galaxy, signing off!",
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
        """Play the startup (boot) animation and block until it completes.

        Must be called *before* start() so the servo idle thread is not
        running yet — startup moves all channels including arms.
        """
        log.info("Playing startup animation …")
        self._animations.play_startup()
        self._animations.wait(timeout=10.0)
        log.info("Startup animation complete.")

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
            "servos":       self._servos is not None,
            "chest_leds":   self._leds._chest is not None,
            "head_leds":    self._leds._head is not None,
            "transcriber":  self._transcriber.is_available(),
            "wake_word":    self._wake_word.is_available(),
            "wake_models":  wake_models,       # int: 0, 1, or 2
            "music_tracks": len(self._music_tracks),
            "camera":       self._camera.is_available(),
        }

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
                servo_stop = self._begin_speech(emotion="neutral")
                try:
                    self._player.play_file(clip_path)
                except Exception:
                    log.exception("Idle clip playback error: %s", clip_path.name)
                finally:
                    self._end_speech(servo_stop)

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
                self._leds.set_head_effect(config.LED_CMD_SPEAKING)
                servo_stop = self._begin_speech()
                try:
                    self._synthesizer.speak(prompt)
                finally:
                    self._end_speech(servo_stop)
                self._player.wait_for_speech()

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
                    self._leds.set_head_effect(config.LED_CMD_SPEAKING)
                    servo_stop = self._begin_speech()
                    try:
                        self._synthesizer.speak(goodbye)
                    finally:
                        self._end_speech(servo_stop)
                    self._player.wait_for_speech()
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
            self._leds.set_head_effect(config.LED_CMD_SPEAKING)
            self._leds.set_chest_effect(config.LED_CMD_SPEAKING)

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
                next_state = self._execute_command(cmd)
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
        servo_stop = self._begin_speech(emotion="neutral")
        try:
            self._synthesizer.speak(phrase)
        except Exception:
            log.exception("Shutdown speech: TTS error — continuing to shutdown")
        finally:
            self._end_speech(servo_stop)

        # Stop servo idle thread before the animation so arm channels are free.
        if self._servos is not None:
            self._servos.stop()

        # Theatrical power-down sequence (blocking — 2.2 s).
        self._animations.play_shutdown()

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
        self._state = new_state

    # ------------------------------------------------------------------
    # Wake word callback  (called from WakeWordDetector audio thread)
    # ------------------------------------------------------------------

    def _on_wake_word(self, model_name: str) -> None:
        if self._state == State.IDLE:
            self._pipeline_t0 = time.monotonic()
            log.info("Wake word detected (%s)", model_name)
            self._wake_event.set()
        # Ignore detections in ACTIVE/SHUTDOWN (suppression should already
        # block the callback, but this is a belt-and-suspenders guard).

    # ------------------------------------------------------------------
    # Speech helpers
    # ------------------------------------------------------------------

    def _play_wake_greeting(self) -> None:
        """Greet the user on wake word.

        GREETER_PROBABILITY chance: play a holding clip while concurrently
        capturing a frame and generating a personalized Rex-style greeting via
        gpt-4o.  When both finish, speak the personalized greeting.  If the
        clip ends before the greeting is ready, wait silently then speak.  If
        the greeting fails for any reason, fall back to the canned greeting.

        Falls back to a simple canned greeting if the camera is unavailable,
        the API call fails, or the random roll doesn't land.

        Always completes (audio finishes) before returning so the caller can
        open the mic immediately afterwards.
        """
        do_personalized = (
            random.random() < config.GREETER_PROBABILITY
            and self._camera.is_available()
        )

        if do_personalized:
            log.info("Wake greeting: attempting personalized greeting")
            _HOLDING_CLIP = config.ASSETS_DIR / "audio" / "This is your cap.mp3"
            greeting_result: list[str | None] = [None]

            def _generate_personalized() -> None:
                frame = self._camera.capture_frame()
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
                # Play holding clip while camera capture + GPT-4o calls run in background.
                if _HOLDING_CLIP.exists():
                    self._player.play_file(_HOLDING_CLIP)
                # Clip done (or missing) — wait silently if greeting isn't ready yet.
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

        # Simple canned greeting — audio file or short TTS line.
        _CANNED_AUDIO = config.ASSETS_DIR / "audio" / "Hi There.mp3"
        _CANNED_TTS = [
            "Hey hey hey!",
            "What's up, lifeform!",
            "HEY! You're back!",
            "*BWOOP* Hello there!",
            "Oga's Cantina is OPEN!",
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

    def _play_return_to_idle_chime(self) -> None:
        """Play the startup chime to signal Rex is done listening, then wait
        for it to finish before entering IDLE."""
        self._player.play_chime()
        self._player.wait_for_music(timeout=10.0)

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
            # Raise elbow to speaking position so speak_move() has it
            # starting from a raised pose rather than the idle lowered rest.
            self._servos.set_channel_speed(
                config.SERVO_ARM_LEFT, config.SERVO_DEFAULT_SPEED
            )
            self._servos.set_position(config.SERVO_ARM_LEFT, 7100)

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

    def _speak_llm(self, text: str, image: str | None = None, t0: float | None = None) -> State | None:
        """Stream text (and optional vision frame) through ChatGPT → ElevenLabs."""
        servo_stop = self._begin_speech(emotion="neutral")
        try:
            tokens = self._llm.chat_stream(text, image=image, t0=t0)
            self._synthesizer.speak_stream(tokens, t0=t0)
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
