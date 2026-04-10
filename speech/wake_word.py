"""
speech/wake_word.py — Continuous wake word detection using OpenWakeWord.

Runs a background thread that streams 80 ms audio chunks from the USB mic
and feeds each one to an OpenWakeWord Model loaded with two custom .onnx
models (hey_rex.onnx and hey_r3x.onnx). When either model's score exceeds
WAKE_WORD_THRESHOLD the registered callback fires with the model name, then
a cooldown period suppresses further detections.

Typical startup sequence:
    def on_wake(model_name: str):
        print(f"Wake word: {model_name}")

    detector = WakeWordDetector(on_detection=on_wake)
    if not detector.is_available():
        log.warning("No wake word models found — wake word disabled")
    else:
        detector.warmup()
        detector.start()
        ...
        detector.stop()

Echo suppression:
    detector.suppressed = True   # while Rex is speaking
    detector.suppressed = False  # when done
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
from openwakeword.model import Model as OWWModel

import config

log = logging.getLogger(__name__)


class WakeWordDetector:
    """Continuous background wake word detector.

    Thread safety: `suppressed` may be set from any thread. All other
    methods should be called from the main thread only.
    """

    def __init__(self, on_detection: Callable[[str], None]) -> None:
        """
        Args:
            on_detection: called from the audio thread when a wake word fires.
                          Receives the model name string (e.g. "hey_rex").
                          Keep it short — it blocks the audio capture loop.
        """
        self._callback = on_detection
        self._model: OWWModel | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._suppress = threading.Event()   # set → skip callbacks (Rex is speaking)
        self._last_detection: float = 0.0   # monotonic timestamp of last trigger
        # Pause/resume: _pause_event tells the audio thread to close the stream;
        # _idle_event is set by the audio thread once the stream is closed so
        # pause() can block until the mic is actually free.
        self._pause_event = threading.Event()
        self._idle_event = threading.Event()
        self._idle_event.set()   # starts idle (stream not yet open)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _active_model_paths() -> tuple:
        return (
            config.WAKE_WORD_MODEL_1,
            config.WAKE_WORD_MODEL_2,
            config.WAKE_WORD_MODEL_3,
            config.WAKE_WORD_MODEL_4,
        )

    @staticmethod
    def _sleep_model_paths() -> tuple:
        return (config.WAKE_SLEEP_MODEL,)

    def is_available(self) -> bool:
        """Return True if at least one normal wake word model exists on disk.

        Does NOT load the model. Safe to call at any time.
        """
        return any(p.exists() for p in self._active_model_paths())

    def warmup(self) -> None:
        """Load the normal wake word model(s) into memory. Call once at startup.

        Loads whichever configured non-sleep .onnx files actually exist.
        """
        self.use_active_models()

    def use_active_models(self) -> None:
        """Load the normal wake word models, excluding the sleep-only model."""
        self._swap_models(
            self._active_model_paths(),
            profile_name="active",
            missing_label="wake word model(s)",
        )

    def use_sleep_model(self) -> None:
        """Load only the sleep-mode wake word model."""
        self._swap_models(
            self._sleep_model_paths(),
            profile_name="sleep",
            missing_label="sleep wake word model",
        )

    def _swap_models(
        self,
        candidate_paths: tuple[Path, ...],
        *,
        profile_name: str,
        missing_label: str,
    ) -> None:
        """Load a specific wake-word model profile, pausing capture if needed."""
        was_running = self._thread is not None and self._thread.is_alive()
        if was_running:
            self.pause()

        paths = [p for p in candidate_paths if p.exists()]
        if not paths:
            if was_running:
                self.resume()
            raise RuntimeError(
                f"No {missing_label} found for {profile_name} profile.\n"
                + "\n".join(f"  {p}" for p in candidate_paths)
                + "\nPlace trained .onnx files at those paths or update .env."
            )

        missing = [p for p in candidate_paths if not p.exists()]
        total = len(candidate_paths)
        if missing:
            log.warning(
                "%s not found for %s profile (running with %d/%d): %s",
                missing_label.capitalize(),
                profile_name,
                len(paths),
                total,
                [str(m) for m in missing],
            )

        try:
            log.info(
                "Loading %s wake word model(s): %s",
                profile_name,
                [p.name for p in paths],
            )
            self._model = OWWModel(
                wakeword_model_paths=[str(p) for p in paths],
            )
            loaded = list(self._model.models.keys())
            log.info(
                "Wake word %s models loaded (%d/%d models): %s",
                profile_name,
                len(loaded),
                total,
                loaded,
            )
        finally:
            if was_running:
                self.resume()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background audio capture and detection thread.

        Raises RuntimeError if warmup() has not been called or if already running.
        """
        if self._model is None:
            raise RuntimeError(
                "WakeWordDetector not warmed up — call warmup() before start()."
            )
        if self._thread is not None and self._thread.is_alive():
            log.warning("WakeWordDetector already running — ignoring start()")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="djr3x-wakeword",
        )
        self._thread.start()
        log.info("Wake word detection started (threshold=%.2f, cooldown=%.1f s)",
                 config.WAKE_WORD_THRESHOLD, config.WAKE_WORD_COOLDOWN)

    def stop(self) -> None:
        """Signal the detection thread to exit and wait for it to finish."""
        self._stop_event.set()
        self._pause_event.clear()   # unblock the thread if it's waiting on pause
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                log.warning("WakeWordDetector thread did not stop cleanly within 3 s")
            self._thread = None
        log.info("Wake word detection stopped.")

    def pause(self) -> None:
        """Close the mic stream without stopping the detector thread.

        Blocks until the audio thread confirms the stream is closed (at most
        one 80 ms read chunk), then waits an additional brief moment so that
        PipeWire fully releases the device at the OS level before the caller
        opens its own stream on the same device.  Safe to call from any thread.
        """
        if self._thread is None or not self._thread.is_alive():
            log.debug("Wake word pause(): thread not running, nothing to pause")
            return
        log.debug("Wake word pause(): requesting mic release …")
        self._idle_event.clear()
        self._pause_event.set()
        released = self._idle_event.wait(timeout=1.0)
        if released:
            log.debug("Wake word pause(): mic stream closed — device released")
        else:
            log.warning("Wake word pause(): timed out (1 s) waiting for mic stream to close")
        # Give PipeWire a moment to fully release the device at the OS level
        # before the transcriber opens its own InputStream on the same device.
        time.sleep(0.05)

    def resume(self) -> None:
        """Reopen the mic stream and resume detection.

        The audio thread reopens the InputStream on its next loop iteration.
        Safe to call from any thread.
        """
        if self._thread is None or not self._thread.is_alive():
            log.debug("Wake word resume(): thread not running, nothing to resume")
            return
        log.debug("Wake word resume(): clearing pause flag — stream will reopen on next loop")
        self._pause_event.clear()

    # ------------------------------------------------------------------
    # Echo suppression
    # ------------------------------------------------------------------

    @property
    def suppressed(self) -> bool:
        """True while callbacks are suppressed (e.g. Rex is speaking)."""
        return self._suppress.is_set()

    @suppressed.setter
    def suppressed(self, value: bool) -> None:
        """Set suppression from any thread. Thread-safe."""
        if value:
            self._suppress.set()
        else:
            self._suppress.clear()

    # ------------------------------------------------------------------
    # Internal — audio thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background thread: capture mic → predict → callback.

        The outer loop lets pause()/resume() close and reopen the InputStream
        without killing this thread — the mic is released while the transcriber
        holds it and reclaimed once transcription is done.
        """
        _open_attempts = 0
        while not self._stop_event.is_set():
            # Paused: stream is (or should be) closed — signal idle and wait.
            if self._pause_event.is_set():
                self._idle_event.set()
                time.sleep(0.02)
                _open_attempts = 0
                continue

            _stream_opened = False
            try:
                self._idle_event.clear()
                with _open_input_stream() as stream:
                    _stream_opened = True
                    _open_attempts = 0
                    log.debug("Wake word audio stream open (chunk=%d samples, %.0f ms)",
                              config.WAKE_WORD_CHUNK_SIZE,
                              config.WAKE_WORD_CHUNK_SIZE / config.AUDIO_SAMPLE_RATE * 1000)

                    while not self._stop_event.is_set() and not self._pause_event.is_set():
                        frames, overflowed = stream.read(config.WAKE_WORD_CHUNK_SIZE)
                        if overflowed:
                            log.debug("Wake word stream: buffer overflow")

                        # Mix down to 1-D mono int16 (OpenWakeWord requirement)
                        audio = frames.astype(np.float32).mean(axis=1).astype(np.int16)

                        scores: dict[str, float] = self._model.predict(audio)

                        now = time.monotonic()

                        # cooldown guard
                        if now - self._last_detection < config.WAKE_WORD_COOLDOWN:
                            continue

                        # echo suppression guard
                        if self._suppress.is_set():
                            continue

                        # check all loaded models; fire on first threshold cross
                        for model_name, score in scores.items():
                            if score >= config.WAKE_WORD_THRESHOLD:
                                log.info(
                                    "Wake word detected: %r  score=%.3f  (threshold=%.2f)",
                                    model_name, score, config.WAKE_WORD_THRESHOLD,
                                )
                                self._last_detection = now
                                try:
                                    self._callback(model_name)
                                except Exception:
                                    log.exception(
                                        "Exception in wake word callback for %r", model_name
                                    )
                                break   # one callback per chunk maximum

            except sd.PortAudioError:
                if not _stream_opened:
                    _open_attempts += 1
                    if _open_attempts <= 3:
                        log.warning(
                            "Wake word: mic open failed (attempt %d/3) — retrying in 2 s",
                            _open_attempts,
                        )
                        time.sleep(2.0)
                        continue
                log.exception("Wake word: microphone error — detection thread exiting")
                break
            except Exception:
                log.exception("Wake word: unexpected error in detection thread")
                break
            finally:
                log.debug("Wake word: mic stream closed (finally block — idle_event will be set)")
                self._idle_event.set()   # stream is closed; pause() may unblock


def _input_devices_to_try() -> list[int | None]:
    device = config.AUDIO_INPUT_DEVICE
    return [device, None] if device is not None else [None]


def _open_input_stream():
    last_exc: sd.PortAudioError | None = None
    for device in _input_devices_to_try():
        try:
            if device is None and config.AUDIO_INPUT_DEVICE is not None:
                log.warning("Wake word: falling back to default input device")
            return sd.InputStream(
                samplerate=config.AUDIO_SAMPLE_RATE,
                channels=config.MIC_CHANNELS,
                dtype="int16",
                device=device,
                blocksize=config.WAKE_WORD_CHUNK_SIZE,
            )
        except sd.PortAudioError as exc:
            last_exc = exc
            log.warning("Wake word: input device %s failed — %s", device, exc)
    assert last_exc is not None
    raise last_exc
