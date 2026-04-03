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

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if at least one wake word model file exists on disk.

        Does NOT load the model. Safe to call at any time.
        """
        return config.WAKE_WORD_MODEL_1.exists() or config.WAKE_WORD_MODEL_2.exists()

    def warmup(self) -> None:
        """Load wake word model(s) into memory. Call once at startup.

        Loads whichever of the two configured .onnx files actually exist.
        Raises RuntimeError if no model files are found.
        Raises RuntimeError if called more than once.
        """
        if self._model is not None:
            log.warning("WakeWordDetector.warmup() called more than once — ignoring")
            return

        paths = [
            p for p in (config.WAKE_WORD_MODEL_1, config.WAKE_WORD_MODEL_2)
            if p.exists()
        ]
        if not paths:
            raise RuntimeError(
                "No wake word model files found.\n"
                f"  Model 1: {config.WAKE_WORD_MODEL_1}\n"
                f"  Model 2: {config.WAKE_WORD_MODEL_2}\n"
                "Place trained .onnx files at those paths or update .env."
            )

        missing = [
            p for p in (config.WAKE_WORD_MODEL_1, config.WAKE_WORD_MODEL_2)
            if not p.exists()
        ]
        if missing:
            log.warning(
                "Wake word model(s) not found (running with %d/%d): %s",
                len(paths), 2, [str(m) for m in missing],
            )

        log.info(
            "Loading %d wake word model(s): %s",
            len(paths), [p.name for p in paths],
        )
        self._model = OWWModel(
            wakeword_model_paths=[str(p) for p in paths],
            inference_framework="onnx",
        )
        loaded = list(self._model.models.keys())
        log.info("Wake word models loaded: %s", loaded)

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
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                log.warning("WakeWordDetector thread did not stop cleanly within 3 s")
            self._thread = None
        log.info("Wake word detection stopped.")

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
        """Background thread: capture mic → predict → callback."""
        try:
            with sd.InputStream(
                samplerate=config.AUDIO_SAMPLE_RATE,
                channels=config.AUDIO_CHANNELS,
                dtype="int16",
                device=config.MIC_DEVICE_INDEX,
                blocksize=config.WAKE_WORD_CHUNK_SIZE,
            ) as stream:
                log.debug("Wake word audio stream open (chunk=%d samples, %.0f ms)",
                          config.WAKE_WORD_CHUNK_SIZE,
                          config.WAKE_WORD_CHUNK_SIZE / config.AUDIO_SAMPLE_RATE * 1000)

                while not self._stop_event.is_set():
                    frames, overflowed = stream.read(config.WAKE_WORD_CHUNK_SIZE)
                    if overflowed:
                        log.debug("Wake word stream: buffer overflow")

                    # flatten to 1-D mono int16 (OpenWakeWord requirement)
                    audio = frames[:, 0]

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
            log.exception("Wake word: microphone error — detection thread exiting")
        except Exception:
            log.exception("Wake word: unexpected error in detection thread")
