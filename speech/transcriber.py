"""
speech/transcriber.py — One-shot mic capture and local transcription via Vosk.

Typical call sequence at startup:
    t = Transcriber()
    if not t.is_available():
        sys.exit("Vosk model missing — see assets/models/")
    t.warmup()            # loads model into RAM once, ~1–2 s

Then per utterance (after wake word fires):
    text = t.transcribe() # record → silence-gate → Vosk → str
    if text:
        ...

Silence gating
--------------
Recording stops when the microphone RMS falls below SILENCE_THRESHOLD for
SILENCE_DURATION consecutive seconds, OR when MAX_RECORD_SECONDS elapses.
The silence clock only starts after the first speech chunk is detected so
that an initial quiet moment before the user speaks doesn't cut off early.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import sounddevice as sd
import vosk

import config

log = logging.getLogger(__name__)

# Chunks of silence required to end the recording.
# Computed at import time from config so it's visible in debugger.
_SILENCE_CHUNKS_NEEDED: int = max(
    1,
    round(config.SILENCE_DURATION * config.AUDIO_SAMPLE_RATE / config.AUDIO_CHUNK_SIZE),
)
_MAX_CHUNKS: int = round(
    config.MAX_RECORD_SECONDS * config.AUDIO_SAMPLE_RATE / config.AUDIO_CHUNK_SIZE
)


class Transcriber:
    """Local speech-to-text using Vosk.

    Not thread-safe: only one transcribe() call may run at a time.
    """

    def __init__(self) -> None:
        self._model: vosk.Model | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the Vosk model directory exists on disk.

        Does NOT attempt to load the model. Safe to call at any time.
        """
        return Path(config.VOSK_MODEL_PATH).is_dir()

    def warmup(self) -> None:
        """Load the Vosk model into RAM. Call once at startup.

        Loading takes 1–2 seconds on a Pi 4. After this returns, the first
        transcribe() call will not pay a cold-load penalty.

        Raises FileNotFoundError if the model directory is absent.
        Raises RuntimeError if called more than once (safe to guard against
        accidental double-init).
        """
        if self._model is not None:
            log.warning("Transcriber.warmup() called more than once — ignoring")
            return

        model_path = Path(config.VOSK_MODEL_PATH)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Vosk model not found at '{model_path}'.\n"
                "Download a model from https://alphacephei.com/vosk/models\n"
                f"and unzip it to '{model_path}'."
            )

        vosk.SetLogLevel(config.VOSK_LOG_LEVEL)
        log.info("Loading Vosk model from %s …", model_path)
        self._model = vosk.Model(str(model_path))
        log.info("Vosk model ready.")

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(self) -> str:
        """Capture one utterance from the microphone and return its text.

        Opens the mic, reads chunks, feeds them to Vosk, and stops when
        silence is detected or the hard time cap is reached. Returns the
        final Vosk transcription as a stripped string (may be empty if
        Vosk heard nothing recognisable).

        Raises RuntimeError if warmup() has not been called.
        Raises sounddevice.PortAudioError if the mic cannot be opened.
        """
        if self._model is None:
            raise RuntimeError(
                "Transcriber not warmed up — call warmup() before transcribe()."
            )

        recognizer = vosk.KaldiRecognizer(self._model, config.AUDIO_SAMPLE_RATE)
        recognizer.SetWords(False)          # no word-level timing, faster
        recognizer.SetMaxAlternatives(0)    # single best hypothesis only

        silence_chunks: int = 0
        speech_started: bool = False

        log.debug("Transcriber: mic open, listening …")

        with sd.InputStream(
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype="int16",
            device=config.MIC_DEVICE_INDEX,
            blocksize=config.AUDIO_CHUNK_SIZE,
        ) as stream:
            for _ in range(_MAX_CHUNKS):
                frames, overflowed = stream.read(config.AUDIO_CHUNK_SIZE)
                if overflowed:
                    log.debug("Transcriber: audio buffer overflowed (input too slow)")

                # frames shape: (AUDIO_CHUNK_SIZE, AUDIO_CHANNELS) dtype int16
                samples = frames[:, 0]  # flatten to 1-D mono array
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

                if rms >= config.SILENCE_THRESHOLD:
                    if not speech_started:
                        log.debug("Transcriber: speech start detected (rms=%.0f)", rms)
                    speech_started = True
                    silence_chunks = 0
                elif speech_started:
                    silence_chunks += 1

                recognizer.AcceptWaveform(samples.tobytes())

                if speech_started and silence_chunks >= _SILENCE_CHUNKS_NEEDED:
                    log.debug(
                        "Transcriber: silence end-of-speech "
                        "(%d chunks ≥ %d needed)",
                        silence_chunks, _SILENCE_CHUNKS_NEEDED,
                    )
                    break
            else:
                log.debug("Transcriber: hit MAX_RECORD_SECONDS cap (%.1f s)", config.MAX_RECORD_SECONDS)

        result = json.loads(recognizer.FinalResult())
        text = result.get("text", "").strip()
        log.debug("Vosk result: %r", text)
        return text
