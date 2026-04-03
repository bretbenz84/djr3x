"""
speech/transcriber.py — One-shot mic capture and cloud transcription via OpenAI Whisper.

Typical call sequence at startup:
    t = Transcriber()
    t.warmup()       # no-op — kept for interface compatibility
    text = t.transcribe()

Then per utterance (after wake word fires):
    text = t.transcribe()   # record → silence-gate → Whisper API → str
    if text:
        ...

Silence gating
--------------
Recording stops when the microphone RMS falls below SILENCE_THRESHOLD for
SILENCE_DURATION consecutive seconds, OR when MAX_RECORD_SECONDS elapses.
The silence clock only starts after the first speech chunk is detected so
that an initial quiet moment before the user speaks doesn't cut off early.
If no speech is detected at all (entire recording is below threshold), the
buffer is discarded and an empty string is returned without an API call.
"""

from __future__ import annotations

import io
import logging
import wave

import numpy as np
import sounddevice as sd
from openai import OpenAI

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
    """Cloud speech-to-text using OpenAI Whisper.

    Not thread-safe: only one transcribe() call may run at a time.
    """

    def __init__(self) -> None:
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Always True — Whisper is accessed via the OpenAI API; no local
        model files are required."""
        return True

    def warmup(self) -> None:
        """No-op — Whisper requires no local model loading.

        Kept so the startup sequence in StateMachine.start() can call
        warmup() uniformly across all subsystems without special-casing
        the transcriber.
        """

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(self, wait_for_speech_seconds: float | None = None) -> str | None:
        """Capture one utterance from the microphone and return its text.

        Opens the mic, reads chunks using the same silence-gating logic as
        the former Vosk implementation, then encodes the buffer as a WAV
        and sends it to the Whisper API. Returns the transcription as a
        stripped string, or empty string if no speech was detected, or the
        API call fails.

        If wait_for_speech_seconds is given, returns None (without an API
        call) if speech has not started within that many seconds. This lets
        the caller distinguish "no speech in the window" from "speech was
        detected but Whisper returned nothing".

        Raises sounddevice.PortAudioError if the mic cannot be opened.
        """
        frames_list: list[np.ndarray] = []
        silence_chunks: int = 0
        speech_started: bool = False

        wait_chunks: int | None = (
            round(wait_for_speech_seconds * config.AUDIO_SAMPLE_RATE / config.AUDIO_CHUNK_SIZE)
            if wait_for_speech_seconds is not None
            else None
        )

        log.debug("Transcriber: mic open, listening …")

        with sd.InputStream(
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype="int16",
            device=config.AUDIO_INPUT_DEVICE,
            blocksize=config.AUDIO_CHUNK_SIZE,
        ) as stream:
            for chunk_index in range(_MAX_CHUNKS):
                frames, overflowed = stream.read(config.AUDIO_CHUNK_SIZE)
                if overflowed:
                    log.debug("Transcriber: audio buffer overflowed (input too slow)")

                # frames shape: (AUDIO_CHUNK_SIZE, AUDIO_CHANNELS) dtype int16
                samples = frames[:, 0]  # flatten to 1-D mono array
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

                if rms >= config.TRANSCRIBE_SPEECH_THRESHOLD:
                    if not speech_started:
                        log.debug("Transcriber: speech start detected (rms=%.0f)", rms)
                    speech_started = True
                    silence_chunks = 0
                elif speech_started:
                    silence_chunks += 1

                frames_list.append(samples)

                if speech_started and silence_chunks >= _SILENCE_CHUNKS_NEEDED:
                    log.debug(
                        "Transcriber: silence end-of-speech "
                        "(%d chunks ≥ %d needed)",
                        silence_chunks, _SILENCE_CHUNKS_NEEDED,
                    )
                    break

                # Early exit: speech hasn't started and the caller's wait window expired.
                if not speech_started and wait_chunks is not None and chunk_index + 1 >= wait_chunks:
                    log.debug(
                        "Transcriber: no speech within %.1f s — returning None",
                        wait_for_speech_seconds,
                    )
                    return None
            else:
                log.debug(
                    "Transcriber: hit MAX_RECORD_SECONDS cap (%.1f s)",
                    config.MAX_RECORD_SECONDS,
                )

        # Nothing above the silence threshold — skip the API call entirely.
        if not speech_started:
            return ""

        # Encode the captured buffer as a WAV in memory and send to Whisper.
        pcm = np.concatenate(frames_list)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(config.AUDIO_CHANNELS)
            wf.setsampwidth(2)   # int16 = 2 bytes per sample
            wf.setframerate(config.AUDIO_SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)

        try:
            result = self._client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", buf.read()),
                language=config.WHISPER_LANGUAGE,
            )
            text = result.text.strip()
        except Exception:
            log.exception("Whisper transcription API error")
            return ""

        log.debug("Whisper result: %r", text)
        return text
