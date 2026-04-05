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
import time
import wave

import numpy as np
import sounddevice as sd
from openai import OpenAI

import config

log = logging.getLogger(__name__)

# Whisper-specific recording limits — tighter than the generic caps to reduce
# the audio sent to the API and lower end-to-end latency.
_SILENCE_CHUNKS_NEEDED: int = max(
    1,
    round(config.WHISPER_SILENCE_DURATION * config.AUDIO_SAMPLE_RATE / config.AUDIO_CHUNK_SIZE),
)
_MAX_CHUNKS: int = round(
    config.WHISPER_MAX_RECORD_SECONDS * config.AUDIO_SAMPLE_RATE / config.AUDIO_CHUNK_SIZE
)


class Transcriber:
    """Cloud speech-to-text using OpenAI Whisper.

    Not thread-safe: only one transcribe() call may run at a time.
    """

    def __init__(self) -> None:
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        # Speech detection threshold — may be overridden by calibrate_noise_floor().
        self._speech_threshold: int = config.TRANSCRIBE_SPEECH_THRESHOLD

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

    def calibrate_noise_floor(
        self,
        duration: float | None = None,
    ) -> None:
        """Record ambient silence and set the speech detection threshold.

        threshold = clamp(
            noise_rms * NOISE_FLOOR_MULTIPLIER,
            TRANSCRIBE_SPEECH_THRESHOLD_MIN,
            TRANSCRIBE_SPEECH_THRESHOLD_MAX,
        )

        Call once at startup (after the mic device is known to be free) so the
        threshold adapts to the actual room noise rather than relying on the
        fixed config default.  Safe to call again if the environment changes.

        Raises sounddevice.PortAudioError if the mic cannot be opened.
        """
        if duration is None:
            duration = config.TRANSCRIBE_NOISE_FLOOR_DURATION
        n_frames = round(duration * config.AUDIO_SAMPLE_RATE)
        log.info(
            "Calibrating noise floor (%.1f s on device %s) …",
            duration, config.AUDIO_INPUT_DEVICE,
        )
        try:
            raw = sd.rec(
                n_frames,
                samplerate=config.AUDIO_SAMPLE_RATE,
                channels=config.AUDIO_INPUT_CHANNELS,
                dtype="int16",
                device=config.AUDIO_INPUT_DEVICE,
                blocking=True,
            )
        except sd.PortAudioError:
            log.warning(
                "Noise floor calibration failed — keeping threshold at %d",
                self._speech_threshold,
            )
            return

        mono = raw[:, 0].astype(np.float32)
        noise_rms = float(np.sqrt(np.mean(mono ** 2)))
        raw_threshold = int(noise_rms * config.NOISE_FLOOR_MULTIPLIER)
        new_threshold = max(
            config.TRANSCRIBE_SPEECH_THRESHOLD_MIN,
            min(config.TRANSCRIBE_SPEECH_THRESHOLD_MAX, raw_threshold),
        )
        log.info(
            "Noise floor calibration complete: "
            "RMS=%.0f × %.1f = %d → clamped to %d "
            "(min=%d max=%d, was %d)",
            noise_rms, config.NOISE_FLOOR_MULTIPLIER, raw_threshold, new_threshold,
            config.TRANSCRIBE_SPEECH_THRESHOLD_MIN,
            config.TRANSCRIBE_SPEECH_THRESHOLD_MAX,
            self._speech_threshold,
        )
        self._speech_threshold = new_threshold

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        wait_for_speech_seconds: float | None = None,
        t0: float | None = None,
    ) -> str | None:
        """Capture one utterance from the microphone and return its text.

        Opens the mic, reads chunks using silence-gating logic, then encodes
        the buffer as a WAV and sends it to the Whisper API. Trims leading and
        trailing silence before sending so dead air doesn't inflate latency.

        Returns the transcription as a stripped string, empty string if no
        speech was detected, or the API call fails.

        If wait_for_speech_seconds is given, returns None (without an API
        call) if speech has not started within that many seconds. This lets
        the caller distinguish "no speech in the window" from "speech was
        detected but Whisper returned nothing".

        t0: optional monotonic timestamp from wake word detection; when
        provided, all INFO log lines include an elapsed-since-wake marker.

        Raises sounddevice.PortAudioError if the mic cannot be opened.
        """
        frames_list: list[np.ndarray] = []
        silence_chunks: int = 0
        speech_started: bool = False
        consecutive_speech: int = 0        # chunks above threshold in a row
        speech_first_chunk: int | None = None   # first chunk at/above threshold

        wait_chunks: int | None = (
            round(wait_for_speech_seconds * config.AUDIO_SAMPLE_RATE / config.AUDIO_CHUNK_SIZE)
            if wait_for_speech_seconds is not None
            else None
        )

        def _mark(label: str = "") -> str:
            """Return ' [+X.Xs]' elapsed marker when t0 is set, else ''."""
            if t0 is None:
                return ""
            return f" [+{time.monotonic() - t0:.1f}s]"

        log.debug(
            "Transcriber: opening InputStream (device=%s, rate=%d Hz, "
            "chunk=%d frames, threshold=%d)",
            config.AUDIO_INPUT_DEVICE, config.AUDIO_SAMPLE_RATE,
            config.AUDIO_CHUNK_SIZE, self._speech_threshold,
        )

        _SENTINEL = object()
        _early_return: object = _SENTINEL  # set to None if early exit triggered
        for _attempt in range(3):
            frames_list = []
            silence_chunks = 0
            speech_started = False
            consecutive_speech = 0
            speech_first_chunk = None
            try:
                with sd.InputStream(
                    samplerate=config.AUDIO_SAMPLE_RATE,
                    channels=config.AUDIO_INPUT_CHANNELS,
                    dtype="int16",
                    device=config.AUDIO_INPUT_DEVICE,
                    blocksize=config.AUDIO_CHUNK_SIZE,
                ) as stream:
                    log.info("Mic open, listening …%s", _mark())
                    for chunk_index in range(_MAX_CHUNKS):
                        frames, overflowed = stream.read(config.AUDIO_CHUNK_SIZE)
                        if overflowed:
                            log.debug("Transcriber: audio buffer overflowed (input too slow)")

                        # frames shape: (AUDIO_CHUNK_SIZE, AUDIO_CHANNELS) dtype int16
                        samples = frames[:, 0]  # flatten to 1-D mono array
                        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                        log.debug(
                            "Transcriber: chunk %3d  rms=%6.0f  speech_started=%-5s  "
                            "consec=%d  silence=%d",
                            chunk_index, rms, speech_started, consecutive_speech, silence_chunks,
                        )

                        if rms >= self._speech_threshold:
                            if speech_first_chunk is None:
                                speech_first_chunk = chunk_index
                            consecutive_speech += 1
                            silence_chunks = 0
                            if not speech_started and consecutive_speech >= config.TRANSCRIBE_MIN_SPEECH_CHUNKS:
                                log.info(
                                    "Speech detected (rms=%.0f) — recording …%s",
                                    rms, _mark(),
                                )
                                speech_started = True
                        else:
                            consecutive_speech = 0
                            if speech_started:
                                silence_chunks += 1

                        frames_list.append(samples)

                        if speech_started and silence_chunks >= _SILENCE_CHUNKS_NEEDED:
                            break

                        # Early exit: speech hasn't been confirmed and the caller's wait window expired.
                        if not speech_started and wait_chunks is not None and chunk_index + 1 >= wait_chunks:
                            log.debug(
                                "Transcriber: no speech within %.1f s — returning None",
                                wait_for_speech_seconds,
                            )
                            _early_return = None
                            break
                    else:
                        log.debug(
                            "Transcriber: hit WHISPER_MAX_RECORD_SECONDS cap (%.1f s)",
                            config.WHISPER_MAX_RECORD_SECONDS,
                        )
                break  # stream opened and recording completed — exit retry loop
            except sd.PortAudioError as exc:
                if _attempt < 2:
                    log.warning(
                        "Transcriber: mic open failed (attempt %d/3): %s — retrying in 2 s",
                        _attempt + 1, exc,
                    )
                    time.sleep(2.0)
                else:
                    log.exception("Transcriber: mic open failed after 3 attempts")
                    raise

        if _early_return is not _SENTINEL:
            return _early_return  # type: ignore[return-value]

        # Speech was never confirmed — skip the API call entirely.
        if not speech_started:
            return ""

        # Trim dead air before sending to Whisper:
        #   - Leading: start 2 chunks before the first above-threshold chunk
        #     (preserves attack transients while dropping mic-open silence).
        #   - Trailing: drop the silent chunks that triggered end-of-speech;
        #     they add duration without adding information.
        trim_start = max(0, (speech_first_chunk or 0) - 2)
        trim_end   = max(trim_start + 1, len(frames_list) - silence_chunks)
        frames_for_whisper = frames_list[trim_start:trim_end]

        pcm = np.concatenate(frames_for_whisper) if frames_for_whisper else np.array([], dtype=np.int16)
        audio_duration = len(pcm) / config.AUDIO_SAMPLE_RATE

        log.info(
            "Silence detected — sending %.1f s audio to Whisper …%s",
            audio_duration, _mark(),
        )

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # int16 = 2 bytes per sample
            wf.setframerate(config.AUDIO_SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)

        t_api = time.monotonic()
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

        log.info(
            "Whisper transcription complete (API %.1f s)%s",
            time.monotonic() - t_api, _mark(),
        )
        log.debug("Whisper result: %r", text)
        return _filter_hallucination(text)


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------

def _filter_hallucination(text: str) -> str:
    """Return text unchanged if it looks like real speech, otherwise "".

    Drops results that are:
      - empty / whitespace only
      - shorter than WHISPER_MIN_WORDS words
      - contain a known Whisper hallucination substring
    """
    if not text:
        return ""

    lower = text.lower()

    for phrase in config.WHISPER_HALLUCINATION_FILTER:
        if phrase in lower:
            log.info("Whisper hallucination filtered: %r (matched %r)", text, phrase)
            return ""

    words = text.split()
    if len(words) < config.WHISPER_MIN_WORDS:
        log.info("Whisper result too short (%d word(s)), filtered: %r", len(words), text)
        return ""

    return text
