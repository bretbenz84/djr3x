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

Two-phase silence detection
---------------------------
Phase 1 — Waiting for speech to begin:
    After the mic opens, keep listening for up to TRANSCRIBE_SPEECH_WAIT_SECONDS
    (default 5.0 s).  The silence clock does NOT run during this phase — dead air
    before the person starts speaking never ends the recording early.  If no speech
    is detected before the window expires, return None so the caller can prompt
    "are you there?".

Phase 2 — Recording active speech:
    Once speech_started = True (RMS above threshold for TRANSCRIBE_MIN_SPEECH_CHUNKS
    consecutive chunks), switch to end-of-speech detection.  Require
    TRANSCRIBE_END_SILENCE_SECONDS (default 1.5 s) of sustained silence to stop.
    Any chunk above the RMS threshold resets the silence counter back to zero so
    brief inter-word pauses never cut off an utterance mid-sentence.
    WHISPER_MAX_RECORD_SECONDS is a hard safety cap regardless of silence.
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

# Chunk counts derived from config — computed once at import time.
_END_SILENCE_CHUNKS: int = max(
    1,
    round(config.TRANSCRIBE_END_SILENCE_SECONDS * config.AUDIO_SAMPLE_RATE / config.AUDIO_CHUNK_SIZE),
)
_MAX_CHUNKS: int = round(
    config.WHISPER_MAX_RECORD_SECONDS * config.AUDIO_SAMPLE_RATE / config.AUDIO_CHUNK_SIZE
)
_STALL_CHUNKS: int = max(
    1,
    round(1.5 * config.AUDIO_SAMPLE_RATE / config.AUDIO_CHUNK_SIZE),
)


class Transcriber:
    """Speech-to-text — either local mlx-whisper (Apple Silicon) or OpenAI Whisper API.

    Not thread-safe: only one transcribe() call may run at a time.
    """

    def __init__(self) -> None:
        self._use_local: bool = config.USE_LOCAL_TRANSCRIPTION
        self._mlx_model = None   # loaded in warmup() when _use_local is True

        if self._use_local:
            log.info("Transcriber: using local mlx-whisper (Apple Silicon)")
        else:
            self._client = OpenAI(
                api_key=config.OPENAI_API_KEY,
                timeout=config.OPENAI_TIMEOUT_SECONDS,
            )
            log.info("Transcriber: using Whisper API")

        # Speech detection threshold — may be overridden by calibrate_noise_floor().
        self._speech_threshold: int = config.TRANSCRIBE_SPEECH_THRESHOLD

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Always True — both backends are expected to be reachable."""
        return True

    def warmup(self) -> None:
        """Load mlx-whisper model on Apple Silicon; no-op for the API path.

        Falls back to the Whisper API if mlx_whisper cannot be imported so
        the program still runs on macOS without the optional dependency.
        """
        if not self._use_local:
            return
        try:
            import mlx_whisper  # type: ignore[import]
            import mlx.core as mx  # type: ignore[import]
            from mlx_whisper.transcribe import ModelHolder  # type: ignore[import]
            from huggingface_hub import snapshot_download  # type: ignore[import]

            log.info(
                "Transcriber: loading mlx-whisper model %s …",
                config.LOCAL_WHISPER_MODEL,
            )
            # Resolve the model to a local snapshot directory once.  All
            # subsequent calls pass this filesystem path so snapshot_download
            # is never contacted again.
            self._mlx_model_path: str = snapshot_download(config.LOCAL_WHISPER_MODEL)

            # Load the weights and inject them directly into ModelHolder —
            # the class-level cache that mlx_whisper.transcribe() consults.
            # Without this, the first transcribe() call finds ModelHolder.model
            # is None and re-loads from disk (and re-contacts HuggingFace if
            # given a repo name).  Setting it here means every call is a cache
            # hit from the start.
            self._mlx_model = mlx_whisper.load_models.load_model(
                self._mlx_model_path, dtype=mx.float16
            )
            ModelHolder.model = self._mlx_model
            ModelHolder.model_path = self._mlx_model_path

            log.info("Transcriber: mlx-whisper model loaded and cached in ModelHolder")
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Transcriber: mlx-whisper unavailable (%s) — falling back to Whisper API",
                exc,
            )
            self._use_local = False
            self._mlx_model = None
            self._mlx_model_path = ""
            self._client = OpenAI(
                api_key=config.OPENAI_API_KEY,
                timeout=config.OPENAI_TIMEOUT_SECONDS,
            )

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
            raw = _record_with_fallback(n_frames)
        except sd.PortAudioError:
            log.warning(
                "Noise floor calibration failed — keeping threshold at %d",
                self._speech_threshold,
            )
            return

        mono = raw.astype(np.float32).mean(axis=1)
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
        allow_short: bool = False,
    ) -> str | None:
        """Capture one utterance from the microphone and return its text.

        Opens the mic, runs two-phase silence detection, then encodes the
        buffer as a WAV and sends it to the Whisper API.  Trims leading and
        trailing silence before sending so dead air doesn't inflate latency.

        Phase 1 — waiting for speech:
            Listens for up to wait_for_speech_seconds (default:
            TRANSCRIBE_SPEECH_WAIT_SECONDS) without stopping on silence.
            Returns None if no speech begins within that window so the caller
            can trigger an "are you there?" prompt.

        Phase 2 — recording active speech:
            Once speech_started = True, requires TRANSCRIBE_END_SILENCE_SECONDS
            of sustained silence to stop.  Any chunk above the RMS threshold
            resets the counter; brief pauses never cut off an utterance.

        Returns the transcription string, "" if speech was detected but Whisper
        returned nothing (or the API fails), or None if Phase 1 timed out.

        t0: optional monotonic timestamp from wake word detection; when
        provided, all INFO log lines include an elapsed-since-wake marker.

        Raises sounddevice.PortAudioError if the mic cannot be opened.
        """
        _wait_secs: float = (
            wait_for_speech_seconds
            if wait_for_speech_seconds is not None
            else config.TRANSCRIBE_SPEECH_WAIT_SECONDS
        )
        _wait_chunks: int = round(
            _wait_secs * config.AUDIO_SAMPLE_RATE / config.AUDIO_CHUNK_SIZE
        )

        def _mark() -> str:
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
        # Use a meaningfully stricter threshold to START speech than to
        # CONTINUE speech. The previous logic could collapse both thresholds
        # to the calibration floor (e.g. 300), which made steady room noise
        # look like speech after only a few chunks.
        speech_detect_threshold = max(
            config.TRANSCRIBE_SPEECH_THRESHOLD_MIN,
            int(self._speech_threshold * 0.75),
        )
        # Require a modest bump above the calibrated floor, but cap the boost
        # so a noisy calibration (e.g. startup chatter / room noise) does not
        # make real speech effectively unreachable.
        speech_start_threshold = min(
            self._speech_threshold + 40,
            max(
                speech_detect_threshold + 30,
                int(self._speech_threshold * 1.02),
            ),
        )
        speech_confirm_peak_threshold = min(
            speech_start_threshold + 60,
            max(
                speech_start_threshold + 25,
                int(self._speech_threshold * 1.08),
            ),
        )
        stall_threshold = max(10.0, speech_detect_threshold * 0.05)
        # Once in Phase 2 (speech confirmed), only reset the silence counter when
        # RMS is clearly above the detection floor.  Background noise and mechanical/
        # speaker bleedthrough typically sits at 300–600 RMS; real speech is 1000–5000.
        # Using a 2.5× multiplier leaves a clear gap so post-speech noise does not
        # repeatedly reset the counter and extend the recording.
        silence_reset_threshold = int(speech_detect_threshold * config.TRANSCRIBE_SILENCE_RESET_MULTIPLIER)
        log.info(
            "Transcriber thresholds: calibrated=%d, speech-start=%d, speech-detect=%d, confirm-peak=%d, silence-reset=%d, stall<=%.0f%s",
            self._speech_threshold,
            speech_start_threshold,
            speech_detect_threshold,
            speech_confirm_peak_threshold,
            silence_reset_threshold,
            stall_threshold,
            _mark(),
        )
        log.debug(
            "Transcriber: speech-detect threshold=%d (base=%d), speech-confirm-peak=%d, silence-reset=%d, stall threshold=%.0f",
            speech_detect_threshold,
            self._speech_threshold,
            speech_confirm_peak_threshold,
            silence_reset_threshold,
            stall_threshold,
        )

        _SENTINEL = object()
        _early_return: object = _SENTINEL  # set to None if Phase 1 times out
        frames_list: list[np.ndarray] = []
        speech_started: bool = False
        speech_first_chunk: int | None = None

        for _attempt in range(3):
            frames_list = []
            speech_started = False
            speech_first_chunk = None
            consecutive_speech: int = 0
            speech_run_peak: float = 0.0
            silence_chunks: int = 0
            near_zero_chunks: int = 0
            voiced_chunks: int = 0

            try:
                with _open_input_stream() as stream:
                    log.info(
                        "Mic open, waiting for speech (%.1fs window) …%s",
                        _wait_secs, _mark(),
                    )
                    for chunk_index in range(_MAX_CHUNKS):
                        frames, overflowed = stream.read(config.AUDIO_CHUNK_SIZE)
                        if overflowed:
                            log.debug("Transcriber: audio buffer overflowed (input too slow)")

                        # Mix down to 1-D mono int16
                        samples = frames.astype(np.float32).mean(axis=1).astype(np.int16)
                        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                        log.debug(
                            "Transcriber: chunk %3d  rms=%6.0f  phase=%s  "
                            "consec=%d  silence=%d",
                            chunk_index, rms,
                            "2-recording" if speech_started else "1-waiting",
                            consecutive_speech, silence_chunks,
                        )

                        if rms <= stall_threshold:
                            near_zero_chunks += 1
                        else:
                            near_zero_chunks = 0

                        # A long run of near-zero chunks usually means the mic
                        # stream is stalled or returning effectively empty audio.
                        # Reopen once via the normal retry loop instead of
                        # waiting out the full listen window on dead input.
                        if near_zero_chunks >= _STALL_CHUNKS:
                            raise sd.PortAudioError(
                                f"input stream stalled: {near_zero_chunks} near-zero chunks"
                            )

                        active_threshold = (
                            speech_detect_threshold if speech_started else speech_start_threshold
                        )

                        if rms >= active_threshold:
                            # --- above-threshold chunk ---
                            if speech_first_chunk is None:
                                speech_first_chunk = chunk_index
                            consecutive_speech += 1
                            speech_run_peak = max(speech_run_peak, rms)
                            voiced_chunks += 1
                            # In Phase 1, always bust silence (silence counter is
                            # unused in Phase 1, but keep it clean).
                            # In Phase 2, only reset the silence counter when the
                            # chunk is clearly voiced — well above the detection
                            # floor.  Background noise and speaker/servo bleedthrough
                            # typically sits at 300–600 RMS; real speech is 1000+.
                            # Chunks between active_threshold and silence_reset_threshold
                            # are recorded but do NOT reset the silence clock, so a
                            # short utterance ("you suck") doesn't extend the recording
                            # by 5+ seconds of ambient noise.
                            if not speech_started or rms >= silence_reset_threshold:
                                silence_chunks = 0

                            if (
                                not speech_started
                                and consecutive_speech >= config.TRANSCRIBE_MIN_SPEECH_CHUNKS
                                and speech_run_peak >= speech_confirm_peak_threshold
                            ):
                                # ---- Phase 1 → Phase 2 transition ----
                                speech_started = True
                                log.info("Speech detected, recording …%s", _mark())
                        else:
                            # --- below-threshold chunk ---
                            consecutive_speech = 0
                            speech_run_peak = 0.0
                            if speech_started:
                                silence_chunks += 1   # only count silence in Phase 2

                        frames_list.append(samples)

                        # Phase 2: sustained silence → end of utterance
                        if speech_started and silence_chunks >= _END_SILENCE_CHUNKS:
                            silence_secs = (
                                silence_chunks * config.AUDIO_CHUNK_SIZE / config.AUDIO_SAMPLE_RATE
                            )
                            log.info(
                                "End of speech detected (%.1fs silence)%s",
                                silence_secs, _mark(),
                            )
                            break

                        # Phase 1: no speech within the wait window → return None
                        if not speech_started and chunk_index + 1 >= _wait_chunks:
                            log.info(
                                "No speech detected within %.1fs — returning None%s",
                                _wait_secs, _mark(),
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

        # On the normal command path, avoid sending ultra-short, weak clips to
        # Whisper. These are the main source of near-silence hallucinations
        # like "thank you" when the user hesitates too long before speaking.
        if not allow_short:
            if (
                audio_duration < config.TRANSCRIBE_MIN_WHISPER_SECONDS
                or voiced_chunks < config.TRANSCRIBE_MIN_VOICED_CHUNKS
            ):
                log.info(
                    "Transcriber: skipping Whisper for short/weak clip "
                    "(duration=%.1fs, voiced_chunks=%d, min_duration=%.1fs, min_voiced=%d)%s",
                    audio_duration,
                    voiced_chunks,
                    config.TRANSCRIBE_MIN_WHISPER_SECONDS,
                    config.TRANSCRIBE_MIN_VOICED_CHUNKS,
                    _mark(),
                )
                return ""

        log.info(
            "Silence detected — transcribing %.1f s audio (%s) …%s",
            audio_duration,
            "mlx-whisper" if self._use_local else "Whisper API",
            _mark(),
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
            if self._use_local:
                import mlx_whisper  # type: ignore[import]
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(buf.read())
                    tmp_path = tmp.name
                try:
                    result = mlx_whisper.transcribe(
                        tmp_path,
                        # Pass the local snapshot directory, not the HuggingFace
                        # repo name — this skips the remote revision check so
                        # every call uses the already-loaded in-memory weights.
                        path_or_hf_repo=self._mlx_model_path,
                        language=config.WHISPER_LANGUAGE or None,
                    )
                    text = (result.get("text") or "").strip()
                finally:
                    os.unlink(tmp_path)
                log.info(
                    "mlx-whisper transcription complete (%.1f s)%s",
                    time.monotonic() - t_api, _mark(),
                )
            else:
                result = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("audio.wav", buf.read()),
                    language=config.WHISPER_LANGUAGE,
                )
                text = result.text.strip()
                log.info(
                    "Whisper API transcription complete (%.1f s)%s",
                    time.monotonic() - t_api, _mark(),
                )
        except Exception:
            log.exception("Transcription error")
            return ""

        log.debug("Transcription result: %r", text)
        return _filter_hallucination(text, allow_short=allow_short)


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------

_MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def _is_date_hallucination(text: str) -> bool:
    """Return True if *text* is a bare date pattern Whisper commonly hallucinates.

    Matches:
      - A lone month name:           "January"
      - A lone 4-digit year:         "2020"
      - Month + year:                "January 2020"
      - Year + month:                "2020 January"
    """
    import re
    stripped = text.strip().rstrip(".")
    parts = stripped.split()
    if len(parts) == 1:
        word = parts[0].lower()
        return word in _MONTHS or bool(re.fullmatch(r"(19|20)\d{2}", word))
    if len(parts) == 2:
        a, b = parts[0].lower(), parts[1].lower()
        year_re = re.compile(r"(19|20)\d{2}")
        a_is_month, b_is_month = a in _MONTHS, b in _MONTHS
        a_is_year = bool(year_re.fullmatch(a))
        b_is_year = bool(year_re.fullmatch(b))
        return (a_is_month and b_is_year) or (a_is_year and b_is_month)
    return False


def _is_repetitive_hallucination(text: str) -> bool:
    """Return True if text is a looping repetition of a short phrase.

    Whisper sometimes outputs the same 2-6 word chunk dozens of times when
    it receives near-silence or low-energy audio (e.g. "a little bit of a
    little bit of ...").  We detect this by sliding a window of N words
    across the token list and checking whether a candidate phrase repeats
    enough times to dominate the output.
    """
    words = text.lower().split()
    total = len(words)
    if total < 6:
        return False
    # Try phrase lengths from 2 to 6 words
    for n in range(2, 7):
        if n >= total:
            break
        phrase = tuple(words[:n])
        # Count non-overlapping occurrences of the phrase
        count = 0
        i = 0
        while i <= total - n:
            if tuple(words[i:i + n]) == phrase:
                count += 1
                i += n
            else:
                i += 1
        # If the phrase accounts for ≥60% of all words, it's a loop
        if count >= 3 and (count * n) / total >= 0.60:
            return True
    return False


def _filter_hallucination(text: str, allow_short: bool = False) -> str:
    """Return text unchanged if it looks like real speech, otherwise "".

    Drops results that are:
      - empty / whitespace only
      - contain a known Whisper hallucination substring
      - a bare month name, year, or 'month year' / 'year month' pattern
      - a looping repetition of a short phrase

    allow_short is retained for API compatibility with existing call sites,
    but short results are no longer filtered purely for length.
    """
    if not text:
        return ""

    lower = text.lower()
    exact = lower.strip().strip(" \t\r\n.!?,:;\"'")

    if exact in config.WHISPER_HALLUCINATION_EXACT:
        log.info("Whisper hallucination filtered: %r (exact short match %r)", text, exact)
        return ""

    for phrase in config.WHISPER_HALLUCINATION_FILTER:
        if phrase in lower:
            log.info("Whisper hallucination filtered: %r (matched %r)", text, phrase)
            return ""

    if _is_date_hallucination(text):
        log.info("Whisper hallucination filtered (date pattern): %r", text)
        return ""

    if _is_repetitive_hallucination(text):
        log.info("Whisper hallucination filtered (repetitive phrase loop): %r", text[:120])
        return ""

    return text


def _input_devices_to_try() -> list[int | None]:
    device = config.AUDIO_INPUT_DEVICE
    return [device, None] if device is not None else [None]


def _record_with_fallback(n_frames: int) -> np.ndarray:
    last_exc: sd.PortAudioError | None = None
    for device in _input_devices_to_try():
        try:
            if device is None and config.AUDIO_INPUT_DEVICE is not None:
                log.warning("Transcriber: falling back to default input device for calibration")
            return sd.rec(
                n_frames,
                samplerate=config.AUDIO_SAMPLE_RATE,
                channels=config.MIC_CHANNELS,
                dtype="int16",
                device=device,
                blocking=True,
            )
        except sd.PortAudioError as exc:
            last_exc = exc
            log.warning("Transcriber: calibration input device %s failed — %s", device, exc)
    assert last_exc is not None
    raise last_exc


def _open_input_stream():
    last_exc: sd.PortAudioError | None = None
    for device in _input_devices_to_try():
        try:
            if device is None and config.AUDIO_INPUT_DEVICE is not None:
                log.warning("Transcriber: falling back to default input device")
            return sd.InputStream(
                samplerate=config.AUDIO_SAMPLE_RATE,
                channels=config.MIC_CHANNELS,
                dtype="int16",
                device=device,
                blocksize=config.AUDIO_CHUNK_SIZE,
            )
        except sd.PortAudioError as exc:
            last_exc = exc
            log.warning("Transcriber: input device %s failed — %s", device, exc)
    assert last_exc is not None
    raise last_exc
