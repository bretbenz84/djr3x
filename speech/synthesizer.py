"""
speech/synthesizer.py — ElevenLabs TTS streaming for DJ-R3X.

Two entry points:

  speak(text)
    Complete text string → ElevenLabs stream() → player speech queue.
    Checks the on-disk PCM cache first; populates it on a miss.
    Blocks until playback finishes.

  speak_stream(text_iter)
    Iterator of text tokens (from ChatGPT streaming) →
    ElevenLabs convert_realtime() WebSocket → player speech queue.
    Audio starts playing before the full response is generated.
    Not cached (full text isn't known in advance).
    Blocks until playback finishes.

Cache format: raw PCM int16 mono 16000 Hz wrapped in a .wav container,
stored in config.AUDIO_CACHE_DIR, keyed by SHA-256 of the text string.
Cache hits skip the API call entirely and play through player.play_file().
"""

from __future__ import annotations

import hashlib
import logging
import time
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

import config
from audio.player import AudioPlayer, SPEECH_SAMPLE_RATE

log = logging.getLogger(__name__)


class Synthesizer:
    """ElevenLabs TTS client for DJ-R3X."""

    def __init__(self, player: AudioPlayer) -> None:
        self._player = player
        self._client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
        self._voice_settings = VoiceSettings(
            stability=config.ELEVENLABS_STABILITY,
            similarity_boost=config.ELEVENLABS_SIMILARITY,
        )
        config.AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Synthesize a complete text string and play it through the player.

        Checks the cache first. On a miss, streams PCM from ElevenLabs and
        saves the result as a .wav for future use. Blocks until done.
        """
        text = text.strip()
        if not text:
            return

        cached = _cache_path(text)
        if cached.exists():
            log.debug("TTS cache hit: %s", cached.name)
            self._player.play_file(cached)
            return

        log.debug("TTS streaming from ElevenLabs: %.60s…", text)
        try:
            chunks = self._client.text_to_speech.stream(
                voice_id=config.ELEVENLABS_VOICE_ID,
                text=text,
                model_id=config.ELEVENLABS_MODEL_ID,
                output_format="pcm_16000",
                voice_settings=self._voice_settings,
                optimize_streaming_latency=4,   # maximum latency reduction
            )
            self._pipe_to_player(chunks, cache_path=cached)
        except Exception:
            log.exception("ElevenLabs TTS error for text: %.60s…", text)
            self._player.stop_speech()
            raise

    def speak_stream(self, text_iter: Iterator[str], t0: float | None = None) -> None:
        """Accept a streaming token iterator (e.g. from ChatGPT) and pipe it
        through ElevenLabs convert_realtime() to the player.

        Uses a WebSocket connection so audio begins playing before the full
        response has been generated. Not cached. Blocks until done.
        """
        try:
            chunks = self._client.text_to_speech.convert_realtime(
                voice_id=config.ELEVENLABS_VOICE_ID,
                text=text_iter,
                model_id=config.ELEVENLABS_MODEL_ID,
                output_format="pcm_16000",
                voice_settings=self._voice_settings,
            )
            self._pipe_to_player(chunks, cache_path=None, t0=t0)
        except Exception:
            log.exception("ElevenLabs realtime TTS error")
            self._player.stop_speech()
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pipe_to_player(
        self,
        audio_chunks: Iterator[bytes],
        cache_path: Path | None,
        t0: float | None = None,
    ) -> None:
        """Stream audio chunks to the player as they arrive.

        If cache_path is given, accumulates the raw PCM bytes and writes a
        .wav cache file after all chunks have been received successfully.

        Always calls player.end_speech() + wait_for_speech() on exit,
        even if an exception occurs mid-stream, so the player is left
        in a clean state.
        """
        accumulator: list[bytes] | None = [] if cache_path is not None else None
        first_chunk_logged = False

        try:
            for chunk in audio_chunks:
                if not chunk:
                    continue
                if not first_chunk_logged:
                    elapsed = f" [+{time.monotonic() - t0:.1f}s]" if t0 is not None else ""
                    log.info("First audio chunk playing%s", elapsed)
                    first_chunk_logged = True
                chunk = _apply_gain(chunk)
                self._player.feed_speech_chunk(chunk)
                if accumulator is not None:
                    accumulator.append(chunk)
        finally:
            # always signal end-of-speech so the player doesn't stay active
            self._player.end_speech()
            finished = self._player.wait_for_speech(timeout=30.0)
            if not finished:
                log.warning("wait_for_speech() timed out — forcing stop")
                self._player.stop_speech()

        # write cache only on clean completion (not inside finally)
        if cache_path is not None and accumulator:
            _write_wav_cache(cache_path, b"".join(accumulator))


# ---------------------------------------------------------------------------
# Module-level helpers (no instance state needed)
# ---------------------------------------------------------------------------

def _apply_gain(chunk: bytes) -> bytes:
    """Multiply PCM int16 samples by SYNTHESIZER_VOLUME_GAIN and clip.

    Returns the original bytes unchanged when gain is 1.0 to avoid the
    numpy round-trip overhead on every chunk.
    """
    if config.SYNTHESIZER_VOLUME_GAIN == 1.0:
        return chunk
    samples = np.frombuffer(chunk, dtype=np.int16).copy()
    boosted = np.clip(
        samples.astype(np.float32) * config.SYNTHESIZER_VOLUME_GAIN,
        -32768.0, 32767.0,
    ).astype(np.int16)
    return boosted.tobytes()


def _cache_path(text: str) -> Path:
    """Return the .wav cache path for this exact text string.
    Uses the first 16 hex chars of SHA-256 — collision probability is
    negligible for the handful of canned responses this project will have.
    """
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    return config.AUDIO_CACHE_DIR / f"{digest}.wav"


def _write_wav_cache(path: Path, pcm_bytes: bytes) -> None:
    """Write raw PCM int16 mono bytes to a .wav file at SPEECH_SAMPLE_RATE."""
    try:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)              # int16 = 2 bytes per sample
            wf.setframerate(SPEECH_SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        log.debug("TTS cached → %s (%d bytes)", path.name, len(pcm_bytes))
    except OSError:
        log.warning("Failed to write TTS cache %s", path, exc_info=True)
