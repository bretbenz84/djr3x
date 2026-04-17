"""
speech/synthesizer.py — selectable TTS backends for DJ-R3X.

`Synthesizer` is the public facade used by the state machine. The concrete
backend is chosen from `config.TTS_PROVIDER` at startup:

  - `elevenlabs`: existing streaming/cached cloud TTS
  - `piper`: local ONNX model loaded once and held open for the process
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
import wave
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from audio.player import AudioPlayer, SPEECH_SAMPLE_RATE
import config

log = logging.getLogger(__name__)

try:
    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs
except ImportError:  # pragma: no cover - optional backend dependency
    ElevenLabs = None
    VoiceSettings = None

try:
    from piper.voice import PiperVoice
except ImportError:  # pragma: no cover - optional backend dependency
    PiperVoice = None


class _Backend(Protocol):
    provider_name: str

    def speak(self, text: str) -> None: ...
    def speak_stream(self, text_iter: Iterator[str], t0: float | None = None) -> None: ...


class Synthesizer:
    """Facade that dispatches to the configured TTS backend."""

    def __init__(self, player: AudioPlayer) -> None:
        self._player = player
        if config.TTS_PROVIDER == "piper":
            self._backend: _Backend = _PiperSynthesizer(player)
        else:
            self._backend = _ElevenLabsSynthesizer(player)

    @property
    def provider_name(self) -> str:
        return self._backend.provider_name

    def speak(self, text: str) -> None:
        self._backend.speak(text)

    def speak_stream(self, text_iter: Iterator[str], t0: float | None = None) -> None:
        self._backend.speak_stream(text_iter, t0=t0)


class _ElevenLabsSynthesizer:
    provider_name = "ElevenLabs"

    def __init__(self, player: AudioPlayer) -> None:
        if ElevenLabs is None or VoiceSettings is None:
            raise RuntimeError(
                "TTS_PROVIDER=elevenlabs requires the 'elevenlabs' package"
            )
        self._player = player
        self._client = ElevenLabs(
            api_key=config.ELEVENLABS_API_KEY,
            timeout=config.ELEVENLABS_TIMEOUT_SECONDS,
        )
        self._voice_settings = VoiceSettings(
            stability=config.ELEVENLABS_STABILITY,
            similarity_boost=config.ELEVENLABS_SIMILARITY,
        )
        config.AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        log.info("Speaking via ElevenLabs: %s", text)
        cached = _resolve_cache_path(text)
        if cached is not None:
            log.debug("TTS cache hit: %s", cached.name)
            self._player.play_file(cached)
            return

        cached = _cache_path(text)
        log.debug("TTS streaming from ElevenLabs: %.60s…", text)
        try:
            chunks = self._client.text_to_speech.stream(
                voice_id=config.ELEVENLABS_VOICE_ID,
                text=text,
                model_id=config.ELEVENLABS_MODEL_ID,
                output_format="pcm_16000",
                voice_settings=self._voice_settings,
                optimize_streaming_latency=4,
            )
            _pipe_to_player(
                self._player,
                chunks,
                cache_path=cached,
                sample_rate=SPEECH_SAMPLE_RATE,
            )
        except Exception:
            log.exception("ElevenLabs TTS error for text: %.60s…", text)
            self._player.stop_speech()
            raise

    def speak_stream(self, text_iter: Iterator[str], t0: float | None = None) -> None:
        try:
            chunks = self._client.text_to_speech.convert_realtime(
                voice_id=config.ELEVENLABS_VOICE_ID,
                text=text_iter,
                model_id=config.ELEVENLABS_MODEL_ID,
                output_format="pcm_16000",
                voice_settings=self._voice_settings,
            )
            _pipe_to_player(
                self._player,
                chunks,
                cache_path=None,
                t0=t0,
                sample_rate=SPEECH_SAMPLE_RATE,
            )
        except Exception:
            log.exception("ElevenLabs realtime TTS error")
            self._player.stop_speech()
            raise


class _PiperSynthesizer:
    provider_name = "Piper"

    def __init__(self, player: AudioPlayer) -> None:
        if PiperVoice is None:
            raise RuntimeError(
                "TTS_PROVIDER=piper requires the 'piper-tts' package"
            )
        if not config.PIPER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Piper model not found: {config.PIPER_MODEL_PATH}"
            )
        if not config.PIPER_CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Piper config not found: {config.PIPER_CONFIG_PATH}"
            )

        self._player = player
        config.AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Load and hold the ONNX session open once at startup.
        self._voice = PiperVoice.load(
            model_path=config.PIPER_MODEL_PATH,
            config_path=config.PIPER_CONFIG_PATH,
            use_cuda=config.PIPER_USE_CUDA,
        )
        self._sample_rate = int(self._voice.config.sample_rate)
        log.info(
            "Loaded Piper voice from %s (%d Hz)",
            config.PIPER_MODEL_PATH,
            self._sample_rate,
        )

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        log.info("Speaking via Piper: %s", text)
        cached = _resolve_cache_path(text)
        if cached is not None:
            log.debug("TTS cache hit: %s", cached.name)
            self._player.play_file(cached)
            return

        chunks = self._voice.synthesize_stream_raw(
            text,
            speaker_id=config.PIPER_SPEAKER_ID,
            length_scale=config.PIPER_LENGTH_SCALE,
            noise_scale=config.PIPER_NOISE_SCALE,
            noise_w=config.PIPER_NOISE_W,
            sentence_silence=config.PIPER_SENTENCE_SILENCE,
        )
        _pipe_to_player(
            self._player,
            chunks,
            cache_path=_cache_path(text),
            sample_rate=self._sample_rate,
        )

    def speak_stream(self, text_iter: Iterator[str], t0: float | None = None) -> None:
        # Piper is local, but this repo's current interface expects one model
        # session to stay warm and accept complete utterances. We collect the
        # streamed tokens, then synthesize once with the already-loaded voice.
        full_text = "".join(text_iter).strip()
        if not full_text:
            return
        if t0 is not None:
            log.info("Piper stream fallback: synthesizing buffered reply [+%.1fs]", time.monotonic() - t0)
        self.speak(full_text)


def _pipe_to_player(
    player: AudioPlayer,
    audio_chunks: Iterator[bytes],
    *,
    cache_path: Path | None,
    sample_rate: int,
    t0: float | None = None,
) -> None:
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
            player.feed_speech_chunk(chunk, sample_rate=sample_rate)
            if accumulator is not None:
                accumulator.append(chunk)
    finally:
        player.end_speech()
        finished = player.wait_for_speech(timeout=30.0)
        if not finished:
            log.warning("wait_for_speech() timed out — forcing stop")
            player.stop_speech()

    if cache_path is not None and accumulator:
        _write_wav_cache(cache_path, b"".join(accumulator), sample_rate=sample_rate)


def _apply_gain(chunk: bytes) -> bytes:
    """Multiply PCM int16 samples by SYNTHESIZER_VOLUME_GAIN and clip."""
    if config.SYNTHESIZER_VOLUME_GAIN == 1.0:
        return chunk
    samples = np.frombuffer(chunk, dtype=np.int16).copy()
    boosted = np.clip(
        samples.astype(np.float32) * config.SYNTHESIZER_VOLUME_GAIN,
        -32768.0,
        32767.0,
    ).astype(np.int16)
    return boosted.tobytes()


def _cache_path(text: str) -> Path:
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    provider = config.TTS_PROVIDER
    return config.AUDIO_CACHE_DIR / f"{provider}-{digest}.wav"


def _legacy_cache_path(text: str) -> Path:
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    return config.LEGACY_AUDIO_CACHE_DIR / f"{digest}.wav"


def _resolve_cache_path(text: str) -> Path | None:
    current = _cache_path(text)
    if current.exists():
        return current

    legacy = _legacy_cache_path(text)
    if config.TTS_PROVIDER != "elevenlabs" or not legacy.exists() or legacy == current:
        return None

    try:
        current.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(current))
        log.debug("Migrated TTS cache %s -> %s", legacy.name, current.parent)
        return current
    except OSError:
        log.warning("Failed to migrate TTS cache %s -> %s", legacy, current, exc_info=True)
        return legacy


def _write_wav_cache(path: Path, pcm_bytes: bytes, *, sample_rate: int) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        log.debug("TTS cached → %s (%d bytes @ %d Hz)", path.name, len(pcm_bytes), sample_rate)
    except OSError:
        log.warning("Failed to write TTS cache %s", path, exc_info=True)
