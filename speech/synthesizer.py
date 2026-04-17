"""
speech/synthesizer.py — selectable TTS backends for DJ-R3X.

`Synthesizer` is the public facade used by the state machine. The concrete
backend is chosen from `config.TTS_PROVIDER` at startup:

  - `elevenlabs`: existing streaming/cached cloud TTS
  - `piper`: local ONNX model loaded once and held open for the process
  - `xtts`: local Coqui XTTS checkpoint loaded once on Apple Silicon
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

try:
    from piper.config import SynthesisConfig
except ImportError:  # pragma: no cover - optional backend dependency
    SynthesisConfig = None

try:
    import torch
except ImportError:  # pragma: no cover - optional backend dependency
    torch = None

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
except ImportError:  # pragma: no cover - optional backend dependency
    XttsConfig = None
    Xtts = None


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
        elif config.TTS_PROVIDER == "xtts":
            self._backend = _XttsSynthesizer(player)
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
        if PiperVoice is None or SynthesisConfig is None:
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
        self._syn_config = SynthesisConfig(
            speaker_id=config.PIPER_SPEAKER_ID,
            length_scale=config.PIPER_LENGTH_SCALE,
            noise_scale=config.PIPER_NOISE_SCALE,
            noise_w_scale=config.PIPER_NOISE_W,
            normalize_audio=True,
            volume=1.0,
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

        chunks = self._synthesize_pcm_chunks(text)
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

    def _synthesize_pcm_chunks(self, text: str) -> Iterator[bytes]:
        silence_bytes = b""
        if config.PIPER_SENTENCE_SILENCE > 0.0:
            silence_samples = int(config.PIPER_SENTENCE_SILENCE * self._sample_rate)
            silence_bytes = bytes(silence_samples * 2)

        for audio_chunk in self._voice.synthesize(text, syn_config=self._syn_config):
            yield audio_chunk.audio_int16_bytes
            if silence_bytes:
                yield silence_bytes


class _XttsSynthesizer:
    provider_name = "XTTS"

    def __init__(self, player: AudioPlayer) -> None:
        if config.PLATFORM != "macos_silicon":
            raise RuntimeError("TTS_PROVIDER=xtts is supported only on macOS Apple Silicon")
        if torch is None or XttsConfig is None or Xtts is None:
            raise RuntimeError(
                "TTS_PROVIDER=xtts requires the 'TTS' package and torch on Apple Silicon"
            )
        for label, path in (
            ("XTTS config", config.XTTS_CONFIG_PATH),
            ("XTTS checkpoint", config.XTTS_CHECKPOINT_PATH),
            ("XTTS vocab", config.XTTS_VOCAB_PATH),
            ("XTTS speaker reference", config.XTTS_SPEAKER_WAV),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        self._player = player
        config.AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._device = self._select_device()

        self._config = XttsConfig()
        self._config.load_json(str(config.XTTS_CONFIG_PATH))
        self._model = Xtts.init_from_config(self._config)
        self._model.load_checkpoint(
            self._config,
            checkpoint_path=str(config.XTTS_CHECKPOINT_PATH),
            vocab_path=str(config.XTTS_VOCAB_PATH),
            use_deepspeed=False,
        )
        self._model.to(self._device)
        self._model.eval()
        self._sample_rate = int(
            getattr(self._config.audio, "output_sample_rate", 24000)
        )
        log.info(
            "Loaded XTTS model from %s on %s (%d Hz)",
            config.XTTS_CHECKPOINT_PATH,
            self._device,
            self._sample_rate,
        )
        self._gpt_cond_latent, self._speaker_embedding = (
            self._model.get_conditioning_latents(
                audio_path=[str(config.XTTS_SPEAKER_WAV)]
            )
        )

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        log.info("Speaking via XTTS: %s", text)
        cached = _resolve_cache_path(text)
        if cached is not None:
            log.debug("TTS cache hit: %s", cached.name)
            self._player.play_file(cached)
            return

        out = self._model.inference(
            text,
            config.XTTS_LANGUAGE,
            self._gpt_cond_latent,
            self._speaker_embedding,
            temperature=config.XTTS_TEMPERATURE,
            length_penalty=config.XTTS_LENGTH_PENALTY,
            repetition_penalty=config.XTTS_REPETITION_PENALTY,
            top_k=config.XTTS_TOP_K,
            top_p=config.XTTS_TOP_P,
            speed=config.XTTS_SPEED,
            enable_text_splitting=config.XTTS_ENABLE_TEXT_SPLITTING,
        )
        pcm_bytes = _float_audio_to_pcm16_bytes(out["wav"])
        _pipe_to_player(
            self._player,
            iter((pcm_bytes,)),
            cache_path=_cache_path(text),
            sample_rate=self._sample_rate,
        )

    def speak_stream(self, text_iter: Iterator[str], t0: float | None = None) -> None:
        text = "".join(text_iter).strip()
        if not text:
            return

        log.info("Speaking via XTTS stream: %.60s…", text)
        try:
            chunks = self._model.inference_stream(
                text,
                config.XTTS_LANGUAGE,
                self._gpt_cond_latent,
                self._speaker_embedding,
                temperature=config.XTTS_TEMPERATURE,
                length_penalty=config.XTTS_LENGTH_PENALTY,
                repetition_penalty=config.XTTS_REPETITION_PENALTY,
                top_k=config.XTTS_TOP_K,
                top_p=config.XTTS_TOP_P,
                speed=config.XTTS_SPEED,
                enable_text_splitting=config.XTTS_ENABLE_TEXT_SPLITTING,
            )
            _pipe_to_player(
                self._player,
                (_float_audio_to_pcm16_bytes(chunk) for chunk in chunks),
                cache_path=None,
                sample_rate=self._sample_rate,
                t0=t0,
            )
        except Exception:
            log.exception("XTTS realtime TTS error")
            self._player.stop_speech()
            raise

    @staticmethod
    def _select_device() -> str:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


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


def _float_audio_to_pcm16_bytes(audio: object) -> bytes:
    if torch is not None and isinstance(audio, torch.Tensor):
        samples = audio.detach().float().cpu().numpy().reshape(-1)
    else:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16).tobytes()


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
