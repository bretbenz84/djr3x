"""
audio/player.py — Audio playback for DJ-R3X.

Two independent output streams:
  - Speech stream: on-demand callback-based OutputStream opened by a background
    worker thread when audio data arrives. Closes (releases the device) as soon
    as the end-of-stream marker is processed. Computes real-time RMS on every
    audio callback so leds.py can drive mouth brightness.
  - Music stream: explicit OutputStream in a background thread. No RMS
    tracking. Supports loop and clean stop.

The speech stream is never held open between utterances, so PipeWire / ALSA
can grant the device to music and chime streams without conflict.

Usage:
    player = AudioPlayer()

    # Streaming TTS (synthesizer calls these):
    player.feed_speech_chunk(pcm_bytes)   # call repeatedly as chunks arrive
    player.end_speech()                   # signal end of stream

    # Cached response files:
    player.play_file("assets/audio/hello.wav")  # blocks until done

    # Background music:
    player.play_music("assets/audio/cantina.mp3", loop=True)
    player.stop_music()

    # Mouth LED brightness (read by leds.py at any time):
    brightness = player.rms   # float 0.0–255.0

    player.close()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import io
import subprocess

log = logging.getLogger(__name__)

import numpy as np
import sounddevice as sd
import soundfile as sf

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ElevenLabs is asked to produce PCM at this rate (see synthesizer.py).
# ReSpeaker Lite USB device only supports 16000 Hz.
SPEECH_SAMPLE_RATE: int = config.SPEECH_SAMPLE_RATE

# Chunk size (frames) fed from a cached file into the speech queue at once.
# Larger = fewer queue operations; smaller = more responsive stop().
_FILE_CHUNK_FRAMES: int = config.AUDIO_CHUNK_SIZE * 8


# ---------------------------------------------------------------------------
# Internal sentinel type
# ---------------------------------------------------------------------------

@dataclass
class _EndMarker:
    """Pushed onto the speech queue to signal end of a playback segment.
    `done` is set by the audio callback once the marker is reached, allowing
    play_file() to block until the file has actually finished playing."""
    done: threading.Event | None = field(default=None)


@dataclass
class _SpeechChunk:
    """Mono float32 speech samples plus whether the droid effect should run."""
    samples: np.ndarray
    apply_droid_effect: bool = False


class _DroidVoiceEffect:
    """Small stateful speech processor tuned for a droid/radio character."""

    _HIGHPASS_HZ = 180.0
    _LOWPASS_HZ = 4500.0
    _COMPRESS_THRESHOLD = 0.16
    _COMPRESS_RATIO = 4.0
    _ATTACK_MS = 4.0
    _RELEASE_MS = 90.0
    _MAKEUP_GAIN = 1.15
    _SATURATION_DRIVE = 1.35
    _BITCRUSH_BITS = 11
    _BITCRUSH_MIX = 0.18

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._hp_alpha = self._highpass_alpha(self._HIGHPASS_HZ, sample_rate)
        self._lp_alpha = self._lowpass_alpha(self._LOWPASS_HZ, sample_rate)
        self._attack_coeff = self._time_coeff(self._ATTACK_MS, sample_rate)
        self._release_coeff = self._time_coeff(self._RELEASE_MS, sample_rate)
        self._sat_norm = 1.0 / np.tanh(self._SATURATION_DRIVE)
        self._bitcrush_levels = float((1 << (self._BITCRUSH_BITS - 1)) - 1)
        self._two_pi_over_sr = (2.0 * np.pi) / sample_rate
        self.reset()

    @staticmethod
    def _time_coeff(duration_ms: float, sample_rate: int) -> float:
        samples = max(1.0, duration_ms * 0.001 * sample_rate)
        return float(np.exp(-1.0 / samples))

    @staticmethod
    def _highpass_alpha(cutoff_hz: float, sample_rate: int) -> float:
        dt = 1.0 / sample_rate
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        return float(rc / (rc + dt))

    @staticmethod
    def _lowpass_alpha(cutoff_hz: float, sample_rate: int) -> float:
        dt = 1.0 / sample_rate
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        return float(dt / (rc + dt))

    def reset(self) -> None:
        self._hp_prev_input = 0.0
        self._hp_prev_output = 0.0
        self._lp_state_1 = 0.0
        self._lp_state_2 = 0.0
        self._compress_env = 0.0
        self._tremolo_phase = 0.0
        self._ring_mod_phase = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0 or not config.ENABLE_DROID_EFFECT:
            return samples

        out = np.empty_like(samples, dtype=np.float32)
        hp_alpha = self._hp_alpha
        lp_alpha = self._lp_alpha
        threshold = self._COMPRESS_THRESHOLD
        ratio = self._COMPRESS_RATIO
        attack_coeff = self._attack_coeff
        release_coeff = self._release_coeff
        makeup_gain = self._MAKEUP_GAIN
        saturation_drive = self._SATURATION_DRIVE
        saturation_norm = self._sat_norm
        tremolo_enabled = config.ENABLE_DROID_TREMOLO
        tremolo_rate = max(0.0, config.DROID_TREMOLO_RATE_HZ)
        tremolo_depth = config.DROID_TREMOLO_DEPTH
        tremolo_phase = self._tremolo_phase
        ring_mod_enabled = config.ENABLE_DROID_RING_MOD
        ring_mod_rate = max(0.0, config.DROID_RING_MOD_RATE_HZ)
        ring_mod_depth = config.DROID_RING_MOD_DEPTH
        ring_mod_phase = self._ring_mod_phase
        phase_scale = self._two_pi_over_sr
        crush_mix = (
            self._BITCRUSH_MIX if config.DROID_EFFECT_BITCRUSH_ENABLED else 0.0
        )
        crush_levels = self._bitcrush_levels
        hp_prev_input = self._hp_prev_input
        hp_prev_output = self._hp_prev_output
        lp_state_1 = self._lp_state_1
        lp_state_2 = self._lp_state_2
        compress_env = self._compress_env

        for i, sample in enumerate(samples):
            bandpassed = hp_alpha * (hp_prev_output + float(sample) - hp_prev_input)
            hp_prev_input = float(sample)
            hp_prev_output = bandpassed

            lp_state_1 += lp_alpha * (bandpassed - lp_state_1)
            lp_state_2 += lp_alpha * (lp_state_1 - lp_state_2)
            shaped = lp_state_2

            level = abs(shaped)
            coeff = attack_coeff if level > compress_env else release_coeff
            compress_env = coeff * compress_env + (1.0 - coeff) * level
            if compress_env > threshold:
                compressed = threshold + (compress_env - threshold) / ratio
                shaped *= (compressed / max(compress_env, 1e-6)) * makeup_gain
            else:
                shaped *= makeup_gain

            shaped = np.tanh(shaped * saturation_drive) * saturation_norm
            if ring_mod_enabled and ring_mod_depth > 0.0 and ring_mod_rate > 0.0:
                carrier = np.sin(ring_mod_phase)
                shaped = shaped * (1.0 - ring_mod_depth) + (
                    shaped * carrier * ring_mod_depth
                )
                ring_mod_phase += ring_mod_rate * phase_scale
                if ring_mod_phase >= 2.0 * np.pi:
                    ring_mod_phase -= 2.0 * np.pi
            elif tremolo_enabled and tremolo_depth > 0.0 and tremolo_rate > 0.0:
                tremolo = 1.0 - tremolo_depth + tremolo_depth * (
                    0.5 * (1.0 + np.sin(tremolo_phase))
                )
                shaped *= tremolo
                tremolo_phase += tremolo_rate * phase_scale
                if tremolo_phase >= 2.0 * np.pi:
                    tremolo_phase -= 2.0 * np.pi
            if crush_mix > 0.0:
                crushed = np.round(shaped * crush_levels) / crush_levels
                shaped += (crushed - shaped) * crush_mix

            if shaped > 1.0:
                shaped = 1.0
            elif shaped < -1.0:
                shaped = -1.0
            out[i] = shaped

        self._hp_prev_input = hp_prev_input
        self._hp_prev_output = hp_prev_output
        self._lp_state_1 = lp_state_1
        self._lp_state_2 = lp_state_2
        self._compress_env = compress_env
        self._tremolo_phase = tremolo_phase
        self._ring_mod_phase = ring_mod_phase
        return out


# ---------------------------------------------------------------------------
# AudioPlayer
# ---------------------------------------------------------------------------

class AudioPlayer:
    """Manages speech and music playback for DJ-R3X.

    Thread safety note: `_rms` is written only inside the sounddevice audio
    callback thread and read from the LED/main thread. CPython's GIL makes a
    single float assignment atomic enough for this use case.
    """

    def __init__(self) -> None:
        self._rms: float = 0.0           # smoothed 0.0–255.0; read by leds.py

        # Fired when the first audio samples of a new speech segment actually
        # reach the output device.  Used by LEDController.start_mouth() so the
        # mouth doesn't pre-glow before sound comes out of the speakers.
        self._audio_started: threading.Event = threading.Event()
        self._speech_status_logged: bool = False
        self._music_status_logged: bool = False

        # --- speech stream state ---
        self._speech_queue: queue.SimpleQueue[_SpeechChunk | _EndMarker] = (
            queue.SimpleQueue()
        )
        self._speech_buf: np.ndarray | None = None
        self._speech_buf_apply_droid_effect: bool = False
        self._speech_buf_pos: int = 0
        self._speech_active = threading.Event()
        self._speech_active.set()        # starts "idle"
        self._droid_effect = _DroidVoiceEffect(SPEECH_SAMPLE_RATE)

        # --- speech worker thread (opens/closes stream on demand) ---
        self._speech_stop = threading.Event()
        self._speech_thread = threading.Thread(
            target=self._speech_worker,
            daemon=True,
            name="djr3x-speech",
        )
        self._speech_thread.start()

        # --- music stream state ---
        self._music_thread: threading.Thread | None = None
        self._music_stop = threading.Event()
        self._music_volume: float = 1.0   # software fade multiplier (0.0–1.0)

    # ------------------------------------------------------------------
    # Speech — streaming TTS interface (called by synthesizer.py)
    # ------------------------------------------------------------------

    def feed_speech_chunk(self, pcm_bytes: bytes) -> None:
        """Push one raw PCM int16 chunk from the ElevenLabs stream into the
        playback queue. The chunk is played in order with any queued chunks."""
        if not pcm_bytes:
            return
        self._speech_queue.put(
            _SpeechChunk(
                samples=_pcm16_bytes_to_float32(pcm_bytes),
                apply_droid_effect=config.ENABLE_DROID_EFFECT,
            )
        )

    def end_speech(self) -> None:
        """Signal that the TTS stream is complete. The player will drain any
        remaining buffered audio then fall silent."""
        self._speech_active.clear()
        self._speech_queue.put(_EndMarker(done=None))

    def wait_for_speech(self, timeout: float = 30.0) -> bool:
        """Block until the current speech segment finishes (or timeout).
        Returns True if finished cleanly, False on timeout."""
        return self._speech_active.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Speech — cached file playback
    # ------------------------------------------------------------------

    def play_file(self, path: str | Path) -> None:
        """Load and play a cached .wav or .mp3 response file through the
        speech stream (mouth RMS tracking is active). Blocks until the file
        has finished playing or stop_speech() is called."""
        path = Path(path)
        data, sr = _load_audio_file(path, target_sr=SPEECH_SAMPLE_RATE)
        # Mix stereo (or higher) down to mono for the speech stream.
        if data.ndim == 2:
            data = data.mean(axis=1)
        samples = np.clip(data.astype(np.float32, copy=False), -1.0, 1.0)
        apply_droid_effect = _should_apply_droid_effect_to_file(path)

        done = threading.Event()
        self._speech_active.clear()

        for i in range(0, len(samples), _FILE_CHUNK_FRAMES):
            self._speech_queue.put(
                _SpeechChunk(
                    samples=samples[i : i + _FILE_CHUNK_FRAMES],
                    apply_droid_effect=apply_droid_effect,
                )
            )
        self._speech_queue.put(_EndMarker(done=done))

        # block until the callback processes the end marker, or timeout
        play_duration = len(samples) / SPEECH_SAMPLE_RATE
        done.wait(timeout=play_duration + 3.0)

    # ------------------------------------------------------------------
    # Speech — interrupt
    # ------------------------------------------------------------------

    def stop_speech(self) -> None:
        """Immediately silence speech playback and flush all queued audio.
        Sets any pending done events so callers blocked in play_file() unblock.
        """
        # drain the queue, firing any pending done events
        while True:
            try:
                item = self._speech_queue.get_nowait()
                if isinstance(item, _EndMarker) and item.done is not None:
                    item.done.set()
            except queue.Empty:
                break

        # reset buffer state (audio callback checks these)
        self._speech_buf = None
        self._speech_buf_apply_droid_effect = False
        self._speech_buf_pos = 0
        self._droid_effect.reset()
        self._rms = 0.0
        self._audio_started.clear()
        self._speech_active.set()   # unblock any wait_for_speech() caller

        # put an EndMarker so the callback raises CallbackStop and closes
        # the stream if one is currently open
        self._speech_queue.put(_EndMarker(done=None))

    # ------------------------------------------------------------------
    # Music
    # ------------------------------------------------------------------

    def play_music(self, path: str | Path, loop: bool = False) -> None:
        """Play a music file in a background thread. Replaces any currently
        playing music. No mouth-LED RMS tracking on this stream."""
        self.stop_music()
        self._music_stop.clear()
        self._music_thread = threading.Thread(
            target=self._music_worker,
            args=(Path(path), loop),
            daemon=True,
            name="djr3x-music",
        )
        self._music_thread.start()

    def stop_music(self) -> None:
        """Stop background music and wait for the worker thread to exit."""
        self._music_stop.set()
        if self._music_thread is not None:
            self._music_thread.join(timeout=2.0)
            self._music_thread = None

    @property
    def is_music_playing(self) -> bool:
        """True while a music track is actively playing in the background."""
        return self._music_thread is not None and self._music_thread.is_alive()

    def fade_music(self, duration: float = 3.0) -> None:
        """Gradually reduce music volume to 0 over *duration* seconds, then stop.

        Blocks the caller for *duration* seconds while stepping the volume down
        in 100 ms increments.  Resets _music_volume to 1.0 after stopping so
        the next play_music() call starts at full volume.
        """
        steps = max(1, int(duration / 0.1))
        for i in range(steps):
            self._music_volume = 1.0 - float(i + 1) / steps
            time.sleep(0.1)
        self._music_volume = 0.0
        self.stop_music()
        self._music_volume = 1.0   # reset for next play

    def wait_for_music(self, timeout: float = 30.0) -> bool:
        """Block until the current music/chime thread exits (or timeout).
        Returns True if the thread finished cleanly, False on timeout."""
        t = self._music_thread
        if t is None or not t.is_alive():
            return True
        t.join(timeout=timeout)
        return not t.is_alive()

    def play_chime(self) -> None:
        """Play the startup chime through the music output path (no mouth-LED
        RMS tracking).  Uses ffmpeg to decode the MP3 to raw PCM in memory.
        No-ops silently if the file is missing or ffmpeg is unavailable."""
        path = Path(config.STARTUP_CHIME_PATH)
        if not path.exists():
            return
        try:
            data, sr = _load_audio_file(path, target_sr=None)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return
        self.stop_music()
        self._music_stop.clear()
        self._music_thread = threading.Thread(
            target=self._music_worker_array,
            args=(data, sr),
            daemon=True,
            name="djr3x-chime",
        )
        self._music_thread.start()

    # ------------------------------------------------------------------
    # RMS — read by leds.py
    # ------------------------------------------------------------------

    @property
    def rms(self) -> float:
        """Smoothed, gain-scaled RMS of the speech output. Range 0.0–255.0.
        Updated on every audio callback (~64 ms at 16000 Hz / 1024 frames).
        Returns 0.0 when no speech stream is open."""
        return self._rms

    def wait_for_audio_start(self, timeout: float = 5.0) -> bool:
        """Block until the first audio samples of the current speech segment
        actually reach the output device — i.e. the OutputStream callback has
        fired with non-silent data at least once.

        Returns True if audio started within timeout, False otherwise.
        Call this before starting mouth LEDs to avoid pre-glow.
        """
        return self._audio_started.wait(timeout=timeout)

    def clear_audio_started(self) -> None:
        """Reset the audio-started event for a new speech segment.

        Must be called at the start of each _begin_speech() so that
        wait_for_audio_start() always waits for THIS segment's first chunk,
        not a stale value left set by the previous speech.  Without this,
        the mouth-trigger thread returns immediately on every utterance
        after the first, causing pre-glow before audio actually plays.
        """
        self._audio_started.clear()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop all playback and release PortAudio resources."""
        self.stop_speech()
        self.stop_music()
        self._speech_stop.set()
        self._speech_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internal — speech worker thread
    # ------------------------------------------------------------------

    def _speech_worker(self) -> None:
        """Background thread: waits for audio data, then opens an OutputStream
        for exactly the duration of one speech segment and closes it afterward.
        The device is fully released between segments."""
        while not self._speech_stop.is_set():
            # Block until the first item arrives (poll so we can check stop flag).
            try:
                item = self._speech_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if isinstance(item, _EndMarker):
                # EndMarker before any audio (e.g. stop_speech() or end_speech()
                # called before any chunks were queued).
                self._speech_active.set()
                if item.done is not None:
                    item.done.set()
                continue

            # First audio chunk — prime the buffer, then open the stream.
            self._speech_buf = item.samples
            self._speech_buf_apply_droid_effect = item.apply_droid_effect
            self._speech_buf_pos = 0
            self._droid_effect.reset()
            self._audio_started.clear()   # arm the event; callback sets it on first samples

            finished = threading.Event()
            with _open_output_stream(
                samplerate=SPEECH_SAMPLE_RATE,
                channels=config.AUDIO_OUTPUT_CHANNELS,
                dtype="int16",
                blocksize=config.SPEECH_OUTPUT_BLOCKSIZE,
                latency=config.SPEECH_OUTPUT_LATENCY,
                callback=self._speech_callback,
                finished_callback=finished.set,
            ):
                finished.wait()

            # Stream is fully closed — zero out RMS so LEDs go dark.
            self._rms = 0.0
            self._speech_status_logged = False

    # ------------------------------------------------------------------
    # Internal — speech callback (sounddevice audio thread)
    # ------------------------------------------------------------------

    def _speech_callback(
        self,
        outdata: np.ndarray,    # shape (blocksize, channels), dtype int16
        frames: int,
        _time,                  # CffiData timestamp — unused
        _status: sd.CallbackFlags,
    ) -> None:
        if _status and not self._speech_status_logged:
            log.warning("AudioPlayer speech callback status: %s", _status)
            self._speech_status_logged = True

        # Build mono scratch buffer; broadcast to all output channels at the end.
        mono = np.zeros(frames, dtype=np.float32)
        filled = 0
        block_apply_droid_effect = False
        stop_stream = False

        while filled < frames:
            # refill internal chunk buffer from queue when exhausted
            if self._speech_buf is None or self._speech_buf_pos >= len(self._speech_buf):
                try:
                    item = self._speech_queue.get_nowait()
                except queue.Empty:
                    break   # nothing queued yet — output silence this block

                if isinstance(item, _EndMarker):
                    self._speech_buf = None
                    self._speech_buf_apply_droid_effect = False
                    self._speech_buf_pos = 0
                    self._speech_active.set()
                    if item.done is not None:
                        item.done.set()
                    stop_stream = True
                    break
                self._speech_buf = item.samples
                self._speech_buf_apply_droid_effect = item.apply_droid_effect
                self._speech_buf_pos = 0

            take = min(
                frames - filled,
                len(self._speech_buf) - self._speech_buf_pos,
            )
            mono[filled : filled + take] = (
                self._speech_buf[self._speech_buf_pos : self._speech_buf_pos + take]
            )
            block_apply_droid_effect = self._speech_buf_apply_droid_effect
            self._speech_buf_pos += take
            filled += take

        if filled > 0:
            if block_apply_droid_effect:
                mono[:filled] = self._droid_effect.process(mono[:filled])

            if config.AUDIO_VOLUME != 1.0:
                mono[:filled] = np.clip(
                    mono[:filled] * config.AUDIO_VOLUME,
                    -1.0,
                    1.0,
                )
            else:
                np.clip(mono[:filled], -1.0, 1.0, out=mono[:filled])

            rms_raw = float(np.sqrt(np.mean(mono[:filled] ** 2)))
            brightness = min(255.0, rms_raw * config.MOUTH_LED_GAIN * 255.0)
            # Signal on the first non-silent callback so mouth LEDs can start
            # at the exact moment audio reaches the output device.
            if rms_raw > 1e-4 and not self._audio_started.is_set():
                log.info("First audio chunk playing — mouth LED start unlocked")
                self._audio_started.set()
        else:
            brightness = 0.0

        outdata[:] = _float32_to_int16(mono)[:, np.newaxis]

        # exponential smoothing to prevent harsh LED flicker
        alpha = config.MOUTH_LED_SMOOTHING
        self._rms = alpha * brightness + (1.0 - alpha) * self._rms

        if stop_stream:
            raise sd.CallbackStop()

    # ------------------------------------------------------------------
    # Internal — music worker thread
    # ------------------------------------------------------------------

    def _music_worker_array(self, data: np.ndarray, sr: int) -> None:
        """Background thread: play a pre-loaded float32 array once as music."""
        if data.ndim == 1:
            data = data[:, np.newaxis]
        data = data.astype(np.float32)
        self._music_worker_play(data, sr, loop=False)

    def _music_worker(self, path: Path, loop: bool) -> None:
        """Background thread: opens an explicit OutputStream for music so it
        is fully independent from the speech stream and sd.play()."""
        data, sr = _load_audio_file(path, target_sr=None)   # keep native rate

        # normalise to float32, ensure 2-D (frames × channels)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        data = data.astype(np.float32)
        self._music_worker_play(data, sr, loop=loop)

    def _music_worker_play(self, data: np.ndarray, sr: int, loop: bool) -> None:
        """Shared playback loop used by both _music_worker and _music_worker_array."""
        channels = data.shape[1]

        while not self._music_stop.is_set():
            pos = 0
            finished = threading.Event()

            def _callback(outdata: np.ndarray, frames: int, _t, _s) -> None:
                nonlocal pos
                if _s and not self._music_status_logged:
                    log.warning("AudioPlayer music callback status: %s", _s)
                    self._music_status_logged = True
                if self._music_stop.is_set():
                    outdata[:] = 0.0
                    raise sd.CallbackStop()

                remaining = len(data) - pos
                take = min(frames, remaining)
                outdata[:take] = data[pos : pos + take] * config.AUDIO_VOLUME * self._music_volume
                if take < frames:
                    outdata[take:] = 0.0
                pos += take

                if pos >= len(data):
                    raise sd.CallbackStop()

            with _open_output_stream(
                samplerate=sr,
                channels=channels,
                dtype="float32",
                blocksize=config.AUDIO_OUTPUT_BLOCKSIZE,
                latency=config.AUDIO_OUTPUT_LATENCY,
                callback=_callback,
                finished_callback=finished.set,
            ):
                finished.wait()

            self._music_status_logged = False

            if not loop or self._music_stop.is_set():
                break


# ---------------------------------------------------------------------------
# File loading helper
# ---------------------------------------------------------------------------

def _load_audio_file(
    path: Path, target_sr: int | None
) -> tuple[np.ndarray, int]:
    """Load a .wav or .mp3 file as a float32 numpy array.

    Returns (data, sample_rate).
    - data shape: (frames,) for mono, (frames, channels) for stereo.
    - If target_sr is given and differs from the file's native rate, the
      audio is resampled using scipy.signal.resample_poly. scipy is an
      optional dependency; a clear error is raised if it is absent.
    """
    suffix = path.suffix.lower()

    if suffix == ".mp3":
        ffmpeg_cmd = [
            "ffmpeg", "-loglevel", "error", "-i", str(path),
        ]
        if target_sr is not None:
            ffmpeg_cmd.extend(["-ar", str(target_sr)])
        ffmpeg_cmd.extend(["-f", "wav", "pipe:1"])
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            check=True,
        )
        data, sr = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=False)
    else:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)

    if target_sr is not None and sr != target_sr:
        try:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, target_sr)
            up, down = target_sr // g, sr // g
            data = resample_poly(data, up, down, axis=0).astype(np.float32)
        except ImportError:
            raise RuntimeError(
                f"File '{path}' is at {sr} Hz but target is {target_sr} Hz. "
                "Install scipy to enable resampling: pip install scipy"
            )
        sr = target_sr

    return data, sr


def _pcm16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _float32_to_int16(samples: np.ndarray) -> np.ndarray:
    return np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)


def _should_apply_droid_effect_to_file(path: Path) -> bool:
    if not config.ENABLE_DROID_EFFECT or path.suffix.lower() != ".wav":
        return False
    try:
        return path.resolve().parent == config.AUDIO_CACHE_DIR.resolve()
    except OSError:
        return False


def _output_devices_to_try() -> list[int | None]:
    device = config.AUDIO_OUTPUT_DEVICE
    return [device, None] if device is not None else [None]


def _open_output_stream(**kwargs):
    last_exc: sd.PortAudioError | None = None
    for device in _output_devices_to_try():
        try:
            if device is None and config.AUDIO_OUTPUT_DEVICE is not None:
                log.warning("AudioPlayer: falling back to default output device")
            return sd.OutputStream(device=device, **kwargs)
        except sd.PortAudioError as exc:
            last_exc = exc
            log.warning("AudioPlayer: output device %s failed — %s", device, exc)
    assert last_exc is not None
    raise last_exc
