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
import re
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
_MAC_RE = re.compile(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}")


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
    sample_rate: int = SPEECH_SAMPLE_RATE


@dataclass
class _BluetoothDevice:
    mac: str
    name: str = ""
    alias: str = ""
    paired: bool = False
    trusted: bool = False
    connected: bool = False
    has_audio_sink: bool = False

    @property
    def display_name(self) -> str:
        return self.alias or self.name or self.mac


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


class _BluetoothAudioManager:
    """Resolve a paired Bluetooth audio sink into a live PortAudio device."""

    def __init__(self) -> None:
        self._enabled = config.AUDIO_OUTPUT_MODE == "bluetooth"
        self._lock = threading.Lock()
        self._cached_output_index: int | None = None
        self._cached_output_name: str = ""
        self._cached_target_mac: str = ""
        self._default_sink_name: str = ""
        self._default_sink_target_mac: str = ""
        # Time-based short-circuit: if we successfully resolved within this
        # window, skip re-running bluetoothctl (avoids spurious reconnects).
        self._last_resolve_time: float = 0.0
        self._resolve_cache_ttl: float = 15.0

        if self._enabled:
            log.info(
                "AudioPlayer: Bluetooth output mode enabled (target=%s)",
                config.AUDIO_BLUETOOTH_DEVICE or "auto",
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def preferred_output_devices(self) -> list[int | None]:
        if not self._enabled:
            device = config.AUDIO_OUTPUT_DEVICE
            return [device, None] if device is not None else [None]

        candidates: list[int | None] = []
        with self._lock:
            bt_device = self._resolve_output_device()
        if bt_device is not None:
            candidates.append(bt_device)
        elif self._default_sink_name:
            candidates.append(None)
        if config.AUDIO_OUTPUT_DEVICE is not None and config.AUDIO_OUTPUT_DEVICE not in candidates:
            candidates.append(config.AUDIO_OUTPUT_DEVICE)
        if None not in candidates:
            candidates.append(None)
        return candidates

    def _resolve_output_device(self) -> int | None:
        cached = self._cached_output_index
        if cached is not None:
            if self._device_index_exists(cached):
                return cached
            self._clear_cache()

        # Short-circuit: if we recently confirmed the default sink is set,
        # skip the expensive bluetoothctl calls to avoid spurious reconnects.
        if (
            self._default_sink_name
            and time.monotonic() - self._last_resolve_time < self._resolve_cache_ttl
        ):
            return None

        target = self._select_target_device()
        if target is None:
            log.warning("AudioPlayer: no paired Bluetooth audio sink matched %r", config.AUDIO_BLUETOOTH_DEVICE)
            return None

        ready = target.connected
        if not ready and config.AUDIO_BLUETOOTH_AUTO_CONNECT:
            ready = self._connect_device(target.mac)
            if ready:
                target = self._device_info(target.mac) or target

        if not ready:
            log.warning(
                "AudioPlayer: Bluetooth device %s is not connected",
                target.display_name,
            )
            self._default_sink_name = ""
            self._default_sink_target_mac = ""
            return None

        sink_name = self._set_default_sink(target)

        match = _find_best_output_device(_bluetooth_match_tokens(target))
        if match is None and sink_name:
            self._default_sink_name = sink_name
            self._default_sink_target_mac = target.mac
            self._last_resolve_time = time.monotonic()
            log.info(
                "AudioPlayer: using system default output routed to %s for %s",
                sink_name,
                target.display_name,
            )
            return None

        match = match or self._find_portaudio_output(target)
        if match is None:
            log.warning(
                "AudioPlayer: Bluetooth device %s connected but no PortAudio output appeared",
                target.display_name,
            )
            self._default_sink_name = ""
            self._default_sink_target_mac = ""
            return None

        index, name = match
        self._cached_output_index = index
        self._cached_output_name = name
        self._cached_target_mac = target.mac
        self._default_sink_name = ""
        self._default_sink_target_mac = ""
        self._last_resolve_time = time.monotonic()
        log.info(
            "AudioPlayer: using Bluetooth output device %d (%s) for %s",
            index,
            name,
            target.display_name,
        )
        return index

    def _clear_cache(self) -> None:
        self._cached_output_index = None
        self._cached_output_name = ""
        self._cached_target_mac = ""
        self._default_sink_name = ""
        self._default_sink_target_mac = ""
        self._last_resolve_time = 0.0

    def _device_index_exists(self, index: int) -> bool:
        try:
            sd.query_devices(index, kind="output")
            return True
        except Exception:
            return False

    def _select_target_device(self) -> _BluetoothDevice | None:
        paired = self._paired_audio_devices()
        if not paired:
            return None

        target = config.AUDIO_BLUETOOTH_DEVICE.strip()
        if not target or target.lower() == "auto":
            return paired[0]

        target_norm = _normalize_token(target)
        if _MAC_RE.fullmatch(target):
            exact = next((d for d in paired if d.mac.lower() == target.lower()), None)
            if exact is not None:
                return exact

        for device in paired:
            haystacks = {
                _normalize_token(device.mac),
                _normalize_token(device.alias),
                _normalize_token(device.name),
            }
            if target_norm and any(target_norm and target_norm in h for h in haystacks if h):
                return device
        return None

    def _paired_audio_devices(self) -> list[_BluetoothDevice]:
        devices = self._list_paired_devices()
        audio_devices: list[_BluetoothDevice] = []
        for mac in devices:
            info = self._device_info(mac)
            if info is None or not info.paired or not info.has_audio_sink:
                continue
            audio_devices.append(info)

        def _sort_key(device: _BluetoothDevice) -> tuple[int, int, str]:
            connected_rank = 0 if (config.AUDIO_BLUETOOTH_PREFER_CONNECTED and device.connected) else 1
            trusted_rank = 0 if device.trusted else 1
            return (connected_rank, trusted_rank, device.display_name.lower())

        audio_devices.sort(key=_sort_key)
        return audio_devices

    def _list_paired_devices(self) -> list[str]:
        outputs: list[str] = []
        for args in (["devices", "Paired"], ["paired-devices"]):
            stdout = self._run_command(["bluetoothctl", *args], timeout=5.0)
            if not stdout:
                continue
            outputs = _parse_bluetoothctl_devices(stdout)
            if outputs:
                break
        return outputs

    def _device_info(self, mac: str) -> _BluetoothDevice | None:
        stdout = self._run_command(["bluetoothctl", "info", mac], timeout=5.0)
        if not stdout:
            return None
        return _parse_bluetoothctl_info(stdout, mac)

    def _connect_device(self, mac: str) -> bool:
        self._clear_cache()
        log.info("AudioPlayer: connecting Bluetooth audio device %s …", mac)
        self._run_command(["bluetoothctl", "connect", mac], timeout=config.AUDIO_BLUETOOTH_CONNECT_TIMEOUT)
        deadline = time.monotonic() + max(1.0, config.AUDIO_BLUETOOTH_CONNECT_TIMEOUT)
        while time.monotonic() < deadline:
            info = self._device_info(mac)
            if info is not None and info.connected:
                return True
            time.sleep(0.5)
        return False

    def _find_portaudio_output(self, device: _BluetoothDevice) -> tuple[int, str] | None:
        deadline = time.monotonic() + max(1.0, config.AUDIO_BLUETOOTH_DISCOVERY_TIMEOUT)
        tokens = _bluetooth_match_tokens(device)
        while time.monotonic() < deadline:
            match = _find_best_output_device(tokens)
            if match is not None:
                return match
            time.sleep(0.5)
        return None

    def _set_default_sink(self, device: _BluetoothDevice) -> str:
        sinks = self._list_pactl_sinks()
        if not sinks:
            return ""
        tokens = set(_bluetooth_match_tokens(device))
        for sink_name in sinks:
            sink_lower = sink_name.lower()
            sink_norm = _normalize_token(sink_name)
            if any(token and (token in sink_lower or token in sink_norm) for token in tokens):
                self._run_command(["pactl", "set-default-sink", sink_name], timeout=5.0)
                log.info(
                    "AudioPlayer: set default PipeWire/Pulse sink to %s for %s",
                    sink_name,
                    device.display_name,
                )
                return sink_name
        return ""

    def _list_pactl_sinks(self) -> list[str]:
        stdout = self._run_command(["pactl", "list", "short", "sinks"], timeout=5.0)
        sinks: list[str] = []
        for line in stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].strip():
                sinks.append(parts[1].strip())
        return sinks

    @staticmethod
    def _run_command(args: list[str], timeout: float) -> str:
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            log.warning("AudioPlayer: command failed (%s): %s", " ".join(args), exc)
            return ""
        if result.returncode != 0 and result.stderr.strip():
            log.debug(
                "AudioPlayer: command %s returned %d: %s",
                " ".join(args),
                result.returncode,
                result.stderr.strip(),
            )
        return result.stdout


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
        self._bluetooth_audio = _BluetoothAudioManager()
        self._speech_stream_sample_rate: int = SPEECH_SAMPLE_RATE
        self._bluetooth_keepalive_stop = threading.Event()
        self._bluetooth_keepalive_thread: threading.Thread | None = None
        self._bluetooth_output_ready = threading.Event()

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

        if self._bluetooth_audio.enabled:
            self._bluetooth_keepalive_thread = threading.Thread(
                target=self._bluetooth_keepalive_worker,
                daemon=True,
                name="djr3x-bt-keepalive",
            )
            self._bluetooth_keepalive_thread.start()

    # ------------------------------------------------------------------
    # Speech — streaming TTS interface (called by synthesizer.py)
    # ------------------------------------------------------------------

    def feed_speech_chunk(
        self,
        pcm_bytes: bytes,
        *,
        sample_rate: int = SPEECH_SAMPLE_RATE,
    ) -> None:
        """Push one raw PCM int16 chunk from the ElevenLabs stream into the
        playback queue. The chunk is played in order with any queued chunks."""
        if not pcm_bytes:
            return
        self._speech_queue.put(
            _SpeechChunk(
                samples=_pcm16_bytes_to_float32(pcm_bytes),
                apply_droid_effect=config.ENABLE_DROID_EFFECT,
                sample_rate=sample_rate,
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
        data, sr = _load_audio_file(path, target_sr=None)
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
                    sample_rate=sr,
                )
            )
        self._speech_queue.put(_EndMarker(done=done))

        # block until the callback processes the end marker, or timeout
        play_duration = len(samples) / sr
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
            target=self._music_worker_safe,
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
            target=self._music_worker_array_safe,
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
        """Reset per-segment speech timing state before a new utterance.

        Must be called at the start of each _begin_speech() so that
        wait_for_audio_start() always waits for THIS segment's first chunk,
        not a stale value left set by the previous speech.  Without this,
        the mouth-trigger thread returns immediately on every utterance
        after the first, causing pre-glow before audio actually plays.

        Also clears any stale RMS left over from the previous segment so
        speech-reactive motion cannot inherit a non-zero level before the
        new audio stream has actually started.
        """
        self._audio_started.clear()
        self._rms = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop all playback and release PortAudio resources."""
        self.stop_speech()
        self.stop_music()
        self._speech_stop.set()
        self._speech_thread.join(timeout=2.0)
        self._bluetooth_output_ready.clear()
        self._bluetooth_keepalive_stop.set()
        if self._bluetooth_keepalive_thread is not None:
            self._bluetooth_keepalive_thread.join(timeout=2.0)

    def wait_for_bluetooth_output(self, timeout: float) -> bool:
        """Wait until a Bluetooth output device is live for this process.

        Returns True immediately when Bluetooth mode is disabled.
        """
        if not self._bluetooth_audio.enabled:
            return True
        return self._bluetooth_output_ready.wait(timeout=timeout)

    def _bluetooth_keepalive_worker(self) -> None:
        """Hold a silent Bluetooth output stream open for the process lifetime.

        PipeWire/Bluetooth sink discovery can lag behind the initial
        `bluetoothctl` connection, and some receivers are more stable when an
        output stream remains attached. This worker retries until a compatible
        Bluetooth output appears, then keeps a silent stream open until
        shutdown.
        """
        retry_delay = 2.0
        warned_waiting = False

        while not self._bluetooth_keepalive_stop.is_set():
            try:
                with self._bluetooth_audio._lock:
                    bt_device = self._bluetooth_audio._resolve_output_device()
                    default_sink_name = self._bluetooth_audio._default_sink_name
                devices_to_try = [bt_device] if bt_device is not None else ([None] if default_sink_name else [])
                if not devices_to_try:
                    raise sd.PortAudioError("Bluetooth output device not ready yet")
                device, stream_sr, label = _resolve_output_stream_settings_for_devices(
                    devices_to_try,
                    requested_samplerate=SPEECH_SAMPLE_RATE,
                    channels=config.AUDIO_OUTPUT_CHANNELS,
                    dtype="int16",
                )
                self._bluetooth_output_ready.set()
                if warned_waiting:
                    log.info("AudioPlayer: Bluetooth keepalive attached to %s", label)
                    warned_waiting = False

                _last_underflow_warn: list[float] = [0.0]

                def _callback(outdata: np.ndarray, _frames: int, _time, status) -> None:
                    if status:
                        now = time.monotonic()
                        if now - _last_underflow_warn[0] >= 10.0:
                            log.warning("AudioPlayer Bluetooth keepalive status: %s", status)
                            _last_underflow_warn[0] = now
                    outdata[:] = 0
                    if self._bluetooth_keepalive_stop.is_set():
                        raise sd.CallbackStop()

                finished = threading.Event()
                # BT A2DP has inherent latency; "high" prevents underflows in
                # the keepalive stream without affecting audible output quality.
                with sd.OutputStream(
                    device=device,
                    samplerate=stream_sr,
                    channels=config.AUDIO_OUTPUT_CHANNELS,
                    dtype="int16",
                    blocksize=config.SPEECH_OUTPUT_BLOCKSIZE,
                    latency="high",
                    callback=_callback,
                    finished_callback=finished.set,
                ):
                    while not self._bluetooth_keepalive_stop.wait(1.0):
                        pass
                    finished.wait(timeout=1.0)
            except Exception as exc:
                self._bluetooth_output_ready.clear()
                if not warned_waiting:
                    log.info(
                        "AudioPlayer: waiting for Bluetooth output device to appear for keepalive: %s",
                        exc,
                    )
                    warned_waiting = True
                self._bluetooth_keepalive_stop.wait(retry_delay)

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
            self._speech_buf_apply_droid_effect = item.apply_droid_effect
            self._droid_effect.reset()
            self._audio_started.clear()   # arm the event; callback sets it on first samples

            try:
                requested_sample_rate = item.sample_rate or SPEECH_SAMPLE_RATE
                device, stream_sr, _name = _resolve_output_stream_settings(
                    self,
                    requested_samplerate=requested_sample_rate,
                    channels=config.AUDIO_OUTPUT_CHANNELS,
                    dtype="int16",
                )
                self._speech_stream_sample_rate = stream_sr
                if self._droid_effect._sample_rate != stream_sr:
                    self._droid_effect = _DroidVoiceEffect(stream_sr)
                self._speech_buf = _prepare_speech_chunk_for_output(item, stream_sr)
                self._speech_buf_pos = 0
                preroll_frames_remaining = max(
                    0,
                    int(config.AUDIO_OUTPUT_PREROLL_SECONDS * stream_sr),
                )

                finished = threading.Event()
                with sd.OutputStream(
                    device=device,
                    samplerate=stream_sr,
                    channels=config.AUDIO_OUTPUT_CHANNELS,
                    dtype="int16",
                    blocksize=config.SPEECH_OUTPUT_BLOCKSIZE,
                    latency=config.SPEECH_OUTPUT_LATENCY,
                    callback=self._speech_callback,
                    finished_callback=finished.set,
                ):
                    if preroll_frames_remaining > 0:
                        log.info(
                            "AudioPlayer: applying %.0f ms speech output preroll",
                            (preroll_frames_remaining / stream_sr) * 1000.0,
                        )
                    self._speech_preroll_frames_remaining = preroll_frames_remaining
                    finished.wait()
            except Exception:
                log.exception("AudioPlayer: speech stream failed")
                self._speech_active.set()
            finally:
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

        preroll_remaining = getattr(self, "_speech_preroll_frames_remaining", 0)
        if preroll_remaining > 0:
            take_silence = min(frames, preroll_remaining)
            self._speech_preroll_frames_remaining = preroll_remaining - take_silence
            if take_silence >= frames:
                outdata[:] = 0
                self._rms = 0.0
                return
            filled = take_silence

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
                self._speech_buf = _prepare_speech_chunk_for_output(
                    item,
                    self._speech_stream_sample_rate,
                )
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

    def _music_worker_array_safe(self, data: np.ndarray, sr: int) -> None:
        try:
            self._music_worker_array(data, sr)
        except Exception:
            log.exception("AudioPlayer: chime playback failed")

    def _music_worker(self, path: Path, loop: bool) -> None:
        """Background thread: opens an explicit OutputStream for music so it
        is fully independent from the speech stream and sd.play()."""
        data, sr = _load_audio_file(path, target_sr=None)   # keep native rate

        # normalise to float32, ensure 2-D (frames × channels)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        data = data.astype(np.float32)
        self._music_worker_play(data, sr, loop=loop)

    def _music_worker_safe(self, path: Path, loop: bool) -> None:
        try:
            self._music_worker(path, loop)
        except Exception:
            log.exception("AudioPlayer: music playback failed")

    def _music_worker_play(self, data: np.ndarray, sr: int, loop: bool) -> None:
        """Shared playback loop used by both _music_worker and _music_worker_array."""
        channels = data.shape[1]

        while not self._music_stop.is_set():
            pos = 0
            finished = threading.Event()
            device, stream_sr, _name = _resolve_output_stream_settings(
                self,
                requested_samplerate=sr,
                channels=channels,
                dtype="float32",
            )
            stream_data = (
                _resample_audio_array(data, sr, stream_sr, axis=0)
                if stream_sr != sr
                else data
            )
            preroll_frames_remaining = max(
                0,
                int(config.AUDIO_OUTPUT_PREROLL_SECONDS * stream_sr),
            )

            def _callback(outdata: np.ndarray, frames: int, _t, _s) -> None:
                nonlocal pos
                nonlocal preroll_frames_remaining
                if _s and not self._music_status_logged:
                    log.warning("AudioPlayer music callback status: %s", _s)
                    self._music_status_logged = True
                if self._music_stop.is_set():
                    outdata[:] = 0.0
                    raise sd.CallbackStop()

                if preroll_frames_remaining > 0:
                    take_silence = min(frames, preroll_frames_remaining)
                    outdata[:take_silence] = 0.0
                    preroll_frames_remaining -= take_silence
                    if take_silence >= frames:
                        return
                    out_offset = take_silence
                else:
                    out_offset = 0

                remaining = len(stream_data) - pos
                take = min(frames - out_offset, remaining)
                outdata[out_offset : out_offset + take] = (
                    stream_data[pos : pos + take]
                    * config.AUDIO_VOLUME
                    * self._music_volume
                )
                if out_offset + take < frames:
                    outdata[out_offset + take :] = 0.0
                pos += take

                if pos >= len(stream_data):
                    raise sd.CallbackStop()

            with sd.OutputStream(
                device=device,
                samplerate=stream_sr,
                channels=channels,
                dtype="float32",
                blocksize=config.AUDIO_OUTPUT_BLOCKSIZE,
                latency=config.AUDIO_OUTPUT_LATENCY,
                callback=_callback,
                finished_callback=finished.set,
            ):
                if preroll_frames_remaining > 0:
                    log.info(
                        "AudioPlayer: applying %.0f ms music output preroll",
                        (preroll_frames_remaining / stream_sr) * 1000.0,
                    )
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
        resolved_parent = path.resolve().parent
        return resolved_parent in {
            config.AUDIO_CACHE_DIR.resolve(),
            config.LEGACY_AUDIO_CACHE_DIR.resolve(),
        }
    except OSError:
        return False


def _output_devices_to_try() -> list[int | None]:
    # Backwards-compatible helper retained for non-class callers; the
    # AudioPlayer instance uses its Bluetooth manager instead.
    device = config.AUDIO_OUTPUT_DEVICE
    return [device, None] if device is not None else [None]


def _output_devices_from_player(player: AudioPlayer | None) -> list[int | None]:
    if player is not None:
        return player._bluetooth_audio.preferred_output_devices()
    return _output_devices_to_try()


def _resolve_output_stream_settings(
    player: AudioPlayer | None,
    *,
    requested_samplerate: int,
    channels: int,
    dtype: str,
) -> tuple[int | None, int, str]:
    return _resolve_output_stream_settings_for_devices(
        _output_devices_from_player(player),
        requested_samplerate=requested_samplerate,
        channels=channels,
        dtype=dtype,
    )


def _resolve_output_stream_settings_for_devices(
    devices_to_try: list[int | None],
    *,
    requested_samplerate: int,
    channels: int,
    dtype: str,
) -> tuple[int | None, int, str]:
    last_exc: sd.PortAudioError | None = None
    seen_devices: set[int | None] = set()

    for device_choice in devices_to_try:
        for device in _expand_output_device_choice(device_choice):
            if device in seen_devices:
                continue
            seen_devices.add(device)
            label = _device_label(device)
            for samplerate in _candidate_output_sample_rates(
                device,
                requested_samplerate=requested_samplerate,
            ):
                try:
                    sd.check_output_settings(
                        device=device,
                        samplerate=samplerate,
                        channels=channels,
                        dtype=dtype,
                    )
                    if samplerate != requested_samplerate:
                        log.info(
                            "AudioPlayer: using output sample rate %d Hz on %s (requested %d Hz)",
                            samplerate,
                            label,
                            requested_samplerate,
                        )
                    return device, samplerate, label
                except sd.PortAudioError as exc:
                    last_exc = exc
            log.warning(
                "AudioPlayer: output device %s does not accept %d ch %s at usable sample rates",
                label,
                channels,
                dtype,
            )

    assert last_exc is not None
    raise last_exc


def _parse_bluetoothctl_devices(output: str) -> list[str]:
    devices: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("Device "):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) >= 2 and _MAC_RE.fullmatch(parts[1]):
            devices.append(parts[1])
    return devices


def _parse_bluetoothctl_info(output: str, mac: str) -> _BluetoothDevice:
    device = _BluetoothDevice(mac=mac)
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Name: "):
            device.name = line.split(": ", 1)[1].strip()
        elif line.startswith("Alias: "):
            device.alias = line.split(": ", 1)[1].strip()
        elif line.startswith("Paired: "):
            device.paired = line.split(": ", 1)[1].strip().lower() == "yes"
        elif line.startswith("Trusted: "):
            device.trusted = line.split(": ", 1)[1].strip().lower() == "yes"
        elif line.startswith("Connected: "):
            device.connected = line.split(": ", 1)[1].strip().lower() == "yes"
        elif "Audio Sink" in line:
            device.has_audio_sink = True
    return device


def _normalize_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _mac_variants(mac: str) -> list[str]:
    compact = _normalize_token(mac)
    raw_lower = mac.lower()
    underscored = raw_lower.replace(":", "_")
    dashed = raw_lower.replace(":", "-")
    dotted = raw_lower.replace(":", ".")
    return [raw_lower, compact, underscored, dashed, dotted]


def _bluetooth_match_tokens(device: _BluetoothDevice) -> list[str]:
    tokens: list[str] = []
    for value in (device.alias, device.name):
        if value:
            tokens.append(value.lower())
            tokens.append(_normalize_token(value))
    for variant in _mac_variants(device.mac):
        tokens.append(variant)
        tokens.append(f"bluez_output.{variant}")
    seen: set[str] = set()
    unique_tokens: list[str] = []
    for token in tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        unique_tokens.append(token)
    return unique_tokens


def _find_best_output_device(tokens: list[str]) -> tuple[int, str] | None:
    try:
        devices = sd.query_devices()
    except Exception as exc:
        log.warning("AudioPlayer: could not query output devices: %s", exc)
        return None

    best: tuple[int, str, int] | None = None
    for index, info in enumerate(devices):
        if int(info.get("max_output_channels", 0)) <= 0:
            continue
        name = str(info.get("name", ""))
        name_lower = name.lower()
        name_norm = _normalize_token(name)
        score = 0
        for token in tokens:
            if token in name_lower:
                score = max(score, 6 if ":" in token or "_" in token or "bluez_output" in token else 4)
            elif token in name_norm:
                score = max(score, 5)
        if "bluez" in name_lower or "bluetooth" in name_lower:
            score += 1
        if score <= 0:
            continue
        if best is None or score > best[2]:
            best = (index, name, score)

    if best is None:
        return None
    return (best[0], best[1])


def _default_like_output_devices() -> list[int]:
    try:
        devices = sd.query_devices()
    except Exception as exc:
        log.warning("AudioPlayer: could not query default-like output devices: %s", exc)
        return []

    scored: list[tuple[int, int]] = []
    for index, info in enumerate(devices):
        if int(info.get("max_output_channels", 0)) <= 0:
            continue
        name = str(info.get("name", "")).lower()
        score = 0
        if "pipewire" in name:
            score = 5
        elif "pulse" in name:
            score = 4
        elif name == "default":
            score = 3
        elif "default" in name:
            score = 2
        elif "sysdefault" in name:
            score = 1
        if score > 0:
            scored.append((index, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [index for index, _ in scored]


def _expand_output_device_choice(device: int | None) -> list[int | None]:
    if device is not None:
        return [device]
    expanded: list[int | None] = []
    expanded.extend(_default_like_output_devices())
    expanded.append(None)
    return expanded


def _device_label(device: int | None) -> str:
    if device is None:
        return "system default"
    try:
        info = sd.query_devices(device, kind="output")
        return f"{device} ({info['name']})"
    except Exception:
        return str(device)


def _device_default_samplerate(device: int | None) -> int | None:
    try:
        info = sd.query_devices(device, kind="output")
    except Exception:
        return None
    default_sr = float(info.get("default_samplerate", 0) or 0)
    if default_sr <= 0:
        return None
    return int(round(default_sr))


def _candidate_output_sample_rates(
    device: int | None,
    *,
    requested_samplerate: int,
) -> list[int]:
    rates: list[int] = [requested_samplerate]
    default_sr = _device_default_samplerate(device)
    if default_sr is not None and default_sr not in rates:
        rates.append(default_sr)
    for rate in (48000, 44100, 32000, 16000):
        if rate not in rates:
            rates.append(rate)
    return rates


def _prepare_speech_chunk_for_output(
    chunk: _SpeechChunk,
    target_sample_rate: int,
) -> np.ndarray:
    if chunk.sample_rate == target_sample_rate:
        return chunk.samples.astype(np.float32, copy=False)
    return _resample_audio_array(
        chunk.samples.astype(np.float32, copy=False),
        chunk.sample_rate,
        target_sample_rate,
        axis=0,
    )


def _resample_audio_array(
    data: np.ndarray,
    src_sr: int,
    dst_sr: int,
    *,
    axis: int = 0,
) -> np.ndarray:
    if src_sr == dst_sr:
        return data.astype(np.float32, copy=False)
    try:
        from math import gcd
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError(
            "Audio resampling requires scipy; install scipy to use Bluetooth/variable-rate output"
        ) from exc
    g = gcd(src_sr, dst_sr)
    up, down = dst_sr // g, src_sr // g
    return resample_poly(data, up, down, axis=axis).astype(np.float32)
