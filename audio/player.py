"""
audio/player.py — Audio playback for DJ-R3X.

Two independent output streams:
  - Speech stream: persistent callback-based OutputStream. Routes both
    streaming TTS chunks and cached audio files. Computes real-time RMS
    on every audio callback so leds.py can drive mouth brightness.
  - Music stream: explicit OutputStream in a background thread. No RMS
    tracking. Supports loop and clean stop.

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

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ElevenLabs is asked to produce PCM at this rate (see synthesizer.py).
# Must be a rate supported by the output device; 22050 is universally safe.
SPEECH_SAMPLE_RATE: int = 22050

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

        # --- speech stream state ---
        self._speech_queue: queue.SimpleQueue[np.ndarray | _EndMarker] = (
            queue.SimpleQueue()
        )
        self._speech_buf: np.ndarray | None = None
        self._speech_buf_pos: int = 0
        self._speech_active = threading.Event()
        self._speech_active.set()        # starts "idle"

        # --- music stream state ---
        self._music_thread: threading.Thread | None = None
        self._music_stop = threading.Event()

        # --- open the persistent speech output stream ---
        self._speech_stream = sd.OutputStream(
            samplerate=SPEECH_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=config.AUDIO_OUTPUT_DEVICE,
            blocksize=config.AUDIO_CHUNK_SIZE,
            callback=self._speech_callback,
        )
        self._speech_stream.start()

    # ------------------------------------------------------------------
    # Speech — streaming TTS interface (called by synthesizer.py)
    # ------------------------------------------------------------------

    def feed_speech_chunk(self, pcm_bytes: bytes) -> None:
        """Push one raw PCM int16 chunk from the ElevenLabs stream into the
        playback queue. The chunk is played in order with any queued chunks."""
        if not pcm_bytes:
            return
        self._speech_queue.put(np.frombuffer(pcm_bytes, dtype=np.int16).copy())

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
        # float32 → int16 for the speech stream
        samples = np.clip(data * 32767.0, -32768, 32767).astype(np.int16)

        done = threading.Event()
        self._speech_active.clear()

        for i in range(0, len(samples), _FILE_CHUNK_FRAMES):
            self._speech_queue.put(samples[i : i + _FILE_CHUNK_FRAMES])
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
        self._speech_buf_pos = 0
        self._rms = 0.0
        self._speech_active.set()   # unblock any wait_for_speech() caller

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
        import subprocess
        import io
        path = Path(config.STARTUP_CHIME_PATH)
        if not path.exists():
            return
        try:
            result = subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "wav", "pipe:1"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return
        data, sr = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=False)
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
        Updated on every audio callback (~23 ms at 22050 Hz / 1024 frames).
        Returns 0.0 when nothing is playing."""
        return self._rms

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop all playback and release PortAudio resources."""
        self.stop_speech()
        self.stop_music()
        self._speech_stream.stop()
        self._speech_stream.close()

    # ------------------------------------------------------------------
    # Internal — speech callback (sounddevice audio thread)
    # ------------------------------------------------------------------

    def _speech_callback(
        self,
        outdata: np.ndarray,    # shape (blocksize, 1), dtype int16
        frames: int,
        _time,                  # CffiData timestamp — unused
        _status: sd.CallbackFlags,
    ) -> None:
        output = outdata[:, 0]  # flat view into the mono channel
        filled = 0

        while filled < frames:
            # refill internal chunk buffer from queue when exhausted
            if self._speech_buf is None or self._speech_buf_pos >= len(self._speech_buf):
                try:
                    item = self._speech_queue.get_nowait()
                except queue.Empty:
                    break   # nothing queued — output silence below

                if isinstance(item, _EndMarker):
                    self._speech_buf = None
                    self._speech_buf_pos = 0
                    self._speech_active.set()
                    if item.done is not None:
                        item.done.set()
                    break
                # item is an np.ndarray of int16 samples
                self._speech_buf = item
                self._speech_buf_pos = 0

            take = min(
                frames - filled,
                len(self._speech_buf) - self._speech_buf_pos,
            )
            output[filled : filled + take] = (
                self._speech_buf[self._speech_buf_pos : self._speech_buf_pos + take]
            )
            self._speech_buf_pos += take
            filled += take

        # zero-pad any unfilled frames (silence when queue is empty)
        if filled < frames:
            output[filled:] = 0

        # --- real-time RMS for mouth LED ---
        # only compute on frames that actually contain audio, not padding
        if filled > 0:
            chunk_f32 = output[:filled].astype(np.float32)
            rms_raw = float(np.sqrt(np.mean(chunk_f32 ** 2)))
            # int16 max = 32767; scale to 0-1, apply gain, map to 0-255
            brightness = min(255.0, (rms_raw / 32767.0) * config.MOUTH_LED_GAIN * 255.0)
        else:
            brightness = 0.0

        # exponential smoothing to prevent harsh LED flicker
        alpha = config.MOUTH_LED_SMOOTHING
        self._rms = alpha * brightness + (1.0 - alpha) * self._rms

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
        is fully independent from both the speech stream and sd.play()."""
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
                if self._music_stop.is_set():
                    outdata[:] = 0.0
                    raise sd.CallbackStop()

                remaining = len(data) - pos
                take = min(frames, remaining)
                outdata[:take] = data[pos : pos + take]
                if take < frames:
                    outdata[take:] = 0.0
                pos += take

                if pos >= len(data):
                    raise sd.CallbackStop()

            with sd.OutputStream(
                samplerate=sr,
                channels=channels,
                dtype="float32",
                device=config.AUDIO_OUTPUT_DEVICE,
                callback=_callback,
                finished_callback=finished.set,
            ):
                finished.wait()

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
        import subprocess
        import io
        result = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "wav", "pipe:1"],
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
