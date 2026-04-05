"""
hardware/servos.py — Pololu Maestro Mini servo control for DJ-R3X.

Protocol
--------
Uses the Pololu compact binary protocol over USB serial (/dev/ttyACM0).
All position values are in quarter-microseconds (qµs):
    6000 qµs = 1500 µs = servo neutral / center
    4000 qµs = 1000 µs = one mechanical extreme
    8000 qµs = 2000 µs = other mechanical extreme

Thread model
------------
  Background thread (_idle_thread)
    Continuously moves the arm and hand channels (3-6) to random positions
    within the emotion-biased range, with timing from
    SERVO_IDLE_MOVE_INTERVAL_MIN / MAX.  Does NOT touch head or visor.

  Main / state-machine thread
    Calls set_emotion(), speak_move(), home(), start(), stop().

  speak_move()
    Moves head tilt, head pan, and visor (channels 0-2) to positions
    derived from a 0-1 audio intensity value.  Uses the same _lock as the
    background thread so serial writes are never interleaved.

Usage
-----
    s = ServoController()
    s.start()
    s.set_emotion("excited")
    s.speak_move(intensity=0.8)   # during TTS
    s.home()
    s.stop()
    s.close()
"""

from __future__ import annotations

import logging
import random
import threading
import time

import serial

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pololu compact protocol byte codes
# ---------------------------------------------------------------------------

_CMD_SET_TARGET = 0x84   # Set Target:       0x84 ch lo hi
_CMD_SET_SPEED  = 0x87   # Set Speed:        0x87 ch lo hi
_CMD_SET_ACCEL  = 0x89   # Set Acceleration: 0x89 ch lo hi

# Channels moved by the background idle thread (arms only)
_IDLE_CHANNELS = config.ARM_CHANNELS

# Channels moved during speech (head group)
_SPEECH_CHANNELS = config.HEAD_CHANNELS

# All managed channels in channel-number order
_ALL_CHANNELS = sorted(config.SERVO_CHANNELS.keys())


# ---------------------------------------------------------------------------
# Protocol helpers (module-level, no state)
# ---------------------------------------------------------------------------

def _encode(cmd: int, channel: int, value: int) -> bytes:
    """Encode a 4-byte Pololu compact protocol command.

    value is split into two 7-bit bytes (Pololu's LSB-first encoding):
        low  = value & 0x7F
        high = (value >> 7) & 0x7F
    """
    return bytes([cmd, channel, value & 0x7F, (value >> 7) & 0x7F])


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# ServoController
# ---------------------------------------------------------------------------

class ServoController:
    """Drives the Pololu Maestro Mini for DJ-R3X.

    Not safe to call start()/stop()/set_emotion()/speak_move() concurrently
    from multiple threads — use from the state machine thread only.
    speak_move() may also be called from a TTS/audio thread; the internal
    _lock makes serial writes thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._emotion: str = "neutral"
        self._stop_event = threading.Event()
        self._idle_thread: threading.Thread | None = None

        log.info("Opening Maestro serial port %s @ %d baud",
                 config.MAESTRO_PORT, config.MAESTRO_BAUD)
        self._serial = serial.Serial(
            config.MAESTRO_PORT,
            config.MAESTRO_BAUD,
            timeout=1,
        )

        # Apply per-channel acceleration, default speed, then move to home
        self._apply_acceleration()
        self._apply_speed(config.SERVO_DEFAULT_SPEED)
        self.home()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background idle-motion thread (arms + hands)."""
        if self._idle_thread is not None and self._idle_thread.is_alive():
            log.warning("ServoController.start() called while already running")
            return
        self._stop_event.clear()
        self._idle_thread = threading.Thread(
            target=self._idle_loop,
            daemon=True,
            name="djr3x-servos",
        )
        self._idle_thread.start()
        log.info("Servo idle-motion thread started.")

    def stop(self) -> None:
        """Stop the background idle-motion thread and wait for it to exit."""
        self._stop_event.set()
        if self._idle_thread is not None:
            self._idle_thread.join(timeout=5.0)
            if self._idle_thread.is_alive():
                log.warning("Servo idle thread did not stop cleanly within 5 s")
            self._idle_thread = None
        log.info("Servo idle-motion thread stopped.")

    def close(self) -> None:
        """Stop motion, move to home, and close the serial port."""
        self.stop()
        self.home()
        self._serial.close()
        log.info("Maestro serial port closed.")

    # ------------------------------------------------------------------
    # Emotion
    # ------------------------------------------------------------------

    def set_emotion(self, emotion: str) -> None:
        """Set the current emotion, updating speed and position ranges.

        Valid emotions: "neutral", "excited", "sad".
        Unknown emotion names fall back to "neutral" with a warning.
        """
        if emotion not in config.SERVO_EMOTION_LIMITS:
            log.warning("Unknown emotion %r — falling back to neutral", emotion)
            emotion = "neutral"

        self._emotion = emotion

        speed = {
            "excited": config.SERVO_EXCITED_SPEED,
            "sad":     config.SERVO_SAD_SPEED,
        }.get(emotion, config.SERVO_DEFAULT_SPEED)

        self._apply_speed(speed)
        log.debug("Emotion set to %r (speed=%d)", emotion, speed)

    # ------------------------------------------------------------------
    # Speech-reactive movement (called during TTS playback)
    # ------------------------------------------------------------------

    def speak_move(self, intensity: float = 0.5) -> None:
        """Move head and visor to speech-reactive positions.

        intensity: 0.0 (quiet) → 1.0 (loud), typically derived from
                   AudioPlayer.rms scaled to 0-1.  Maps to neck rotation
                   (up = loud) and visor open amount.  A small random
                   jitter keeps the motion from looking mechanical.

        Thread-safe: acquires _lock, safe to call from audio/TTS thread.
        """
        intensity = max(0.0, min(1.0, float(intensity)))

        neck_lo,  neck_hi  = self._effective_limits(0)   # neck
        lift_lo,  lift_hi  = self._effective_limits(1)   # headlift
        visor_lo, visor_hi = self._effective_limits(3)   # visor

        # neck moves up on louder speech; small random jitter keeps it lively
        jitter = random.randint(-80, 80)
        neck_pos = _clamp(
            int(neck_lo + intensity * (neck_hi - neck_lo)) + jitter,
            neck_lo, neck_hi,
        )

        # headlift wobbles gently around center
        lift_center = (lift_lo + lift_hi) // 2
        lift_spread = int((lift_hi - lift_lo) * 0.15)
        lift_pos = _clamp(
            lift_center + random.randint(-lift_spread, lift_spread),
            lift_lo, lift_hi,
        )

        # visor opens with intensity
        visor_pos = _clamp(
            int(visor_lo + intensity * (visor_hi - visor_lo)),
            visor_lo, visor_hi,
        )

        with self._lock:
            self._send_target(0, neck_pos)
            self._send_target(1, lift_pos)
            self._send_target(3, visor_pos)

    # ------------------------------------------------------------------
    # Safe home position
    # ------------------------------------------------------------------

    def home(self) -> None:
        """Move all servos to their neutral home positions.

        In SERVO_SAFE_MODE, moves one channel at a time with a slow speed
        command before each move and a 0.5 s delay between channels.
        In normal mode, sends all targets simultaneously.

        Call on startup (done by __init__) and on shutdown.
        """
        if config.SERVO_SAFE_MODE:
            log.info("Safe-mode homing: moving channels one at a time.")
            for channel, cfg in config.SERVO_CHANNELS.items():
                with self._lock:
                    self._send_speed(channel, config.SERVO_STARTUP_SPEED)
                    self._send_target(channel, cfg["neutral"])
                log.debug("Homed channel %d (%s) at startup speed %d",
                          channel, cfg["name"], config.SERVO_STARTUP_SPEED)
                time.sleep(0.5)
        else:
            with self._lock:
                for channel, cfg in config.SERVO_CHANNELS.items():
                    self._send_target(channel, cfg["neutral"])
        log.debug("All servos moved to home positions.")

    # ------------------------------------------------------------------
    # Direct position command (public, for sequences layer)
    # ------------------------------------------------------------------

    def set_position(self, channel: int, position: int) -> None:
        """Move a single channel to position (qµs), clamped to its limits.

        Used by sequences/animations.py for scripted keyframe moves.
        Thread-safe.
        """
        lo, hi = self._effective_limits(channel)
        position = _clamp(position, lo, hi)
        with self._lock:
            self._send_target(channel, position)

    # ------------------------------------------------------------------
    # Internal — background idle-motion thread
    # ------------------------------------------------------------------

    def _idle_loop(self) -> None:
        """Continuously move arm and hand servos to random positions.

        Each iteration picks ONE channel to move (natural-looking staggered
        motion) then sleeps for a random interval from config.
        """
        while not self._stop_event.is_set():
            channel = random.choice(_IDLE_CHANNELS)
            lo, hi = self._effective_limits(channel)
            target = random.randint(lo, hi)

            with self._lock:
                self._send_target(channel, target)

            interval = random.uniform(
                config.SERVO_IDLE_MOVE_INTERVAL_MIN,
                config.SERVO_IDLE_MOVE_INTERVAL_MAX,
            )
            # Sleep in small increments so stop_event is checked promptly
            deadline = time.monotonic() + interval
            while not self._stop_event.is_set() and time.monotonic() < deadline:
                time.sleep(0.05)

    # ------------------------------------------------------------------
    # Internal — serial helpers
    # ------------------------------------------------------------------

    def _send_target(self, channel: int, position: int) -> None:
        """Write a Set Target command.  Caller must hold _lock."""
        self._serial.write(_encode(_CMD_SET_TARGET, channel, position))

    def _send_speed(self, channel: int, speed: int) -> None:
        """Write a Set Speed command.  Caller must hold _lock."""
        self._serial.write(_encode(_CMD_SET_SPEED, channel, speed))

    def _apply_speed(self, speed: int) -> None:
        """Set the same speed on every managed channel."""
        with self._lock:
            for channel in _ALL_CHANNELS:
                self._send_speed(channel, speed)

    def _apply_acceleration(self) -> None:
        """Set per-channel acceleration from SERVO_CHANNELS config."""
        with self._lock:
            for channel, cfg in config.SERVO_CHANNELS.items():
                self._serial.write(
                    _encode(_CMD_SET_ACCEL, channel, cfg["acceleration"])
                )

    # ------------------------------------------------------------------
    # Internal — effective range lookup
    # ------------------------------------------------------------------

    def _effective_limits(self, channel: int) -> tuple[int, int]:
        """Return (min, max) qµs for channel, with emotion overrides applied."""
        ch_cfg = config.SERVO_CHANNELS[channel]
        base = (ch_cfg["min"], ch_cfg["max"])
        overrides = config.SERVO_EMOTION_LIMITS.get(self._emotion, {})
        return overrides.get(channel, base)
