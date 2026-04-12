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
import math
import random
import threading
import time
from typing import Callable

import serial

import config

log = logging.getLogger(__name__)
_SERIAL_ERRORS = (serial.SerialException, serial.SerialTimeoutException)

# ---------------------------------------------------------------------------
# Pololu compact protocol byte codes
# ---------------------------------------------------------------------------

_CMD_SET_TARGET = 0x84   # Set Target:       0x84 ch lo hi
_CMD_SET_SPEED  = 0x87   # Set Speed:        0x87 ch lo hi
_CMD_SET_ACCEL  = 0x89   # Set Acceleration: 0x89 ch lo hi
_CMD_GO_HOME    = 0xA2   # Go Home (compact, single byte — no parameters)

# Channels moved by the background idle thread (arms only)
_IDLE_CHANNELS = config.ARM_CHANNELS

# Channels moved during speech (head group)
_SPEECH_CHANNELS = config.HEAD_CHANNELS

# Arm channels animated during speech (elbow, hand, heroarm)
_SPEAK_ARM_CHANNELS = (
    config.SERVO_ARM_LEFT,   # ch 4 — elbow
    config.SERVO_HAND_LEFT,  # ch 5 — hand
    config.SERVO_HAND_RIGHT, # ch 7 — heroarm
)

# Arms amplify audio intensity relative to head so gestures read bigger
_ARM_SPEAK_INTENSITY_MULT: float = 2.0

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

    def __init__(self, serial_factory: Callable[..., serial.Serial] = serial.Serial) -> None:
        self._lock = threading.Lock()
        self._serial_factory = serial_factory
        self._emotion: str = "neutral"
        self._current_speed: int = config.SERVO_DEFAULT_SPEED
        self._stop_event = threading.Event()
        self._idle_thread: threading.Thread | None = None
        self._dance_stop_event = threading.Event()
        self._dance_thread: threading.Thread | None = None
        # Throttle counters for slow-moving arm channels in speak_move().
        # ch 5 (hand): update every 4 calls (~200 ms) — servo needs time to
        #   complete each full-range twist before receiving a new target.
        # ch 4 (elbow): update every SERVO_ELBOW_SPEAK_THROTTLE calls so it
        #   makes slow deliberate raises rather than rapid small jitters.
        self._hand_speak_counter: int = 0
        self._elbow_speak_counter: int = 0
        # Set by pause_arm_idle() to prevent the idle loop from sending conflicting
        # commands to ch 4/5/7 while an arm animation is running.
        self._arm_idle_pause = threading.Event()

        self._serial = self._open_maestro_serial()

        # Apply per-channel acceleration, then snap channels to their slumped
        # starting positions at speed=0 (instant, no visible movement).
        # Rex is physically in this pose after shutdown; this just tells the
        # Maestro where to start from so the STARTUP animation takes over cleanly.
        self._apply_acceleration()
        self._set_initial_positions()
        self._apply_speed(config.SERVO_DEFAULT_SPEED)

    # ------------------------------------------------------------------
    # Serial open with retry
    # ------------------------------------------------------------------

    def _open_maestro_serial(self) -> serial.Serial:
        """Open the Maestro serial port, retrying up to SERIAL_RETRY_ATTEMPTS times.

        Raises serial.SerialException if all attempts fail.
        After a successful open, waits MAESTRO_STARTUP_DELAY seconds so the
        Maestro has time to initialise its USB stack before receiving commands.
        """
        port = config.MAESTRO_PORT
        if not port:  # None or empty string → not configured
            raise serial.SerialException("MAESTRO_PORT is not configured")
        baud = config.MAESTRO_BAUD
        for attempt in range(1, config.SERIAL_RETRY_ATTEMPTS + 1):
            try:
                log.info(
                    "Opening Maestro serial port %s @ %d baud (attempt %d/%d)",
                    port, baud, attempt, config.SERIAL_RETRY_ATTEMPTS,
                )
                ser = self._serial_factory(port, baud, timeout=1)
                log.info(
                    "Maestro port opened on attempt %d — waiting %d s for initialisation",
                    attempt, config.MAESTRO_STARTUP_DELAY,
                )
                time.sleep(config.MAESTRO_STARTUP_DELAY)
                return ser
            except _SERIAL_ERRORS as exc:
                log.warning(
                    "Maestro serial open failed (attempt %d/%d): %s",
                    attempt, config.SERIAL_RETRY_ATTEMPTS, exc,
                )
                if attempt < config.SERIAL_RETRY_ATTEMPTS:
                    log.info(
                        "Retrying Maestro in %.0f s …", config.SERIAL_RETRY_DELAY
                    )
                    time.sleep(config.SERIAL_RETRY_DELAY)
        raise serial.SerialException(
            f"Could not open Maestro port {port} after "
            f"{config.SERIAL_RETRY_ATTEMPTS} attempts"
        )

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
        """Stop motion, power off servos, and close the serial port.

        Does NOT call home() — the shutdown animation in play_shutdown() already
        moves channels to their final slumped pose and calling home() here would
        override that with neutral positions.
        """
        self.stop()
        # Wait for servos to settle in their current (post-animation) positions
        # before cutting PWM output.
        time.sleep(1.0)
        self.power_off()
        self._close_serial(self._serial)
        log.info("Maestro serial port closed.")

    def power_off(self) -> None:
        """Disable PWM output on all channels so servos go limp.

        Sequence:
          1. Send the Maestro compact Go Home command (0xA2) — tells the
             controller to move channels to their stored home positions, which
             also confirms the serial link is alive.
          2. Send Set Target = 0 for every channel.  A target of 0 is the
             Pololu protocol signal to stop emitting PWM pulses; the servo
             de-energises and goes limp immediately.
          3. Flush the OS serial buffer and wait briefly so the Maestro has
             time to process the commands before the port is closed.
        """
        with self._lock:
            # Step 1 — Go Home (0xA2, compact protocol, no parameters)
            log.info("power_off: sending Go Home (0xA2)")
            self._write_command_locked(bytes([_CMD_GO_HOME]))
            self._serial.flush()
            time.sleep(0.1)   # let the Maestro act on Go Home before disabling

            # Step 2 — disable every channel (target = 0 → no PWM pulse)
            for channel in _ALL_CHANNELS:
                raw = _encode(_CMD_SET_TARGET, channel, 0)
                log.info(
                    "power_off: ch %d (%s) → target 0  bytes=%s",
                    channel,
                    config.SERVO_CHANNELS.get(channel, {}).get("name", "?"),
                    raw.hex(),
                )
                self._write_command_locked(raw)

            # Step 3 — flush and wait for Maestro to process
            self._serial.flush()

        time.sleep(0.2)   # allow Maestro to act before port closes
        log.info("Servo power off complete — all channels PWM disabled.")

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

        self._current_speed = speed
        self._apply_speed(speed)
        log.debug("Emotion set to %r (speed=%d)", emotion, speed)

    # ------------------------------------------------------------------
    # Speech-reactive movement (called during TTS playback)
    # ------------------------------------------------------------------

    def speak_move(self, intensity: float = 0.5) -> None:
        """Move head, visor, and expressive arm channels to speech-reactive positions.

        intensity: 0.0 (quiet) → 1.0 (loud), typically derived from
                   AudioPlayer.rms scaled to 0-1.  Maps to neck rotation
                   (up = loud), visor open amount, and arm gesture range.
                   A small random jitter keeps the motion from looking mechanical.

        Arms use _ARM_SPEAK_INTENSITY_MULT so they gesture more aggressively
        than the head — wider swings, more responsive to audio level.

        Thread-safe: acquires _lock, safe to call from audio/TTS thread.
        """
        intensity = max(0.0, min(1.0, float(intensity)))

        neck_lo,  neck_hi  = self._effective_limits(0)   # neck
        lift_lo,  lift_hi  = self._effective_limits(1)   # headlift
        tilt_lo,  tilt_hi  = self._effective_limits(2)   # headtilt
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

        # headtilt rises with intensity (inverted: lower qµs = head up).
        # Constrain to [min, neutral] — can tilt up from neutral but never
        # goes below neutral (which would point the head down).
        tilt_neutral = config.SERVO_CHANNELS[2]["neutral"]
        tilt_ceiling = tilt_neutral   # never go below neutral
        tilt_pos = _clamp(
            int(tilt_ceiling - intensity * (tilt_ceiling - tilt_lo)),
            tilt_lo, tilt_ceiling,
        )

        # visor opens (rises) with intensity — higher qµs = open/revealed.
        # Anchors at ~5800 (slightly below neutral) when silent, rises toward
        # max (6976) at full intensity.  Uses a fixed speak base rather than
        # emotion-derived visor_lo so silent speech never closes the visor.
        _VISOR_SPEAK_BASE = 5800
        visor_pos = _clamp(
            int(_VISOR_SPEAK_BASE + intensity * (visor_hi - _VISOR_SPEAK_BASE)),
            visor_lo, visor_hi,
        )

        # --- Arm gestures ---
        arm_intensity = min(1.0, intensity * _ARM_SPEAK_INTENSITY_MULT)

        elbow_lo, elbow_hi = self._effective_limits(config.SERVO_ARM_LEFT)   # ch 4
        hand_lo,  hand_hi  = self._effective_limits(config.SERVO_HAND_LEFT)  # ch 5
        hero_lo,  hero_hi  = self._effective_limits(config.SERVO_HAND_RIGHT) # ch 7

        # Elbow (ch 4): update every SERVO_ELBOW_SPEAK_THROTTLE calls so the servo
        # has time to complete each raise/lower before a new target arrives.
        # Alternates between the low and high end of at least 50% of its span,
        # scaled by intensity — produces slow deliberate raises instead of jitter.
        self._elbow_speak_counter += 1
        elbow_target: int | None = None
        if self._elbow_speak_counter % config.SERVO_ELBOW_SPEAK_THROTTLE == 0:
            elbow_center    = (elbow_lo + elbow_hi) // 2
            # Amplitude: 50–75% of span, growing with intensity
            elbow_amplitude = int((elbow_hi - elbow_lo) * (0.50 + 0.25 * intensity))
            if (self._elbow_speak_counter // config.SERVO_ELBOW_SPEAK_THROTTLE) % 2 == 0:
                elbow_target = _clamp(elbow_center - elbow_amplitude, elbow_lo, elbow_hi)
            else:
                elbow_target = _clamp(elbow_center + elbow_amplitude, elbow_lo, elbow_hi)

        # Hand (ch 5): only update every 4th call (~200 ms) so the servo can
        # complete each twist before receiving a new target.
        # Alternates between low and high extremes; amplitude is intentionally
        # small (15–30% of range) so the movement is subtle during speech.
        self._hand_speak_counter += 1
        hand_target: int | None = None
        if self._hand_speak_counter % 4 == 0:
            hand_center = (hand_lo + hand_hi) // 2
            amplitude   = int((hand_hi - hand_lo) * (0.15 + 0.15 * arm_intensity))
            if (self._hand_speak_counter // 4) % 2 == 0:
                hand_target = _clamp(hand_center - amplitude, hand_lo, hand_hi)
            else:
                hand_target = _clamp(hand_center + amplitude, hand_lo, hand_hi)

        # Heroarm gestures independently — random sweep scaled with intensity
        hero_center = (hero_lo + hero_hi) // 2
        hero_swing  = int((hero_hi - hero_lo) * 0.40 * arm_intensity) + 80
        hero_pos = _clamp(
            hero_center + random.randint(-hero_swing, hero_swing),
            hero_lo, hero_hi,
        )

        with self._lock:
            # Restore emotion speed on head/visor — idle loop may have slowed them
            for ch in (0, 1, 2, 3):
                self._send_speed(ch, self._current_speed)
            self._send_target(0, neck_pos)
            self._send_target(1, lift_pos)
            self._send_target(2, tilt_pos)
            self._send_target(3, visor_pos)
            # Elbow at default speed so each raise/lower is slow and deliberate.
            # Hand at relaxed speed — subtle, unhurried twist during speech.
            # Heroarm at excited speed for snappy gestures.
            self._send_speed(config.SERVO_ARM_LEFT,   config.SERVO_DEFAULT_SPEED)
            self._send_speed(config.SERVO_HAND_RIGHT, config.SERVO_EXCITED_SPEED)
            self._send_speed(config.SERVO_HAND_LEFT,  config.SERVO_HAND_SPEAK_SPEED_RELAXED)
            if elbow_target is not None:
                self._send_target(config.SERVO_ARM_LEFT, elbow_target)
            if hand_target is not None:
                self._send_target(config.SERVO_HAND_LEFT, hand_target)
            self._send_target(config.SERVO_HAND_RIGHT, hero_pos)

    # ------------------------------------------------------------------
    # Startup initialisation
    # ------------------------------------------------------------------

    def _set_initial_positions(self) -> None:
        """Snap all channels to their startup/slumped positions at speed=0.

        Channels listed in config.SERVO_SLUMPED_POSITIONS receive their
        slumped values; all others receive their neutral positions.
        Speed=0 tells the Maestro to jump to the target instantly with no
        ramp — the servos should already be physically there from the
        previous shutdown animation, so no visible movement occurs.
        """
        with self._lock:
            for channel in _ALL_CHANNELS:
                self._send_speed(channel, 0)
            for channel, cfg in config.SERVO_CHANNELS.items():
                position = config.SERVO_SLUMPED_POSITIONS.get(channel, cfg["neutral"])
                self._send_target(channel, position)
        log.debug("Initial positions set (slumped pose, speed=0).")

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

    def set_channel_speed(self, channel: int, speed: int) -> None:
        """Set the Maestro move speed for a single channel.

        Used by sequences/animations.py when a specific channel needs a
        non-default speed before a scripted sequence (e.g. the hand wave).
        Thread-safe.
        """
        with self._lock:
            self._send_speed(channel, speed)

    def pause_arm_idle(self) -> None:
        """Prevent the idle loop from moving expressive arm channels (ch 4, 5, 7).

        Call before starting an arm animation so the idle thread does not
        fight the animation with conflicting serial commands to those channels.
        """
        self._arm_idle_pause.set()
        log.debug("Arm idle paused (ch 4/5/7 suppressed during animation)")

    def resume_arm_idle(self) -> None:
        """Allow the idle loop to resume moving expressive arm channels."""
        self._arm_idle_pause.clear()
        log.debug("Arm idle resumed")

    # ------------------------------------------------------------------
    # Dance mode
    # ------------------------------------------------------------------

    @property
    def is_dancing(self) -> bool:
        """True while the dance loop thread is actively running."""
        return self._dance_thread is not None and self._dance_thread.is_alive()

    def start_dancing(self) -> None:
        """Start the background dance-loop thread.

        No-op (returns immediately) if already dancing.
        The dance loop runs at 20 Hz, sweeping all servo channels through
        their full range with staggered sine-wave phases so they move
        independently rather than in lockstep.
        """
        if self.is_dancing:
            log.info("start_dancing: already dancing — ignoring")
            return
        self._dance_stop_event.clear()
        self._dance_thread = threading.Thread(
            target=self._dance_loop,
            daemon=True,
            name="djr3x-dance",
        )
        self._dance_thread.start()
        log.info("Dance thread started.")

    def stop_dancing(self) -> None:
        """Stop the dance-loop thread and return all servos to neutral.

        Blocks until the dance thread exits (up to 3 s).
        """
        self._dance_stop_event.set()
        if self._dance_thread is not None:
            self._dance_thread.join(timeout=3.0)
            if self._dance_thread.is_alive():
                log.warning("Dance thread did not stop cleanly within 3 s")
            self._dance_thread = None
        # Return all channels to neutral at a moderate speed.
        with self._lock:
            for channel, cfg in config.SERVO_CHANNELS.items():
                self._send_speed(channel, config.SERVO_DEFAULT_SPEED)
                self._send_target(channel, cfg["neutral"])
        log.info("Dance stopped — servos returned to neutral.")

    # (channel, frequency_multiplier, phase_offset_radians, lo_pct, hi_pct)
    # lo_pct/hi_pct limit each channel to a fraction of its configured range.
    # Default is 0.0–1.0 (full range); tighten individual channels as needed.
    _DANCE_CHANNELS: tuple[tuple[int, float, float, float, float], ...] = (
        (0, 1.00, 0.00, 0.0, 1.0),   # neck       — full range
        (1, 0.73, 0.20, 0.25, 0.75), # headlift   — 25–75 % (avoid extremes)
        (2, 1.31, 0.10, 0.0, 1.0),   # headtilt   — full range
        (3, 0.61, 0.30, 0.0, 1.0),   # visor      — full range
        (4, 1.17, 0.15, 0.0, 1.0),   # elbow      — full range
        (5, 0.83, 0.25, 0.0, 1.0),   # hand       — full range
        (6, 1.43, 0.05, 0.0, 1.0),   # pokerarm   — full range
        (7, 0.91, 0.35, 0.0, 1.0),   # heroarm    — full range
    )
    _DANCE_SPEED = 50          # slightly faster than excited (40); keeps motion smooth
    _DANCE_HZ    = 10          # update rate (Hz) — lower rate suits the gentler speed
    _DANCE_FREQ  = math.pi     # base rad/s → one full sweep every ~2 s

    def _dance_loop(self) -> None:
        """Drive all servo channels through sine-wave patterns at 20 Hz.

        Each channel has its own frequency multiplier so they drift in and out
        of sync, producing a fluid whole-body dancing motion.  All phase offsets
        are near 0 so every channel starts moving at the same time.

        Speed is set high enough that every servo reaches its new target within
        a single 50 ms update interval, so all channels always move in parallel.
        """
        dt = 1.0 / self._DANCE_HZ
        t  = 0.0

        # Set dance speed for all channels and pre-position them at t=0 targets
        # so the loop begins with all servos already in their start positions.
        with self._lock:
            for ch, freq_mult, phase, lo_pct, hi_pct in self._DANCE_CHANNELS:
                self._send_speed(ch, self._DANCE_SPEED)
            for ch, freq_mult, phase, lo_pct, hi_pct in self._DANCE_CHANNELS:
                ch_cfg = config.SERVO_CHANNELS[ch]
                full_lo = ch_cfg["min"]
                full_hi = ch_cfg["max"]
                span = full_hi - full_lo
                lo = full_lo + int(lo_pct * span)
                hi = full_lo + int(hi_pct * span)
                val = math.sin(0.0 * self._DANCE_FREQ * freq_mult + phase)
                pos = _clamp(int(lo + (val + 1.0) / 2.0 * (hi - lo)), lo, hi)
                self._send_target(ch, pos)

        # Wait for all servos to reach their t=0 positions before the loop.
        # At speed 50 the largest travel (~4000 qµs) takes about 0.8 s.
        time.sleep(1.0)

        while not self._dance_stop_event.is_set():
            with self._lock:
                for ch, freq_mult, phase, lo_pct, hi_pct in self._DANCE_CHANNELS:
                    ch_cfg = config.SERVO_CHANNELS[ch]
                    full_lo = ch_cfg["min"]
                    full_hi = ch_cfg["max"]
                    span = full_hi - full_lo
                    lo = full_lo + int(lo_pct * span)
                    hi = full_lo + int(hi_pct * span)
                    val = math.sin(t * self._DANCE_FREQ * freq_mult + phase)
                    pos = _clamp(int(lo + (val + 1.0) / 2.0 * (hi - lo)), lo, hi)
                    self._send_target(ch, pos)
            t += dt
            time.sleep(dt)

    def set_position(self, channel: int, position: int) -> None:
        """Move a single channel to position (qµs), clamped to its limits.

        Used by sequences/animations.py for scripted keyframe moves.
        Thread-safe.
        """
        lo, hi = self._effective_limits(channel)
        position = _clamp(position, lo, hi)
        if channel == config.SERVO_HAND_LEFT:
            log.debug(
                "set_position: ch 5 (hand) → %d qµs  (config limits %d–%d)",
                position, lo, hi,
            )
        with self._lock:
            self._send_target(channel, position)

    # ------------------------------------------------------------------
    # Internal — background idle-motion thread
    # ------------------------------------------------------------------

    def _idle_loop(self) -> None:
        """Continuously move servos to random positions during idle.

        All timers are independent; the loop polls every 50 ms.

        Head (neck ch 0, headlift ch 1): one channel every 3-5 s, slow speed.
            Headlift range is biased toward neutral (level) so Rex doesn't
            spend too much time looking fully upward.
        Headtilt (ch 2): drifts toward IDLE_HEAD_TILT_REST every 4-7 s at
            very slow speed — produces a gradual, relaxed resting pose.
        Visor (ch 3): every 5-8 s, very slow speed,
            constrained to the open (low) 30% of its range.
        Elbow (ch 4): drifts toward IDLE_ELBOW_REST every 4-7 s at very slow
            speed — lowered resting pose; rises at speech start via _begin_speech().
        Expressive arms (ch 5 hand, ch 7 heroarm): one channel every
            ARM_IDLE_INTERVAL_MIN–MAX s, constrained to the center
            ARM_IDLE_RANGE_PERCENT of the channel's min/max span.
        Pokerarm (ch 6): every SERVO_IDLE_MOVE_INTERVAL_MIN–MAX s, full range.
        """
        # Stagger initial moves so they don't all fire at t=0
        _next_head          = time.monotonic() + random.uniform(2.0, 4.0)
        _next_tilt          = time.monotonic() + random.uniform(3.0, 6.0)
        _next_visor         = time.monotonic() + random.uniform(3.0, 6.0)
        _next_elbow         = time.monotonic() + random.uniform(3.0, 5.0)
        _next_expressive    = time.monotonic() + random.uniform(1.0, 2.5)
        _next_other_arm     = time.monotonic() + random.uniform(2.0, 4.0)

        while not self._stop_event.is_set():
            now = time.monotonic()

            # --- Head idle: neck (ch 0) and headlift (ch 1) ---
            if now >= _next_head:
                ch = random.choice(config.IDLE_HEAD_CHANNELS)

                if ch == config.SERVO_HEAD_LIFT:
                    # Higher qµs = head up, lower = head down.  Constrain to
                    # [min, neutral] so Rex always looks upward or level, never
                    # drooping down (Rex is ~3 ft tall and looks up at people).
                    idle_lo = config.SERVO_CHANNELS[1]["min"]       # 1984 — head up
                    idle_hi = config.SERVO_CHANNELS[1]["neutral"]   # 6000 — head level
                else:
                    # Neck: middle 60% of effective range for lazy turns
                    lo, hi    = self._effective_limits(ch)
                    center    = (lo + hi) // 2
                    half_span = (hi - lo) // 2
                    idle_lo   = center - int(half_span * 0.6)
                    idle_hi   = center + int(half_span * 0.6)

                target = random.randint(idle_lo, idle_hi)
                log.debug("Head idle: ch %d (%s) → %d (range %d–%d)",
                          ch, config.SERVO_CHANNELS[ch]["name"], target, idle_lo, idle_hi)
                with self._lock:
                    self._send_speed(ch, config.SERVO_HEAD_IDLE_SPEED)
                    self._send_target(ch, target)
                _next_head = now + random.uniform(3.0, 5.0)

            # --- Headtilt idle (ch 2): slow drift toward resting pose ---
            # Targets a small window around IDLE_HEAD_TILT_REST so Rex
            # gradually settles to a slightly-downward resting angle.
            # SERVO_HEAD_IDLE_SPEED (3) ensures the drift is barely perceptible.
            if now >= _next_tilt:
                tilt_min = config.SERVO_CHANNELS[config.SERVO_HEAD_TILT]["min"]
                tilt_max = config.SERVO_CHANNELS[config.SERVO_HEAD_TILT]["max"]
                target = _clamp(
                    random.randint(
                        config.IDLE_HEAD_TILT_REST - 150,
                        config.IDLE_HEAD_TILT_REST + 200,
                    ),
                    tilt_min, tilt_max,
                )
                log.debug("Headtilt idle: ch 2 → %d (rest=%d)", target, config.IDLE_HEAD_TILT_REST)
                with self._lock:
                    self._send_speed(config.SERVO_HEAD_TILT, config.SERVO_HEAD_IDLE_SPEED)
                    self._send_target(config.SERVO_HEAD_TILT, target)
                _next_tilt = now + random.uniform(4.0, 7.0)

            # --- Visor idle: ch 3 ---
            if now >= _next_visor:
                # Drift around a naturally open resting position — centre ~5650,
                # ±150 qµs window.  Higher = more open; this keeps eyes visible
                # without being fully extended.
                _VISOR_IDLE_CENTER = 5650
                _VISOR_IDLE_HALF   = 150
                v_min = config.SERVO_CHANNELS[config.SERVO_VISOR]["min"]
                v_max = config.SERVO_CHANNELS[config.SERVO_VISOR]["max"]
                target = _clamp(
                    random.randint(_VISOR_IDLE_CENTER - _VISOR_IDLE_HALF,
                                   _VISOR_IDLE_CENTER + _VISOR_IDLE_HALF),
                    v_min, v_max,
                )
                with self._lock:
                    self._send_speed(config.SERVO_VISOR, config.SERVO_VISOR_IDLE_SPEED)
                    self._send_target(config.SERVO_VISOR, target)
                _next_visor = now + random.uniform(5.0, 8.0)

            # --- Elbow idle (ch 4): drift toward IDLE_ELBOW_REST ---
            # Skipped while pause_arm_idle() is active (arm animation in progress).
            # ch 4 is excluded from the expressive arm random section below so
            # only this dedicated timer drives it during idle.
            if now >= _next_elbow and not self._arm_idle_pause.is_set():
                elbow_lo, elbow_hi = self._effective_limits(config.SERVO_ARM_LEFT)
                target = _clamp(
                    random.randint(
                        config.IDLE_ELBOW_REST - 50,
                        config.IDLE_ELBOW_REST + 100,
                    ),
                    elbow_lo, elbow_hi,
                )
                log.debug("Elbow idle: ch 4 → %d (rest=%d)", target, config.IDLE_ELBOW_REST)
                with self._lock:
                    self._send_speed(config.SERVO_ARM_LEFT, config.SERVO_HEAD_IDLE_SPEED)
                    self._send_target(config.SERVO_ARM_LEFT, target)
                _next_elbow = now + random.uniform(4.0, 7.0)

            # --- Expressive arms (ch 5, 7): center ARM_IDLE_RANGE_PERCENT of range ---
            # ch 4 (elbow) has its own dedicated timer above — excluded here.
            # Skipped while pause_arm_idle() is active (arm animation in progress).
            if now >= _next_expressive and not self._arm_idle_pause.is_set():
                ch = random.choice([config.SERVO_HAND_LEFT, config.SERVO_HAND_RIGHT])
                lo, hi     = self._effective_limits(ch)
                center     = (lo + hi) // 2
                half_span  = int((hi - lo) * config.ARM_IDLE_RANGE_PERCENT) // 2
                target = _clamp(
                    random.randint(center - half_span, center + half_span),
                    lo, hi,
                )
                log.debug("Expressive arm idle: ch %d (%s) → %d",
                          ch, config.SERVO_CHANNELS[ch]["name"], target)
                with self._lock:
                    self._send_target(ch, target)
                _next_expressive = now + random.uniform(
                    config.ARM_IDLE_INTERVAL_MIN, config.ARM_IDLE_INTERVAL_MAX
                )

            # --- Pokerarm (ch 6): full range, standard interval ---
            if now >= _next_other_arm:
                lo, hi = self._effective_limits(config.SERVO_ARM_RIGHT)
                target = random.randint(lo, hi)
                with self._lock:
                    self._send_target(config.SERVO_ARM_RIGHT, target)
                _next_other_arm = now + random.uniform(
                    config.SERVO_IDLE_MOVE_INTERVAL_MIN,
                    config.SERVO_IDLE_MOVE_INTERVAL_MAX,
                )

            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Internal — serial helpers
    # ------------------------------------------------------------------

    def _send_target(self, channel: int, position: int) -> None:
        """Write a Set Target command.  Caller must hold _lock."""
        self._write_command_locked(_encode(_CMD_SET_TARGET, channel, position))

    def _send_speed(self, channel: int, speed: int) -> None:
        """Write a Set Speed command.  Caller must hold _lock."""
        self._write_command_locked(_encode(_CMD_SET_SPEED, channel, speed))

    def _apply_speed(self, speed: int) -> None:
        """Set the same speed on every managed channel."""
        with self._lock:
            for channel in _ALL_CHANNELS:
                self._send_speed(channel, speed)

    def _apply_acceleration(self) -> None:
        """Set per-channel acceleration from SERVO_CHANNELS config."""
        with self._lock:
            for channel, cfg in config.SERVO_CHANNELS.items():
                self._write_command_locked(
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

    def _write_command_locked(self, raw: bytes) -> None:
        """Write a Maestro command, reopening the serial port once on failure.

        Caller must hold _lock.
        """
        try:
            self._serial.write(raw)
        except _SERIAL_ERRORS as exc:
            log.warning("Maestro write failed — attempting reconnect: %s", exc)
            self._reconnect_serial_locked()
            self._serial.write(raw)

    def _reconnect_serial_locked(self) -> None:
        """Re-open the Maestro serial port after a runtime disconnect.

        Caller must hold _lock.
        """
        self._close_serial(self._serial)
        self._serial = self._open_maestro_serial()

    @staticmethod
    def _close_serial(port: serial.Serial | None) -> None:
        """Best-effort serial close used by normal shutdown and reconnects."""
        if port is None:
            return
        try:
            port.close()
        except Exception:
            pass
