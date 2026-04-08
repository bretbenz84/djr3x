"""
hardware/leds.py — Arduino Nano LED control for DJ-R3X.

Two Arduino Nanos:
  Chest Nano (NANO_CHEST_PORT) — chest LED ring/strip effects
  Head Nano  (NANO_HEAD_PORT)  — mouth NeoPixel grid + eye LEDs

All commands are newline-terminated ASCII strings.  The Nanos are dumb
executors; all logic lives here on the Pi.

Thread model
------------
  Main / state-machine thread
    Calls set_chest_effect(), set_head_effect(), set_eye_color(),
    start_mouth(), stop_mouth(), start(), stop().

  Mouth brightness thread (_mouth_thread)
    Reads player.rms at ~20 Hz and sends SPEAK_LEVEL:<n> to the head Nano only
    while speech is active.  Created by start_mouth(), torn down by
    stop_mouth() / stop().

Usage
-----
    from audio.player import AudioPlayer
    from hardware.leds import LEDController

    player = AudioPlayer()
    leds   = LEDController(player)
    leds.start()

    leds.set_chest_effect("IDLE")
    leds.set_head_effect("ACTIVE")
    leds.set_eye_color(0, 180, 255)

    leds.start_mouth()   # begins RMS → brightness loop
    ...                  # speech plays
    leds.stop_mouth()

    leds.stop()
    player.close()
"""

from __future__ import annotations

import logging
import threading
import time

import serial

import config

log = logging.getLogger(__name__)

# Mouth brightness polling interval (seconds).  ~30 Hz is smooth without
# flooding the Nano's 115200-baud serial buffer.
_MOUTH_POLL_INTERVAL: float = 1 / 30


# ---------------------------------------------------------------------------
# LEDController
# ---------------------------------------------------------------------------

class LEDController:
    """Manages LED output on both Arduino Nanos.

    Both serial connections are optional at construction time — if a port
    cannot be opened (hardware missing during development) a warning is logged
    and the corresponding Nano is simply skipped on every send.
    """

    def __init__(self, player) -> None:
        """
        Parameters
        ----------
        player : AudioPlayer
            Live AudioPlayer instance.  Its `.rms` property drives the mouth
            brightness thread.
        """
        self._player = player

        self._chest: serial.Serial | None = _open_serial(
            config.NANO_CHEST_PORT, config.LED_NANO_BAUD, label="chest"
        )
        self._head: serial.Serial | None = _open_serial(
            config.NANO_HEAD_PORT, config.LED_NANO_BAUD, label="head"
        )

        # Mouth brightness thread state
        self._mouth_stop   = threading.Event()
        self._mouth_thread: threading.Thread | None = None
        self._mouth_active: bool = False   # True while mouth is in SPEAK mode

        # Serialise writes from the main thread and mouth thread.
        # Each Nano gets its own lock so chest writes never block head writes.
        self._chest_lock = threading.Lock()
        self._head_lock  = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Put both Nanos into the idle state.  Safe to call multiple times."""
        self._send_chest(config.LED_CMD_IDLE)
        self._send_head(config.LED_CMD_IDLE)

    def stop(self) -> None:
        """Stop the mouth thread and turn both Nanos off."""
        self.stop_mouth()
        self._send_chest(config.LED_CMD_OFF)
        self._send_head(config.LED_CMD_OFF)

    def close(self) -> None:
        """stop() then close serial ports."""
        self.stop()
        if self._chest is not None:
            try:
                self._chest.close()
            except Exception:
                pass
        if self._head is not None:
            try:
                self._head.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Effect / color control
    # ------------------------------------------------------------------

    def set_chest_effect(self, effect: str) -> None:
        """Send a named effect command to the chest Nano.

        ``effect`` should be one of the LED_CMD_* constant *values* (without
        the trailing newline) or the constant itself, e.g.:

            leds.set_chest_effect("IDLE")
            leds.set_chest_effect(config.LED_CMD_ACTIVE)  # also fine
        """
        self._send_chest(_normalise_cmd(effect))

    def set_head_effect(self, effect: str) -> None:
        """Send a named effect command to the head Nano (mouth grid)."""
        self._send_head(_normalise_cmd(effect))

    def set_eye_color(self, r: int, g: int, b: int) -> None:
        """Set the eye LED color on the head Nano.

        Parameters are clamped to 0–255.
        """
        r, g, b = _clamp(r), _clamp(g), _clamp(b)
        self._send_head(config.LED_CMD_EYE_COLOR.format(r, g, b))

    # ------------------------------------------------------------------
    # Mouth speaking animation
    # ------------------------------------------------------------------

    def set_sleep_mode(self) -> None:
        """Start the red breathing mouth animation on the head Nano (SLEEP mode)."""
        self._send_head("SLEEP\n")

    def set_mouth_emotion(self, emotion: str) -> None:
        """Send SPEAK:{emotion} to the head Nano to set the mouth colour.

        Call once before start_mouth() so the Arduino knows the emotion
        colour before SPEAK_LEVEL commands begin flowing.  Valid values:
        neutral, happy, excited, sad, angry.
        """
        self._send_head(config.LED_CMD_SPEAK.format(emotion))

    def start_mouth(self) -> None:
        """Start the RMS → SPEAK_LEVEL thread.  No-op if already running."""
        if self._mouth_thread is not None and self._mouth_thread.is_alive():
            return
        self._mouth_active = True
        self._mouth_stop.clear()
        self._mouth_thread = threading.Thread(
            target=self._mouth_worker,
            daemon=True,
            name="djr3x-mouth-leds",
        )
        self._mouth_thread.start()

    def stop_mouth(self) -> None:
        """Stop the mouth level thread and send SPEAK_STOP to the head Nano.

        Sends SPEAK_STOP three times with 50 ms gaps to survive occasional
        serial drops.  Also schedules a 2-second watchdog that re-sends
        SPEAK_STOP if the mouth is still not in SPEAK mode (last resort).
        """
        self._mouth_stop.set()
        if self._mouth_thread is not None:
            self._mouth_thread.join(timeout=1.0)
            if self._mouth_thread.is_alive():
                log.warning(
                    "Mouth thread did not stop within 1 s — "
                    "possible serial hang; SPEAK_STOP may arrive out of order"
                )
            self._mouth_thread = None
        self._mouth_active = False
        log.debug("Mouth: sending SPEAK_STOP × 3 to head Nano")
        for _ in range(3):
            self._send_head(config.LED_CMD_SPEAK_STOP)
            time.sleep(0.05)
        # Watchdog: 2 s from now, re-send SPEAK_STOP if mouth is still off.
        threading.Thread(
            target=self._mouth_watchdog,
            daemon=True,
            name="djr3x-mouth-watchdog",
        ).start()

    # ------------------------------------------------------------------
    # Internal — send helpers
    # ------------------------------------------------------------------

    def _send_chest(self, cmd: str) -> None:
        """Write a command string to the chest Nano."""
        if self._chest is None:
            return
        with self._chest_lock:
            _write(self._chest, cmd, label="chest")

    def _send_head(self, cmd: str) -> None:
        """Write a command string to the head Nano."""
        if self._head is None:
            return
        with self._head_lock:
            _write(self._head, cmd, label="head")

    # ------------------------------------------------------------------
    # Internal — mouth brightness worker thread + watchdog
    # ------------------------------------------------------------------

    def _mouth_worker(self) -> None:
        """Poll player.rms and send SPEAK_LEVEL:<n> to the head Nano at ~20 Hz."""
        while not self._mouth_stop.is_set():
            level = _clamp(int(round(self._player.rms)))
            # Clamp sub-threshold values to zero so transient near-silence
            # during breath pauses doesn't produce ambient pre-glow on the Nano.
            if level < config.MOUTH_LED_MIN_RMS:
                level = 0
            self._send_head(config.LED_CMD_SPEAK_LEVEL.format(level))
            time.sleep(_MOUTH_POLL_INTERVAL)

    def _mouth_watchdog(self) -> None:
        """Last-resort safety net: 2 s after stop_mouth(), re-send SPEAK_STOP if
        the mouth is still not active (i.e. no new speech started in the meantime).
        """
        time.sleep(2.0)
        if not self._mouth_active:
            log.debug("Mouth watchdog: re-sending SPEAK_STOP × 3 (safety net)")
            for _ in range(3):
                self._send_head(config.LED_CMD_SPEAK_STOP)
                time.sleep(0.05)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _open_serial(port: str | None, baud: int, label: str) -> serial.Serial | None:
    """Try to open a serial port, retrying up to SERIAL_RETRY_ATTEMPTS times.

    Returns None immediately (no retry) if port is None — caller has not
    configured this Nano.  Returns None (with a warning) if all attempts fail.
    """
    if port is None:
        log.info("LEDs: %s Nano not configured — skipping", label)
        return None

    for attempt in range(1, config.SERIAL_RETRY_ATTEMPTS + 1):
        try:
            log.info(
                "LEDs: opening %s Nano on %s at %d baud (attempt %d/%d)",
                label, port, baud, attempt, config.SERIAL_RETRY_ATTEMPTS,
            )
            s = serial.Serial(port, baud, timeout=1.0)
            log.info("LEDs: %s Nano opened on attempt %d", label, attempt)
            return s
        except serial.SerialException as exc:
            log.warning(
                "LEDs: could not open %s Nano on %s (attempt %d/%d) — %s",
                label, port, attempt, config.SERIAL_RETRY_ATTEMPTS, exc,
            )
            if attempt < config.SERIAL_RETRY_ATTEMPTS:
                log.info(
                    "LEDs: retrying %s Nano in %.0f s …",
                    label, config.SERIAL_RETRY_DELAY,
                )
                time.sleep(config.SERIAL_RETRY_DELAY)
    log.warning(
        "LEDs: %s Nano on %s unavailable after %d attempts — hardware missing?",
        label, port, config.SERIAL_RETRY_ATTEMPTS,
    )
    return None


def _write(port: serial.Serial, cmd: str, label: str) -> None:
    """Write cmd to port; log and swallow serial errors."""
    try:
        port.write(cmd.encode())
    except serial.SerialException as exc:
        log.warning("LEDs: write to %s Nano failed — %s", label, exc)


def _normalise_cmd(effect: str) -> str:
    """Ensure the command ends with exactly one newline."""
    return effect.strip() + "\n"


def _clamp(value: int) -> int:
    """Clamp an integer to the 0–255 range."""
    return max(0, min(255, value))
