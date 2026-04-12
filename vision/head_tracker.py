"""
vision/head_tracker.py — Real-time two-axis face tracking for DJ-R3X.

Uses a dedicated low-resolution VideoCapture (320×240) and a Haar cascade
classifier to locate faces and continuously point neck (ch 0) and headtilt
(ch 2) at the centred face.  Runs independently of the main 1080p camera so
recognition captures are not affected.

Tracking geometry
-----------------
  X axis (neck, ch 0) — left/right panning:
    Face at frame-left  (x=0)           → neck MAX (9984 qµs) — leftmost
    Face at frame-centre (x=frame_w/2)  → neck NEUTRAL
    Face at frame-right (x=frame_w)     → neck MIN  (1984 qµs) — rightmost

  Y axis (headtilt, ch 2) — subtle tilt only:
    Headtilt is INVERTED — lower qµs = head tilts up.
    Range is clamped to ±15 % of full tilt span from neutral so height
    differences produce only a small, natural-looking tilt.

Thread model
------------
  _tracking_thread — background daemon thread.
    Opens its own VideoCapture at HEAD_TRACKING_RESOLUTION.
    Detects faces with Haar cascade at up to HEAD_TRACKING_UPDATE_HZ.
    Applies exponential smoothing (alpha = HEAD_TRACKING_ALPHA).
    Only sends a servo command when the smoothed position has moved more
    than HEAD_TRACKING_DEAD_ZONE qµs — suppresses micro-jitter.
    When paused: loop stays alive, servo commands are suppressed.
    When no face detected: target is held at the last known position.

Speed note
----------
  The tracker sets _TRACKING_SPEED on neck and headtilt before every servo
  write so a preceding set_emotion() or idle-loop speed change does not make
  tracking sluggish.  The extra serial traffic is minimal (~4 extra bytes
  per tick at 10 Hz).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import cv2

import config as _cfg_module

if TYPE_CHECKING:
    from hardware.servos import ServoController

log = logging.getLogger(__name__)

# Maestro speed used for tracking moves.
# Higher = snappier; lower = smoother but laggy.
# 15 (0.25 µs / 10 ms) lets the servo keep up with a 10 Hz update loop
# while still looking fluid.
_TRACKING_SPEED: int = 15

# Neck and headtilt channel numbers (must match config.SERVO_CHANNELS).
_CH_NECK  = 0
_CH_TILT  = 2


class HeadTracker:
    """Two-axis real-time face tracking.

    Parameters
    ----------
    servo_controller : ServoController | None
        Live ServoController instance.  Pass None (or when servos are
        absent on a dev machine) — the detection loop still runs but
        all servo calls are skipped.
    camera_device_index : int
        OpenCV device index for the tracking camera.  May be the same
        physical device as the main 1080p camera; each VideoCapture opens
        an independent capture session.
    cfg : module, optional
        Config module.  Defaults to ``config`` so production code need
        not pass it; tests can inject a mock.
    """

    def __init__(
        self,
        servo_controller: ServoController | None,
        camera_device_index: int,
        cfg=_cfg_module,
    ) -> None:
        self._servos = servo_controller
        self._device_index = camera_device_index
        self._cfg = cfg

        self._frame_w, self._frame_h = cfg.HEAD_TRACKING_RESOLUTION

        # Neck limits (full range — wide panning)
        _neck = cfg.SERVO_CHANNELS[_CH_NECK]
        self._neck_min:     int = _neck["min"]
        self._neck_max:     int = _neck["max"]
        self._neck_neutral: int = _neck["neutral"]

        # Headtilt limits (clamped to ±15 % of full span from neutral)
        _tilt = cfg.SERVO_CHANNELS[_CH_TILT]
        self._tilt_neutral: int = _tilt["neutral"]
        _tilt_half = int((_tilt["max"] - _tilt["min"]) * 0.15)   # 240 qµs
        self._tilt_lo: int = self._tilt_neutral - _tilt_half      # 4080 qµs (up)
        self._tilt_hi: int = self._tilt_neutral + _tilt_half      # 4560 qµs (down)

        # Current EMA-smoothed positions (float for sub-qµs accumulation)
        self._smooth_neck: float = float(self._neck_neutral)
        self._smooth_tilt: float = float(self._tilt_neutral)

        # Last positions actually sent to the Maestro
        self._sent_neck: int = self._neck_neutral
        self._sent_tilt: int = self._tilt_neutral

        # Current tracking targets (updated on each detection)
        self._target_neck: int = self._neck_neutral
        self._target_tilt: int = self._tilt_neutral

        # Thread control
        self._stop_event  = threading.Event()
        self._pause_event = threading.Event()   # set → paused
        self._thread: threading.Thread | None = None

        # Protects _smooth_*, _sent_*, _target_* shared between the detection
        # thread and wait_for_center() called from the main thread.
        self._state_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background detection thread.  No-op if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self._thread = threading.Thread(
            target=self._tracking_loop,
            daemon=True,
            name="djr3x-head-tracker",
        )
        self._thread.start()
        log.info(
            "HeadTracker: started on device %d at %dx%d",
            self._device_index, self._frame_w, self._frame_h,
        )

    def stop(self) -> None:
        """Signal the detection thread to exit and wait up to 3 s."""
        self._stop_event.set()
        self._pause_event.clear()   # don't leave the thread blocked in a tight poll
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                log.warning("HeadTracker: thread did not stop within 3 s")
            self._thread = None
        log.info("HeadTracker: stopped")

    def pause(self, reason: str = "") -> None:
        """Suspend servo updates while keeping the detection loop alive.

        The EMA state is preserved so tracking resumes smoothly.
        """
        self._pause_event.set()
        log.debug("HeadTracker: paused%s", f" ({reason})" if reason else "")

    def resume(self, reason: str = "") -> None:
        """Resume servo updates after a pause()."""
        self._pause_event.clear()
        log.debug("HeadTracker: resumed%s", f" ({reason})" if reason else "")

    def wait_for_center(self, timeout: float = 2.0) -> bool:
        """Block until both axes have converged on their tracking targets.

        "Converged" means the smoothed position is within HEAD_TRACKING_DEAD_ZONE
        qµs of the current target on both neck and headtilt.

        Returns True if both axes converged before *timeout* seconds elapsed,
        False if the timeout expired first.

        Used immediately before a recognition capture so the face is centred
        in the high-res frame.
        """
        dead_zone = self._cfg.HEAD_TRACKING_DEAD_ZONE
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._state_lock:
                neck_ok = abs(self._smooth_neck - self._target_neck) < dead_zone
                tilt_ok = abs(self._smooth_tilt - self._target_tilt) < dead_zone
            if neck_ok and tilt_ok:
                return True
            time.sleep(0.05)
        log.debug("HeadTracker.wait_for_center: timed out after %.1f s", timeout)
        return False

    @property
    def is_running(self) -> bool:
        """True if the detection thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        """True while paused (servo updates suppressed)."""
        return self._pause_event.is_set()

    # ------------------------------------------------------------------
    # Background detection loop
    # ------------------------------------------------------------------

    def _tracking_loop(self) -> None:
        cap = cv2.VideoCapture(self._device_index)
        if not cap.isOpened():
            log.warning(
                "HeadTracker: could not open camera device %d — "
                "head tracking disabled for this session",
                self._device_index,
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._frame_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_h)

        classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        interval  = 1.0 / self._cfg.HEAD_TRACKING_UPDATE_HZ
        alpha     = self._cfg.HEAD_TRACKING_ALPHA
        dead_zone = self._cfg.HEAD_TRACKING_DEAD_ZONE

        # FPS health-check window — log a warning if we drop below 5 FPS.
        _frame_times: list[float] = []

        log.info("HeadTracker: detection loop running")

        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    log.warning("HeadTracker: failed to read frame — retrying in 100 ms")
                    time.sleep(0.1)
                    continue

                # ── Face detection ────────────────────────────────────────
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = classifier.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                )

                if len(faces) > 0:
                    # Pick the largest face if multiple are visible.
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    face_cx = x + w // 2
                    face_cy = y + h // 2

                    # Map X → neck: left (0) = max, right (frame_w) = min
                    t_neck = int(
                        self._neck_max
                        + (face_cx / self._frame_w) * (self._neck_min - self._neck_max)
                    )
                    t_neck = max(self._neck_min, min(self._neck_max, t_neck))

                    # Map Y → headtilt (inverted + clamped to narrow band)
                    tilt_span = self._tilt_hi - self._tilt_lo
                    t_tilt = int(
                        self._tilt_neutral
                        + ((face_cy / self._frame_h) - 0.5) * tilt_span
                    )
                    t_tilt = max(self._tilt_lo, min(self._tilt_hi, t_tilt))

                    with self._state_lock:
                        self._target_neck = t_neck
                        self._target_tilt = t_tilt
                # else: no face → hold last known target, do not reset to neutral

                # ── EMA smoothing ─────────────────────────────────────────
                with self._state_lock:
                    self._smooth_neck = (
                        alpha * self._target_neck
                        + (1.0 - alpha) * self._smooth_neck
                    )
                    self._smooth_tilt = (
                        alpha * self._target_tilt
                        + (1.0 - alpha) * self._smooth_tilt
                    )
                    new_neck = int(round(self._smooth_neck))
                    new_tilt = int(round(self._smooth_tilt))
                    send_neck = abs(new_neck - self._sent_neck) >= dead_zone
                    send_tilt = abs(new_tilt - self._sent_tilt) >= dead_zone
                    if send_neck:
                        self._sent_neck = new_neck
                    if send_tilt:
                        self._sent_tilt = new_tilt

                # ── Servo commands (outside lock so we never hold it during IO) ──
                if not self._pause_event.is_set() and self._servos is not None:
                    if send_neck:
                        # Reset speed before each send so preceding set_emotion()
                        # or idle-loop speed changes don't slow down tracking.
                        self._servos.set_channel_speed(_CH_NECK, _TRACKING_SPEED)
                        self._servos.set_position(_CH_NECK, new_neck)
                    if send_tilt:
                        self._servos.set_channel_speed(_CH_TILT, _TRACKING_SPEED)
                        self._servos.set_position(_CH_TILT, new_tilt)

                # ── FPS health check ──────────────────────────────────────
                elapsed = time.monotonic() - t0
                _frame_times.append(elapsed)
                if len(_frame_times) > 30:
                    _frame_times.pop(0)
                if len(_frame_times) == 30:
                    avg = sum(_frame_times) / 30
                    fps = 1.0 / avg if avg > 0 else 0.0
                    if fps < 5.0:
                        log.warning(
                            "HeadTracker: average FPS %.1f is below 5 — "
                            "Pi may be under load",
                            fps,
                        )
                    _frame_times.clear()

                # ── Throttle to target Hz ─────────────────────────────────
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            cap.release()
            log.info("HeadTracker: camera released")
