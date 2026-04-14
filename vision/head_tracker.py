"""
vision/head_tracker.py — Real-time three-axis face tracking for DJ-R3X.

Shares the existing vision.camera.Camera instance rather than opening a
second VideoCapture — Linux V4L2 only permits one capture per device, so
all frame reads go through Camera.get_tracking_frame() which resizes the
shared 1080p capture down to 320×240 for detection.

Tracking geometry
-----------------
  X axis (neck, ch 0) — left/right panning:
    Face at frame-left   (x=0)          → neck MIN (1984 qµs) — turns left
    Face at frame-centre (x=frame_w/2)  → neck NEUTRAL
    Face at frame-right  (x=frame_w)    → neck MAX (9984 qµs) — turns right

  Y axis (headlift, ch 1) — follows face height across full range:
    Face at top of frame    (y=0)          → headlift MAX (7744) — head up
    Face at bottom of frame (y=frame_h)    → headlift MIN (1984) — head down
    Full linear mapping so Rex follows tall adults up and short children down.

  Y axis (headtilt, ch 2) — tilts with face height (inverted channel):
    Headtilt is INVERTED — lower qµs = head tilts up, higher = tilts down.
    Continuous linear mapping with a bias shift:
      t_tilt = tilt_neutral + (y_scaled − TILT_Y_BIAS) × tilt_span
    With default bias 0.35 the neutral crossover is at y_scaled ≈ 0.35
    (roughly 38 % from top of frame), so faces at normal viewing height
    already produce a downward tilt.
    Clamped within physical servo limits [3904, 5504].

Thread model
------------
  _tracking_thread — background daemon thread.
    Calls camera.get_tracking_frame() at up to HEAD_TRACKING_UPDATE_HZ.
    Detects faces with a Haar cascade classifier.
    Applies exponential smoothing (alpha = HEAD_TRACKING_ALPHA).
    Only sends a servo command when the smoothed position has moved more
    than HEAD_TRACKING_DEAD_ZONE qµs — suppresses micro-jitter.
    When paused: loop stays alive, servo commands are suppressed.
    When no face detected: after a short timeout, optionally runs a
    deterministic face-search sweep until a face is found or the burst ends.

Speed note
----------
  The tracker resets _TRACKING_SPEED on neck, headlift, and headtilt before
  every servo write so a preceding set_emotion() or idle-loop speed change
  does not make tracking sluggish.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable

import cv2

import config as _cfg_module

if TYPE_CHECKING:
    from hardware.servos import ServoController
    from vision.camera import Camera

log = logging.getLogger(__name__)

# Maestro speed used for tracking moves (units: 0.25 µs / 10 ms; 0 = unlimited).
_TRACKING_SPEED: int = 200

_CH_NECK = 0
_CH_LIFT = 1
_CH_TILT = 2


class HeadTracker:
    """Two-axis real-time face tracking.

    Parameters
    ----------
    servo_controller : ServoController | None
        Live ServoController instance.  Pass None when servos are absent
        (dev machine) — the detection loop still runs but servo calls are
        skipped.
    camera : Camera
        The shared Camera instance.  Frames are fetched via
        camera.get_tracking_frame() so no second VideoCapture is opened.
    cfg : module, optional
        Config module.  Defaults to ``config``; tests may inject a mock.
    """

    def __init__(
        self,
        servo_controller: ServoController | None,
        camera: Camera,
        cfg=_cfg_module,
        on_face_appear: Callable[[], None] | None = None,
    ) -> None:
        self._servos = servo_controller
        self._camera = camera
        self._cfg = cfg

        self._frame_w, self._frame_h = cfg.HEAD_TRACKING_RESOLUTION

        # Wide-angle compensation margins (fraction of frame from each edge).
        # Faces within x_margin of the left/right edge → servo fully extended.
        self._x_margin: float = getattr(cfg, "HEAD_TRACKING_X_MARGIN", 0.15)
        self._y_margin: float = getattr(cfg, "HEAD_TRACKING_Y_MARGIN", 0.10)

        # Bias shifts the tilt neutral crossover above 0.5 so faces in the
        # upper-middle of the frame already produce a downward tilt.
        self._tilt_y_bias: float = getattr(cfg, "HEAD_TRACKING_TILT_Y_BIAS", 0.35)

        # Neck limits (full pan range)
        _neck = cfg.SERVO_CHANNELS[_CH_NECK]
        self._neck_min:     int = _neck["min"]
        self._neck_max:     int = _neck["max"]
        self._neck_neutral: int = _neck["neutral"]

        # Headlift limits — full range to follow any face height
        _lift = cfg.SERVO_CHANNELS[_CH_LIFT]
        self._lift_min:     int = _lift["min"]       # 1984 — head fully down
        self._lift_neutral: int = _lift["neutral"]   # 6000 — head level
        self._lift_max:     int = _lift["max"]       # 7744 — head fully up

        # Headtilt limits (clamped to ±15 % of full span from neutral)
        _tilt = cfg.SERVO_CHANNELS[_CH_TILT]
        self._tilt_neutral: int = _tilt["neutral"]
        _tilt_half = 400                                            # ±400 qµs from neutral
        self._tilt_lo: int = max(_tilt["min"],
                                 self._tilt_neutral - _tilt_half)  # 3920 qµs (tilts up)
        self._tilt_hi: int = min(_tilt["max"],
                                 self._tilt_neutral + _tilt_half)  # 4720 qµs (tilts down)

        # Visor max — sent once at tracker start to keep camera unobstructed
        self._visor_max: int = cfg.SERVO_CHANNELS[3]["max"]

        # EMA-smoothed positions (float for sub-qµs accumulation)
        self._smooth_neck: float = float(self._neck_neutral)
        self._smooth_lift: float = float(self._lift_neutral)   # start level; tracker adjusts
        self._smooth_tilt: float = float(self._tilt_neutral)

        # Last positions actually sent to the Maestro
        self._sent_neck: int = self._neck_neutral
        self._sent_lift: int = self._lift_neutral
        self._sent_tilt: int = self._tilt_neutral

        # Current tracking targets (updated on each successful detection)
        self._target_neck: int = self._neck_neutral
        self._target_lift: int = self._lift_neutral
        self._target_tilt: int = self._tilt_neutral

        # Thread control
        self._stop_event  = threading.Event()
        self._pause_event = threading.Event()   # set → paused
        self._thread: threading.Thread | None = None

        # Protects _smooth_*, _sent_*, _target_* shared between the detection
        # thread and wait_for_center() (called from the main thread).
        self._state_lock = threading.Lock()

        # Face-absence tracking — fires on_face_appear when a face is detected
        # after FACE_APPEAR_ABSENT_SECONDS of no detections, confirmed by
        # FACE_APPEAR_FRAME_COUNT consecutive frames.
        self._on_face_appear: Callable[[], None] | None = on_face_appear
        # Initialise to now so the absence clock starts from tracker start.
        self._last_face_seen_time: float = time.monotonic()
        self._in_face_appear_event: bool = False   # True while counting consecutive frames
        self._appear_frame_count:   int  = 0       # consecutive face-detected frames in event
        self._face_appear_fired:    bool = False   # prevents double-fire per appearance

        # Face-search state — active only while face search is enabled and no
        # face has been detected for HEAD_SEARCH_LOST_FACE_SECONDS.
        self._face_search_enabled: bool = bool(getattr(cfg, "HEAD_SEARCH_ENABLED", True))
        self._search_active: bool = False
        self._search_step_index: int = 0
        self._search_step_started_at: float = 0.0
        self._search_step_move_duration: float = 0.0
        self._search_burst_started_at: float = 0.0
        self._search_cooldown_until: float = 0.0

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
        # Open visor fully so the camera has an unobstructed view.
        if self._servos is not None:
            self._servos.set_tracking_active(True)
            self._servos.set_position(3, self._visor_max)
        log.info(
            "HeadTracker: started — shared camera, tracking frame %dx%d",
            self._frame_w, self._frame_h,
        )

    def stop(self) -> None:
        """Signal the detection thread to exit and wait up to 3 s."""
        self._stop_event.set()
        self._pause_event.clear()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                log.warning("HeadTracker: thread did not stop within 3 s")
            self._thread = None
        if self._servos is not None:
            self._servos.set_tracking_active(False)
        log.info("HeadTracker: stopped")

    def pause(self, reason: str = "") -> None:
        """Suspend servo updates while keeping the detection loop alive."""
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
        False if the timeout expired first.  Used immediately before a
        recognition capture so the face is centred in the high-res frame.
        """
        dead_zone = self._cfg.HEAD_TRACKING_DEAD_ZONE
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._state_lock:
                neck_ok = abs(self._smooth_neck - self._target_neck) < dead_zone
                lift_ok = abs(self._smooth_lift - self._target_lift) < dead_zone
                tilt_ok = abs(self._smooth_tilt - self._target_tilt) < dead_zone
            if neck_ok and lift_ok and tilt_ok:
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

    def face_recently_seen(self, within_seconds: float = 2.0) -> bool:
        """Return True if a face was detected within *within_seconds* of now.

        Thread-safe — reads _last_face_seen_time which is only written by the
        detection thread, and float reads are atomic on CPython.  Used by the
        state machine to decide whether to attempt a face-triggered greeting
        without waiting for the full absence-timer cycle.
        """
        return (time.monotonic() - self._last_face_seen_time) <= within_seconds

    def set_face_search_enabled(self, enabled: bool, reason: str = "") -> None:
        """Enable or disable active face-search sweeps while no face is locked."""
        new_value = bool(enabled) and bool(getattr(self._cfg, "HEAD_SEARCH_ENABLED", True))
        if self._face_search_enabled == new_value:
            return
        self._face_search_enabled = new_value
        if not new_value:
            self._stop_face_search(reason="disabled")
        log.info(
            "HeadTracker: face search %s%s",
            "enabled" if new_value else "disabled",
            f" ({reason})" if reason else "",
        )

    def _search_targets_for_step(self, step_index: int) -> tuple[str, int, int, int]:
        """Return the deterministic servo targets for one face-search step."""
        steps = (
            ("center", self._neck_neutral, self._lift_neutral, self._tilt_neutral),
            ("neck_left", self._neck_min, self._lift_neutral, self._tilt_neutral),
            ("neck_right", self._neck_max, self._lift_neutral, self._tilt_neutral),
            (
                "tilt_down",
                self._neck_neutral,
                self._lift_neutral,
                self._cfg.SERVO_CHANNELS[_CH_TILT]["max"],
            ),
            (
                "head_down",
                self._neck_neutral,
                self._lift_min,
                self._cfg.SERVO_CHANNELS[_CH_TILT]["max"],
            ),
        )
        return steps[step_index % len(steps)]

    def _stop_face_search(
        self,
        *,
        reason: str = "",
        cooldown_until: float = 0.0,
    ) -> None:
        """Reset any in-progress face-search sweep."""
        was_active = self._search_active
        self._search_active = False
        self._search_step_index = 0
        self._search_step_started_at = 0.0
        self._search_step_move_duration = 0.0
        self._search_burst_started_at = 0.0
        self._search_cooldown_until = cooldown_until
        if was_active:
            log.info(
                "HeadTracker: face search stopped%s",
                f" ({reason})" if reason else "",
            )

    def _apply_direct_targets(
        self,
        neck: int,
        lift: int,
        tilt: int,
        *,
        speed: int,
        force: bool = False,
    ) -> None:
        """Apply non-smoothed tracking/search targets immediately."""
        dead_zone = self._cfg.HEAD_TRACKING_DEAD_ZONE
        with self._state_lock:
            self._target_neck = neck
            self._target_lift = lift
            self._target_tilt = tilt
            self._smooth_neck = float(neck)
            self._smooth_lift = float(lift)
            self._smooth_tilt = float(tilt)

            send_neck = force or abs(neck - self._sent_neck) >= dead_zone
            send_lift = force or abs(lift - self._sent_lift) >= dead_zone
            send_tilt = force or abs(tilt - self._sent_tilt) >= dead_zone
            if send_neck:
                self._sent_neck = neck
            if send_lift:
                self._sent_lift = lift
            if send_tilt:
                self._sent_tilt = tilt

        if self._pause_event.is_set() or self._servos is None:
            return
        if send_neck:
            self._servos.set_channel_speed(_CH_NECK, speed)
            self._servos.set_position(_CH_NECK, neck)
        if send_lift:
            self._servos.set_channel_speed(_CH_LIFT, speed)
            self._servos.set_position(_CH_LIFT, lift)
        if send_tilt:
            self._servos.set_channel_speed(_CH_TILT, speed)
            self._servos.set_position(_CH_TILT, tilt)

    def _estimate_move_duration(
        self,
        neck: int,
        lift: int,
        tilt: int,
        *,
        speed: int,
    ) -> float:
        """Estimate how long the slowest search axis will take to reach target."""
        if speed <= 0:
            return 0.0
        with self._state_lock:
            neck_delta = abs(neck - self._sent_neck)
            lift_delta = abs(lift - self._sent_lift)
            tilt_delta = abs(tilt - self._sent_tilt)
        # Pololu speed units are qµs per 10 ms, so:
        # duration_seconds = delta_qµs / speed * 0.01
        max_delta = max(neck_delta, lift_delta, tilt_delta)
        return (max_delta / float(speed)) * 0.01

    def _advance_face_search(self, now: float) -> None:
        """Run one step of the deterministic face-search sweep."""
        if now < self._search_cooldown_until:
            return

        search_speed = int(getattr(self._cfg, "HEAD_SEARCH_SPEED", _TRACKING_SPEED))
        post_move_pause = float(
            getattr(self._cfg, "HEAD_SEARCH_STEP_HOLD_SECONDS", "0.75")
        )
        burst_seconds = float(getattr(self._cfg, "HEAD_SEARCH_BURST_SECONDS", "12.0"))
        cooldown_seconds = float(
            getattr(self._cfg, "HEAD_SEARCH_COOLDOWN_SECONDS", "1.5")
        )

        if not self._search_active:
            self._search_active = True
            self._search_step_index = 0
            self._search_step_started_at = 0.0
            self._search_burst_started_at = now
            log.info(
                "HeadTracker: no face lock for %.1f s — starting face search",
                now - self._last_face_seen_time,
            )

        if (now - self._search_burst_started_at) >= burst_seconds:
            log.info(
                "HeadTracker: face search burst expired after %.1f s — recentering",
                burst_seconds,
            )
            self._apply_direct_targets(
                self._neck_neutral,
                self._lift_neutral,
                self._tilt_neutral,
                speed=search_speed,
            )
            self._stop_face_search(
                reason="burst timeout",
                cooldown_until=now + cooldown_seconds,
            )
            return

        if (
            self._search_step_started_at
            and (now - self._search_step_started_at)
            < (self._search_step_move_duration + post_move_pause)
        ):
            return

        if self._search_step_started_at:
            self._search_step_index += 1
        self._search_step_started_at = now
        step_name, neck, lift, tilt = self._search_targets_for_step(self._search_step_index)
        self._search_step_move_duration = self._estimate_move_duration(
            neck,
            lift,
            tilt,
            speed=search_speed,
        )
        log.debug("HeadTracker: face search step=%s", step_name)
        self._apply_direct_targets(
            neck,
            lift,
            tilt,
            speed=search_speed,
            force=True,
        )

    # ------------------------------------------------------------------
    # Background detection loop
    # ------------------------------------------------------------------

    def _tracking_loop(self) -> None:
        if not self._camera.is_available():
            log.warning(
                "HeadTracker: camera not available at thread start — "
                "head tracking disabled for this session"
            )
            return

        classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        interval  = 1.0 / self._cfg.HEAD_TRACKING_UPDATE_HZ
        alpha     = self._cfg.HEAD_TRACKING_ALPHA
        dead_zone = self._cfg.HEAD_TRACKING_DEAD_ZONE

        # FPS health-check — warn if average drops below 5.
        _frame_times: list[float] = []

        log.info("HeadTracker: detection loop running")

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            frame = self._camera.get_tracking_frame()
            if frame is None:
                # Camera temporarily unavailable — back off and retry.
                time.sleep(0.1)
                continue

            # ── Face detection ────────────────────────────────────────────
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray  = cv2.equalizeHist(gray)
            faces = classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            if len(faces) > 0:
                if self._search_active:
                    log.info("HeadTracker: face detected during search — resuming tracking")
                    self._stop_face_search(reason="face detected")

                # Pick the largest face when multiple are visible.
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_cx = x + w // 2
                face_cy = y + h // 2

                # ── Face-absence tracking ─────────────────────────────────
                now = time.monotonic()
                absent_seconds = getattr(self._cfg, "FACE_APPEAR_ABSENT_SECONDS", 15.0)
                confirm_frames = getattr(self._cfg, "FACE_APPEAR_FRAME_COUNT", 2)
                if now - self._last_face_seen_time >= absent_seconds and not self._in_face_appear_event:
                    # Long absence just ended — start counting confirmation frames.
                    self._in_face_appear_event = True
                    self._appear_frame_count = 0
                    log.debug(
                        "HeadTracker: face appeared after %.1f s absence — counting frames",
                        now - self._last_face_seen_time,
                    )
                if self._in_face_appear_event:
                    self._appear_frame_count += 1
                    if (
                        self._appear_frame_count >= confirm_frames
                        and not self._face_appear_fired
                        and self._on_face_appear is not None
                    ):
                        self._face_appear_fired = True
                        log.info(
                            "HeadTracker: face confirmed after %d frames — firing on_face_appear",
                            self._appear_frame_count,
                        )
                        try:
                            self._on_face_appear()
                        except Exception:
                            log.exception("HeadTracker: on_face_appear callback raised")
                self._last_face_seen_time = now

                # Normalise face position to [0,1] then remap to account for the
                # wide-angle lens.  Faces detected within the edge margin already
                # represent an extreme viewing angle, so we stretch the inner
                # active zone [margin, 1-margin] across the full servo range and
                # clamp anything outside it to the servo extreme.
                x_raw = face_cx / self._frame_w   # 0.0 (left) … 1.0 (right)
                y_raw = face_cy / self._frame_h   # 0.0 (top)  … 1.0 (bottom)

                x_margin = self._x_margin
                y_margin = self._y_margin
                x_active = 1.0 - 2.0 * x_margin  # width of the inner zone
                y_active = 1.0 - 2.0 * y_margin

                # Clamp to [0,1] after scaling — this is what saturates the servo
                # at its full extent when the face reaches the margin zone.
                x_scaled = max(0.0, min(1.0, (x_raw - x_margin) / x_active))
                y_scaled = max(0.0, min(1.0, (y_raw - y_margin) / y_active))

                # X → neck: face-left = neck min, face-right = neck max
                t_neck = int(
                    self._neck_min
                    + x_scaled * (self._neck_max - self._neck_min)
                )
                t_neck = max(self._neck_min, min(self._neck_max, t_neck))

                # Y → headlift: face at top = max (head up), bottom = min (head down)
                t_lift = int(
                    self._lift_max
                    - y_scaled * (self._lift_max - self._lift_min)
                )
                t_lift = max(self._lift_min, min(self._lift_max, t_lift))

                # Y → headtilt (inverted: low qµs = up, high = down).
                # Same continuous linear formula as before — no saturation zones.
                # tilt_y_bias shifts the neutral crossover above 0.5 so faces
                # in the upper-middle of the frame produce a downward tilt.
                tilt_span = self._tilt_hi - self._tilt_lo
                t_tilt = int(
                    self._tilt_neutral
                    + (y_scaled - self._tilt_y_bias) * tilt_span
                )
                t_tilt = max(self._tilt_lo, min(self._tilt_hi, t_tilt))

                log.debug(
                    "HeadTracker targets: neck=%d  lift=%d  tilt=%d  (face cx=%d cy=%d)",
                    t_neck, t_lift, t_tilt, face_cx, face_cy,
                )

                with self._state_lock:
                    self._target_neck = t_neck
                    self._target_lift = t_lift
                    self._target_tilt = t_tilt
            else:
                # No face — either hold the last known target briefly or, when
                # enabled, run the deterministic face-search sweep.
                now = time.monotonic()
                self._in_face_appear_event = False
                self._appear_frame_count   = 0
                self._face_appear_fired    = False
                lost_face_seconds = float(
                    getattr(self._cfg, "HEAD_SEARCH_LOST_FACE_SECONDS", "0.8")
                )
                if (
                    self._face_search_enabled
                    and not self._pause_event.is_set()
                    and (now - self._last_face_seen_time) >= lost_face_seconds
                ):
                    self._advance_face_search(now)
                elif self._search_active:
                    self._stop_face_search(reason="paused or face-loss window reset")

            # ── EMA smoothing ─────────────────────────────────────────────
            with self._state_lock:
                self._smooth_neck = (
                    alpha * self._target_neck
                    + (1.0 - alpha) * self._smooth_neck
                )
                self._smooth_lift = (
                    alpha * self._target_lift
                    + (1.0 - alpha) * self._smooth_lift
                )
                self._smooth_tilt = (
                    alpha * self._target_tilt
                    + (1.0 - alpha) * self._smooth_tilt
                )
                new_neck = int(round(self._smooth_neck))
                new_lift = int(round(self._smooth_lift))
                new_tilt = int(round(self._smooth_tilt))
                send_neck = abs(new_neck - self._sent_neck) >= dead_zone
                send_lift = abs(new_lift - self._sent_lift) >= dead_zone
                send_tilt = abs(new_tilt - self._sent_tilt) >= dead_zone
                if send_neck:
                    self._sent_neck = new_neck
                if send_lift:
                    self._sent_lift = new_lift
                if send_tilt:
                    self._sent_tilt = new_tilt

            # ── Servo commands (outside lock — never hold it during IO) ───
            if not self._pause_event.is_set() and self._servos is not None:
                if send_neck:
                    # Restore tracking speed in case set_emotion() changed it.
                    self._servos.set_channel_speed(_CH_NECK, _TRACKING_SPEED)
                    self._servos.set_position(_CH_NECK, new_neck)
                if send_lift:
                    self._servos.set_channel_speed(_CH_LIFT, _TRACKING_SPEED)
                    self._servos.set_position(_CH_LIFT, new_lift)
                if send_tilt:
                    self._servos.set_channel_speed(_CH_TILT, _TRACKING_SPEED)
                    self._servos.set_position(_CH_TILT, new_tilt)

            # ── FPS health check ──────────────────────────────────────────
            elapsed = time.monotonic() - t0
            _frame_times.append(elapsed)
            if len(_frame_times) > 30:
                _frame_times.pop(0)
            if len(_frame_times) == 30:
                avg = sum(_frame_times) / 30
                fps = 1.0 / avg if avg > 0 else 0.0
                if fps < 3.0:
                    log.warning(
                        "HeadTracker: average FPS %.1f is below 3 — "
                        "Pi may be under load",
                        fps,
                    )
                _frame_times.clear()

            # ── Throttle to target Hz ─────────────────────────────────────
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        log.info("HeadTracker: detection loop stopped")
