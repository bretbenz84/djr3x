"""
vision/camera.py — Webcam capture for DJ-R3X vision input.

Wraps OpenCV to keep a camera handle open across the program lifetime so
capture_frame() returns in milliseconds rather than paying the open/close
cost on each call.

Typical usage:
    camera = Camera()
    camera.warmup()   # opens the device; logs a warning if unavailable
    camera.start()
    ...
    frame_b64 = camera.capture_frame()   # None if camera not available
    ...
    camera.stop()

Graceful degradation:
    If the camera device cannot be opened, is_available() returns False,
    capture_frame() returns None, and no exception propagates — the rest of
    the program continues normally without vision context.
"""

from __future__ import annotations

import base64
import glob
import logging
import os
import re
import threading
import time
from typing import Optional

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)
_VIDEO_DEVICE_RE = re.compile(r"^/dev/video(\d+)$")


def _linux_device_candidates(source: str) -> list[tuple[int | str, str]]:
    """Return Linux camera candidates, preferring numeric `/dev/videoN` opens."""
    candidates: list[tuple[int | str, str]] = []
    seen: set[int | str] = set()

    def _add(candidate: int | str, label: str) -> None:
        if candidate in seen:
            return
        seen.add(candidate)
        candidates.append((candidate, label))

    def _add_path_candidate(path: str, label: str) -> None:
        match = _VIDEO_DEVICE_RE.match(path)
        if match:
            index = int(match.group(1))
            _add(index, f"{label} (index {index})")
        else:
            _add(path, label)

    resolved = os.path.realpath(source)
    if resolved.startswith("/dev/") and resolved != source:
        _add_path_candidate(resolved, f"{source} -> {resolved}")
    _add_path_candidate(source, source)

    # Some Pi/OpenCV V4L2 builds refuse named `/dev/...` capture paths even
    # when a udev alias points at the right camera, so fall back to probing
    # real `/dev/videoN` nodes as numeric indices.
    for path in sorted(glob.glob("/dev/video[0-9]*")):
        _add_path_candidate(path, path)

    # Final safety net when device nodes are not enumerable yet but V4L2
    # indices still work.
    for index in range(6):
        _add(index, f"index {index}")
    return candidates


def _open_capture(source: int | str) -> cv2.VideoCapture:
    """Open a camera by numeric index or stable /dev path."""
    if isinstance(source, int) and config.PLATFORM != "macos_silicon":
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    if isinstance(source, str) and source.startswith("/dev/"):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    if config.PLATFORM == "macos_silicon":
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(source)


def _candidate_sources(source: int | str) -> list[tuple[int | str, str]]:
    """Return sources to try, supporting auto-detection on macOS."""
    if source == "auto":
        # AVFoundation camera ordering can change across reboots and when
        # Continuity Camera / USB webcams appear, so probe a few indices and
        # keep the first device that actually delivers frames.
        return [(idx, f"auto[{idx}]") for idx in range(6)]
    if isinstance(source, str) and source.startswith("/dev/"):
        return _linux_device_candidates(source)
    return [(source, str(source))]


class Camera:
    """Manages a single webcam device for still-frame capture."""

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._available: bool = False
        # Serialises cap.read() calls between capture_frame() (main thread)
        # and get_tracking_frame() (head-tracker background thread) so two
        # concurrent reads never interleave on the same VideoCapture handle.
        self._cap_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    _OPEN_RETRIES = 3
    _OPEN_RETRY_DELAY = 1.0   # seconds between attempts
    _WARMUP_DELAY = 0.5       # seconds after open before test frame

    def warmup(self) -> None:
        """Open the camera device and verify it can deliver frames.

        Retries up to _OPEN_RETRIES times with _OPEN_RETRY_DELAY second pauses
        to handle USB cameras that aren't ready at startup.  A _WARMUP_DELAY
        second pause after a successful open lets the sensor settle before the
        test frame is read.

        Safe to call once at startup.  Sets is_available() based on whether
        the device opened successfully.
        """
        configured_source = config.CAMERA_DEVICE
        configured_label = config.CAMERA_DEVICE_LABEL
        for attempt in range(1, self._OPEN_RETRIES + 1):
            for source, source_label in _candidate_sources(configured_source):
                cap = _open_capture(source)

                if not cap.isOpened():
                    cap.release()
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)

                # Give the sensor a moment to initialise before reading the test frame.
                time.sleep(self._WARMUP_DELAY)

                ok, _ = cap.read()
                if not ok:
                    cap.release()
                    continue

                self._cap = cap
                self._available = True
                log.info(
                    "Camera: source %s ready (%dx%d) after %d attempt(s)",
                    source_label,
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    attempt,
                )
                return

            log.warning(
                "Camera: source %s could not be opened (attempt %d/%d)%s",
                configured_label,
                attempt,
                self._OPEN_RETRIES,
                " — retrying" if attempt < self._OPEN_RETRIES else " — vision disabled",
            )
            if attempt < self._OPEN_RETRIES:
                time.sleep(self._OPEN_RETRY_DELAY)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """No-op — camera is opened in warmup() and stays open until stop().

        Exists so the startup sequence can call start() uniformly across all
        subsystems without special-casing the camera.
        """

    def stop(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._available = False
        log.info("Camera: released.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """True if the camera opened successfully and is ready to capture."""
        return self._available

    def capture_frame(self) -> Optional[str]:
        """Capture a single frame and return it as a base64-encoded JPEG.

        Returns None if the camera is unavailable or the read fails.
        The returned string is suitable for direct use in an OpenAI
        image_url message payload as:
            f"data:image/jpeg;base64,{frame_b64}"
        """
        if not self._available or self._cap is None:
            log.warning("Camera: unavailable at capture time — attempting reopen")
            self._reopen()
            if not self._available or self._cap is None:
                log.debug("Camera: capture_frame called but camera unavailable")
                return None

        # Flush a couple of frames from the open stream so a just-moved camera
        # pose is reflected in the captured image rather than returning an
        # older buffered frame from before the servo movement settled.
        for _ in range(max(0, config.CAMERA_CAPTURE_FLUSH_FRAMES)):
            with self._cap_lock:
                ok, frame = self._cap.read()
            if not ok or frame is None:
                break

        with self._cap_lock:
            ok, frame = self._cap.read()
        if not ok or frame is None:
            log.warning("Camera: frame read failed — attempting reopen")
            self._reopen()
            if not self._available or self._cap is None:
                return None
            with self._cap_lock:
                ok, frame = self._cap.read()
            if not ok or frame is None:
                log.warning("Camera: frame read failed after reopen")
                return None

        frame = cv2.convertScaleAbs(
            frame,
            alpha=config.CAMERA_BRIGHTNESS_GAIN,
            beta=config.CAMERA_BRIGHTNESS_OFFSET,
        )

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.VISION_JPEG_QUALITY]
        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            log.warning("Camera: JPEG encode failed")
            return None

        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        log.debug("Camera: frame captured successfully (%d bytes b64)", len(b64))
        return b64

    def get_tracking_frame(self) -> Optional[np.ndarray]:
        """Capture one raw frame resized to HEAD_TRACKING_RESOLUTION.

        Returns a BGR numpy array ready for cv2.cvtColor / face detection,
        or None if the camera is unavailable or the read fails.

        Thread-safe — shares the same VideoCapture as capture_frame() and
        serialises reads with _cap_lock so the two paths never interleave.
        Intentionally skips the flush loop used by capture_frame() because
        the head tracker only needs any recent frame, not the freshest one
        from a just-settled servo pose.
        """
        if not self._available or self._cap is None:
            return None

        with self._cap_lock:
            ok, frame = self._cap.read()

        if not ok or frame is None:
            return None

        w, h = config.HEAD_TRACKING_RESOLUTION
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)

    def _reopen(self) -> None:
        """Release and reopen the camera after a runtime failure."""
        self.stop()
        self.warmup()
