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
import logging
from typing import Optional

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)


class Camera:
    """Manages a single webcam device for still-frame capture."""

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._available: bool = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Open the camera device and verify it can deliver frames.

        Safe to call once at startup.  Sets is_available() based on whether
        the device opened successfully.
        """
        cap = cv2.VideoCapture(config.CAMERA_DEVICE_INDEX)
        if not cap.isOpened():
            log.warning(
                "Camera: device %d could not be opened — vision disabled",
                config.CAMERA_DEVICE_INDEX,
            )
            cap.release()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Read one frame to confirm the device is actually delivering data.
        ok, _ = cap.read()
        if not ok:
            log.warning(
                "Camera: device %d opened but returned no frame — vision disabled",
                config.CAMERA_DEVICE_INDEX,
            )
            cap.release()
            return

        self._cap = cap
        self._available = True
        log.info(
            "Camera: device %d ready (%dx%d)",
            config.CAMERA_DEVICE_INDEX,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

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
            log.debug("Camera: capture_frame called but camera unavailable")
            return None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            log.warning("Camera: frame read failed")
            return None

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.VISION_JPEG_QUALITY]
        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            log.warning("Camera: JPEG encode failed")
            return None

        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        log.debug("Camera: frame captured successfully (%d bytes b64)", len(b64))
        return b64
