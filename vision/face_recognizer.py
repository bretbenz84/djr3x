"""
vision/face_recognizer.py — Face detection and recognition for DJ-R3X.

Uses dlib directly (no face_recognition wrapper) for full control over
CPU-only inference on Pi 4.

Pipeline per frame:
    base64 JPEG → numpy BGR → RGB → dlib detector → largest rect →
    shape predictor (68 landmarks) → ResNet encoder (128-d) → FaceDB lookup

Typical usage:
    recognizer = FaceRecognizer(db)
    recognizer.warmup()          # loads models; safe to call at startup
    ...
    result = recognizer.identify(frame_b64)
    if result:
        person_id, name, distance = result
        db.update_last_seen(person_id)

Graceful degradation:
    If either model file is missing, is_available() returns False and both
    encode_face() and identify() return None without raising — the rest of
    the program continues normally.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import cv2
import numpy as np

import config
from vision.face_db import FaceDB

log = logging.getLogger(__name__)


class FaceRecognizer:
    """Detects and encodes faces using dlib; matches them against FaceDB."""

    def __init__(self, db: FaceDB) -> None:
        self._db = db
        self._detector = None
        self._shape_predictor = None
        self._face_encoder = None
        self._available = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Load all dlib models into memory.

        Checks that both .dat files exist before attempting to load — missing
        files produce a clear warning rather than a cryptic dlib exception.
        Safe to call once at startup.
        """
        import dlib  # local import keeps startup fast when vision is disabled

        sp_path = config.DLIB_SHAPE_PREDICTOR_PATH
        fm_path = config.DLIB_FACE_MODEL_PATH

        if not sp_path.exists():
            log.warning(
                "FaceRecognizer: shape predictor not found at %s — recognition disabled",
                sp_path,
            )
            return
        if not fm_path.exists():
            log.warning(
                "FaceRecognizer: face model not found at %s — recognition disabled",
                fm_path,
            )
            return

        try:
            self._detector = dlib.get_frontal_face_detector()
            self._shape_predictor = dlib.shape_predictor(str(sp_path))
            self._face_encoder = dlib.face_recognition_model_v1(str(fm_path))
            self._available = True
            log.info(
                "FaceRecognizer: models loaded (detector + shape predictor + ResNet encoder)"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("FaceRecognizer: model load failed — %s", exc)

    def is_available(self) -> bool:
        """True if all models loaded successfully."""
        return self._available

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _b64_to_rgb(self, image_b64: str) -> Optional[np.ndarray]:
        """Decode a base64 JPEG string to an RGB numpy array (H×W×3, uint8)."""
        try:
            raw = base64.b64decode(image_b64)
            buf = np.frombuffer(raw, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                log.warning("FaceRecognizer: failed to decode image buffer")
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:  # noqa: BLE001
            log.warning("FaceRecognizer: image decode error — %s", exc)
            return None

    def _largest_rect(self, rects) -> Optional[object]:
        """Return the dlib rectangle with the greatest area, or None."""
        if len(rects) == 0:
            return None
        return max(rects, key=lambda r: r.width() * r.height())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_face(self, image_b64: str) -> Optional[np.ndarray]:
        """Detect the largest face in *image_b64* and return its 128-d encoding.

        Returns None if models are unavailable, no face is detected, or
        decoding fails.  All processing runs on CPU.
        """
        if not self._available:
            log.debug("FaceRecognizer: encode_face called but models not loaded")
            return None

        rgb = self._b64_to_rgb(image_b64)
        if rgb is None:
            return None

        # upsample=0 means no upsampling — fastest setting for Pi 4.
        # Use upsample=1 only if small/distant faces are being missed and
        # latency budget allows (~4× slower on Pi 4).
        log.info(
            "FaceRecognizer: encode_face — frame shape %s, running HOG detector",
            rgb.shape,
        )
        rects = self._detector(rgb, 0)
        rect = self._largest_rect(rects)
        if rect is None:
            log.info(
                "FaceRecognizer: no face detected in frame (shape=%s, %d rect(s) from detector)",
                rgb.shape, len(rects),
            )
            return None

        shape = self._shape_predictor(rgb, rect)
        # num_jitters=1 is the fastest setting; increase only when enrolment
        # accuracy matters more than speed (each jitter ≈+100 ms on Pi 4).
        raw_encoding = self._face_encoder.compute_face_descriptor(rgb, shape, num_jitters=1)
        encoding = np.array(raw_encoding, dtype=np.float64)
        log.debug("FaceRecognizer: encoded face (rect=%s)", rect)
        return encoding

    def identify(
        self,
        image_b64: str,
        tolerance: float = 0.6,
    ) -> Optional[tuple[int, str, float]]:
        """Encode the largest face in *image_b64* and search FaceDB.

        Returns (person_id, name, distance) if a match within *tolerance* is
        found, otherwise None.  Always logs the closest-match distance so
        near-misses are visible even when they fall just outside tolerance.
        """
        encoding = self.encode_face(image_b64)
        if encoding is None:
            return None

        # Log closest match unconditionally so we can see near-misses.
        closest = self._db.find_closest(encoding)
        if closest is None:
            log.info("FaceRecognizer: identify — database empty, nothing to match against")
            return None

        _cid, cname, cdist = closest
        if cdist <= tolerance:
            log.info(
                "FaceRecognizer: MATCH — '%s' at distance=%.4f (tolerance=%.4f)",
                cname, cdist, tolerance,
            )
        else:
            log.info(
                "FaceRecognizer: NO MATCH — closest is '%s' at distance=%.4f (tolerance=%.4f)",
                cname, cdist, tolerance,
            )

        return self._db.find_person(encoding, tolerance=tolerance)
