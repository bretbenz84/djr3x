"""
vision/face_recognizer.py — Face detection and recognition for DJ-R3X.

Uses dlib directly (no face_recognition wrapper) for full control over
CPU-only inference on Pi 4.

Pipeline per frame:
    base64 JPEG → numpy BGR → RGB → dlib HOG detector (tiered upsample) →
    largest rect → shape predictor (68 landmarks) → ResNet encoder (128-d) →
    FaceDB lookup

Upsample strategy
-----------------
Recognition  (identify):  upsample=0 only.
Enrollment   (encode_face with for_enrollment=True):  upsample=2 directly.

upsample=0 → original resolution (fastest, misses small/distant faces).
upsample=1 → 2× upsampled input   (~4× slower than 0 on Pi 4, good for typical range).
upsample=2 → 4× upsampled input   (~16× slower than 0, best for small faces).

CNN detector note
-----------------
dlib also ships a CNN-based face detector (dlib.cnn_face_detection_model_v1)
that is significantly more accurate than HOG, especially for non-frontal faces
and small faces from wide-angle lenses.  It requires the model file
"mmod_human_face_detector.dat" (available from dlib's model zoo).  Consider
switching if the HOG detector continues to struggle at typical interaction
distance with the JVU430 2.1 mm wide-angle lens.

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
from datetime import datetime
from pathlib import Path
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

        log.debug("FaceRecognizer: warmup — shape_predictor path = %s", sp_path)
        log.debug("FaceRecognizer: warmup — face_model path      = %s", fm_path)
        log.debug("FaceRecognizer: warmup — shape_predictor exists = %s", sp_path.exists())
        log.debug("FaceRecognizer: warmup — face_model exists      = %s", fm_path.exists())

        if not sp_path.exists():
            log.error(
                "FaceRecognizer: shape predictor not found at %s — recognition disabled.\n"
                "  Download: https://github.com/davisking/dlib-models/raw/master/"
                "shape_predictor_68_face_landmarks.dat.bz2\n"
                "  Then: bunzip2 shape_predictor_68_face_landmarks.dat.bz2 → assets/models/",
                sp_path,
            )
            return
        if not fm_path.exists():
            log.error(
                "FaceRecognizer: face recognition model not found at %s — recognition disabled.\n"
                "  This is the ResNet face *encoder* (not the detector).\n"
                "  Download: https://github.com/davisking/dlib-models/raw/master/"
                "dlib_face_recognition_resnet_model_v1.dat.bz2\n"
                "  Then: bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2 → assets/models/\n"
                "  Or set DLIB_FACE_MODEL_PATH in .env to override the path.",
                fm_path,
            )
            return

        try:
            log.debug("FaceRecognizer: loading HOG detector …")
            self._detector = dlib.get_frontal_face_detector()
            log.debug("FaceRecognizer: loading shape predictor from %s …", sp_path)
            self._shape_predictor = dlib.shape_predictor(str(sp_path))
            log.debug("FaceRecognizer: loading face encoder from %s …", fm_path)
            self._face_encoder = dlib.face_recognition_model_v1(str(fm_path))
            self._available = True
            log.info(
                "FaceRecognizer: models loaded (detector + shape predictor + ResNet encoder)"
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "FaceRecognizer: model load failed — %s: %s\n"
                "  shape_predictor path: %s (exists=%s)\n"
                "  face_model path:      %s (exists=%s)",
                type(exc).__name__, exc,
                sp_path, sp_path.exists(),
                fm_path, fm_path.exists(),
            )

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

    def _save_debug_frame(self, image_b64: str, mode: str) -> None:
        """Persist the exact JPEG input for later inspection."""
        try:
            debug_dir: Path = config.FACE_DEBUG_DIR
            debug_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = debug_dir / f"{stamp}_{mode}.jpg"
            path.write_bytes(base64.b64decode(image_b64))
            log.debug("FaceRecognizer: saved debug frame to %s", path)
        except Exception as exc:  # noqa: BLE001
            log.warning("FaceRecognizer: failed to save debug frame — %s", exc)

    def _largest_rect(self, rects) -> Optional[object]:
        """Return the dlib rectangle with the greatest area, or None."""
        if len(rects) == 0:
            return None
        return max(rects, key=lambda r: r.width() * r.height())

    def _detect_tiered(
        self, rgb: np.ndarray, upsample_levels: list[int]
    ) -> tuple[object, int]:
        """Run the HOG detector at each upsample level until a face is found.

        Returns (rects, level_used) where rects is the non-empty detection
        result at the first successful level, or the (empty) result from the
        last level tried if no face was found at any level.
        """
        rects = []
        last_level = upsample_levels[-1] if upsample_levels else 0
        for level in upsample_levels:
            rects = self._detector(rgb, level)
            last_level = level
            if len(rects) > 0:
                return rects, level
        return rects, last_level

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_face(
        self,
        image_b64: str,
        for_enrollment: bool = False,
    ) -> Optional[np.ndarray]:
        """Detect the largest face in *image_b64* and return its 128-d encoding.

        for_enrollment=False (recognition):
            Uses upsample=0 only. At 1080p capture resolution this keeps wake-
            time face recognition fast enough on the Pi 4.

        for_enrollment=True (enrollment):
            Always uses upsample=2 directly — accuracy matters more than
            speed since enrollment only happens once per person.

        Returns None if models are unavailable, no face is detected, or
        decoding fails.  All processing runs on CPU.
        """
        if not self._available:
            log.debug("FaceRecognizer: encode_face called but models not loaded")
            return None

        rgb = self._b64_to_rgb(image_b64)
        if rgb is None:
            return None

        frame_h = rgb.shape[0]
        upsample_levels = [2] if for_enrollment else [0]
        mode = "enrollment" if for_enrollment else "recognition"
        self._save_debug_frame(image_b64, mode)
        log.info(
            "FaceRecognizer: %s — frame %s, HOG upsample strategy %s",
            mode, rgb.shape, upsample_levels,
        )

        rects, level_used = self._detect_tiered(rgb, upsample_levels)
        rect = self._largest_rect(rects)

        if rect is None:
            log.info(
                "FaceRecognizer: no face detected (%s, tried upsample levels %s, "
                "frame shape=%s) — "
                "if faces are consistently missed, consider CNN detector "
                "(mmod_human_face_detector.dat) or moving closer to the camera",
                mode, upsample_levels, rgb.shape,
            )
            return None

        face_pct = rect.height() / frame_h * 100
        log.info(
            "FaceRecognizer: face detected at upsample=%d — rect=%s, "
            "face height=%.1f%% of frame (%s) "
            "[<10%% = too far; 10-25%% = marginal; >25%% = good]",
            level_used, rect, face_pct, mode,
        )

        shape = self._shape_predictor(rgb, rect)
        # num_jitters=1 is the fastest setting for recognition.
        # For enrollment, accuracy matters more; consider num_jitters=3-5 if
        # recognition distances are inconsistent (each jitter ≈+100 ms on Pi 4).
        raw_encoding = self._face_encoder.compute_face_descriptor(rgb, shape, num_jitters=1)
        encoding = np.array(raw_encoding, dtype=np.float64)
        log.debug("FaceRecognizer: encoding complete (rect=%s, mode=%s)", rect, mode)
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
        encoding = self.encode_face(image_b64, for_enrollment=False)
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

    def identify_with_status(
        self,
        image_b64: str,
        tolerance: float = 0.6,
    ) -> tuple[str, Optional[tuple[int, str, float]]]:
        """Return a statusful identification result for wake-routing decisions.

        Status values:
          - "match": face detected and matched within tolerance
          - "no_match": face detected but not recognized
          - "no_face": image decoded but no face was detected
          - "db_empty": face detected but FaceDB has no people
          - "unavailable": recognizer models are not loaded
        """
        if not self._available:
            return "unavailable", None

        encoding = self.encode_face(image_b64, for_enrollment=False)
        if encoding is None:
            return "no_face", None

        closest = self._db.find_closest(encoding)
        if closest is None:
            log.info("FaceRecognizer: identify — database empty, nothing to match against")
            return "db_empty", None

        _cid, cname, cdist = closest
        if cdist <= tolerance:
            log.info(
                "FaceRecognizer: MATCH — '%s' at distance=%.4f (tolerance=%.4f)",
                cname, cdist, tolerance,
            )
            return "match", self._db.find_person(encoding, tolerance=tolerance)

        log.info(
            "FaceRecognizer: NO MATCH — closest is '%s' at distance=%.4f (tolerance=%.4f)",
            cname, cdist, tolerance,
        )
        return "no_match", None
