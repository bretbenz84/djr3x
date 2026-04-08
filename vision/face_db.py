"""
vision/face_db.py — Persistent store for known people and their face encodings.

Uses SQLite (stdlib sqlite3) so no extra dependencies are needed.  Encodings
are numpy float64 arrays serialised with tobytes() / frombuffer().

Typical usage:
    db = FaceDB()
    person_id = db.add_person("Bret", encoding_array)
    result = db.find_person(unknown_encoding)
    if result:
        person_id, name, distance = result
        db.update_last_seen(person_id)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

import config

log = logging.getLogger(__name__)

_CREATE_PEOPLE = """
CREATE TABLE IF NOT EXISTS people (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    first_seen  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen   TIMESTAMP,
    visit_count INTEGER DEFAULT 1,
    notes       TEXT
);
"""

_CREATE_ENCODINGS = """
CREATE TABLE IF NOT EXISTS face_encodings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER REFERENCES people(id),
    encoding    BLOB NOT NULL,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class FaceDB:
    """SQLite-backed store for known-person face encodings."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = Path(db_path) if db_path else Path(config.FACE_DB_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        log.info("FaceDB: opened %s", self._path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(_CREATE_PEOPLE)
            self._conn.execute(_CREATE_ENCODINGS)

    @staticmethod
    def _enc_to_blob(encoding: np.ndarray) -> bytes:
        return encoding.astype(np.float64).tobytes()

    @staticmethod
    def _blob_to_enc(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float64)

    def _all_encodings(self) -> list[tuple[int, np.ndarray]]:
        """Return [(person_id, encoding), ...] for every stored encoding."""
        cur = self._conn.execute(
            "SELECT person_id, encoding FROM face_encodings"
        )
        return [(row["person_id"], self._blob_to_enc(row["encoding"])) for row in cur]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_person(self, name: str, encoding: np.ndarray) -> int:
        """Insert a new person and their first encoding. Returns the new person_id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO people (name) VALUES (?)", (name,)
            )
            person_id: int = cur.lastrowid  # type: ignore[assignment]
            self._conn.execute(
                "INSERT INTO face_encodings (person_id, encoding) VALUES (?, ?)",
                (person_id, self._enc_to_blob(encoding)),
            )
        log.info("FaceDB: added person '%s' (id=%d)", name, person_id)
        return person_id

    def find_person(
        self,
        encoding: np.ndarray,
        tolerance: float = 0.6,
    ) -> Optional[tuple[int, str, float]]:
        """Compare *encoding* against every stored encoding using Euclidean distance.

        Returns (person_id, name, distance) for the closest match within
        *tolerance*, or None if no match is close enough.
        """
        candidates = self._all_encodings()
        if not candidates:
            return None

        query = encoding.astype(np.float64)
        best_person_id: Optional[int] = None
        best_distance = float("inf")

        for person_id, stored in candidates:
            dist = float(np.linalg.norm(query - stored))
            if dist < best_distance:
                best_distance = dist
                best_person_id = person_id

        if best_person_id is None or best_distance > tolerance:
            log.debug(
                "FaceDB: no match (best distance=%.4f, tolerance=%.4f)",
                best_distance,
                tolerance,
            )
            return None

        row = self._conn.execute(
            "SELECT name FROM people WHERE id = ?", (best_person_id,)
        ).fetchone()
        name: str = row["name"]
        log.info(
            "FaceDB: recognised '%s' (id=%d, distance=%.4f)",
            name,
            best_person_id,
            best_distance,
        )
        return best_person_id, name, best_distance

    def update_last_seen(self, person_id: int) -> None:
        """Stamp last_seen to now and increment visit_count."""
        with self._conn:
            self._conn.execute(
                """UPDATE people
                   SET last_seen   = CURRENT_TIMESTAMP,
                       visit_count = visit_count + 1
                 WHERE id = ?""",
                (person_id,),
            )
        log.debug("FaceDB: updated last_seen for person id=%d", person_id)

    def add_encoding(self, person_id: int, encoding: np.ndarray) -> None:
        """Store an additional encoding for an existing person."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO face_encodings (person_id, encoding) VALUES (?, ?)",
                (person_id, self._enc_to_blob(encoding)),
            )
        log.debug("FaceDB: added encoding for person id=%d", person_id)

    def get_person(self, person_id: int) -> Optional[dict]:
        """Return a dict with name, visit_count, first_seen, last_seen, or None."""
        row = self._conn.execute(
            "SELECT name, visit_count, first_seen, last_seen FROM people WHERE id = ?",
            (person_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_people(self) -> list[dict]:
        """Return all known people ordered by name."""
        rows = self._conn.execute(
            "SELECT id, name, visit_count, first_seen, last_seen FROM people ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]

    def rename_person(self, person_id: int, new_name: str) -> None:
        """Update the display name for an existing person."""
        with self._conn:
            self._conn.execute(
                "UPDATE people SET name = ? WHERE id = ?", (new_name, person_id)
            )
        log.info("FaceDB: renamed person id=%d to '%s'", person_id, new_name)

    def delete_person(self, person_id: int) -> None:
        """Remove a person and all their encodings."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM face_encodings WHERE person_id = ?", (person_id,)
            )
            self._conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        log.info("FaceDB: deleted person id=%d", person_id)

    def close(self) -> None:
        self._conn.close()
        log.debug("FaceDB: connection closed")
