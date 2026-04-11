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
import os
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

_CREATE_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id        INTEGER REFERENCES people(id) ON DELETE CASCADE,
    category         TEXT,
    key              TEXT,
    value            TEXT,
    raw_quote        TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at       TIMESTAMP,
    follow_up_after  TIMESTAMP,
    followed_up      BOOLEAN DEFAULT FALSE
);
"""


class FaceDB:
    """SQLite-backed store for known-person face encodings."""

    # Absolute path to this file's directory → project root → assets/face_db.sqlite.
    # Using __file__ guarantees the path is the same regardless of working directory.
    _DEFAULT_PATH: Path = Path(__file__).resolve().parent.parent / "assets" / "face_db.sqlite"

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = Path(db_path).resolve() if db_path else self._DEFAULT_PATH
        existed = self._path.exists()
        log.info("FaceDB: path = %s (exists=%s)", self._path, existed)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._log_startup_stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(_CREATE_PEOPLE)
            self._conn.execute(_CREATE_ENCODINGS)
            self._conn.execute(_CREATE_MEMORIES)

    def _log_startup_stats(self) -> None:
        """Log people/encoding counts so we can confirm the DB persisted correctly."""
        try:
            n_people = self._conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
            n_enc    = self._conn.execute("SELECT COUNT(*) FROM face_encodings").fetchone()[0]
            if n_people:
                rows = self._conn.execute(
                    "SELECT id, name, visit_count FROM people ORDER BY id"
                ).fetchall()
                names = ", ".join(f"'{r['name']}' (id={r['id']}, visits={r['visit_count']})" for r in rows)
                log.info(
                    "FaceDB: loaded — %d person(s), %d encoding(s): %s",
                    n_people, n_enc, names,
                )
            else:
                log.info("FaceDB: loaded — empty (0 people, 0 encodings)")
        except Exception:
            log.exception("FaceDB: startup stats query failed")

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
        enc_norm = float(np.linalg.norm(encoding))
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO people (name) VALUES (?)", (name,)
            )
            person_id: int = cur.lastrowid  # type: ignore[assignment]
            blob = self._enc_to_blob(encoding)
            self._conn.execute(
                "INSERT INTO face_encodings (person_id, encoding) VALUES (?, ?)",
                (person_id, blob),
            )
        log.info(
            "FaceDB: added person '%s' (id=%d) — encoding dim=%d norm=%.4f blob=%d bytes",
            name, person_id, len(encoding), enc_norm, len(blob),
        )
        return person_id

    def find_closest(
        self,
        encoding: np.ndarray,
    ) -> Optional[tuple[int, str, float]]:
        """Return (person_id, name, distance) for the closest stored encoding,
        regardless of tolerance.  Returns None only if the database is empty.

        Use this when you want the distance for diagnostic purposes even if the
        match would be rejected by find_person().
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

        row = self._conn.execute(
            "SELECT name FROM people WHERE id = ?", (best_person_id,)
        ).fetchone()
        return best_person_id, row["name"], best_distance

    def find_person(
        self,
        encoding: np.ndarray,
        tolerance: float = 0.6,
    ) -> Optional[tuple[int, str, float]]:
        """Compare *encoding* against every stored encoding using Euclidean distance.

        Returns (person_id, name, distance) for the closest match within
        *tolerance*, or None if no match is close enough.
        """
        closest = self.find_closest(encoding)
        if closest is None:
            log.debug("FaceDB: find_person — database is empty")
            return None

        person_id, name, distance = closest
        if distance > tolerance:
            log.info(
                "FaceDB: no match — closest is '%s' (id=%d) at distance=%.4f (tolerance=%.4f)",
                name, person_id, distance, tolerance,
            )
            return None

        log.info(
            "FaceDB: recognised '%s' (id=%d, distance=%.4f, tolerance=%.4f)",
            name, person_id, distance, tolerance,
        )
        return person_id, name, distance

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

    def delete_all_people(self) -> None:
        """Remove all people and all stored face encodings."""
        with self._conn:
            self._conn.execute("DELETE FROM face_encodings")
            self._conn.execute("DELETE FROM people")
        log.info("FaceDB: deleted all people and encodings")

    # ------------------------------------------------------------------
    # Memory API
    # ------------------------------------------------------------------

    def add_memory(
        self,
        person_id: int,
        category: str,
        key: str,
        value: str,
        raw_quote: str,
        expires_at: Optional[str] = None,
        follow_up_after: Optional[str] = None,
    ) -> int:
        """Insert a new memory for a person. Returns the new memory id."""
        with self._conn:
            cur = self._conn.execute(
                """INSERT INTO memories
                   (person_id, category, key, value, raw_quote, expires_at, follow_up_after)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (person_id, category, key, value, raw_quote, expires_at, follow_up_after),
            )
        memory_id: int = cur.lastrowid  # type: ignore[assignment]
        log.info(
            "FaceDB: added memory for person_id=%d [%s/%s]: %r (id=%d)",
            person_id, category, key, value, memory_id,
        )
        return memory_id

    def get_memories(self, person_id: int) -> list[dict]:
        """Return all non-expired memories for a person, newest first."""
        rows = self._conn.execute(
            """SELECT id, category, key, value, raw_quote, created_at,
                      expires_at, follow_up_after, followed_up
               FROM memories
               WHERE person_id = ?
                 AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
               ORDER BY created_at DESC""",
            (person_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pending_followups(self, person_id: int) -> list[dict]:
        """Return memories whose follow_up_after time has passed and haven't been asked."""
        rows = self._conn.execute(
            """SELECT id, category, key, value, raw_quote, created_at,
                      expires_at, follow_up_after
               FROM memories
               WHERE person_id = ?
                 AND follow_up_after IS NOT NULL
                 AND follow_up_after <= CURRENT_TIMESTAMP
                 AND followed_up = FALSE
                 AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
               ORDER BY follow_up_after ASC""",
            (person_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def has_memory_for_local_day(self, person_id: int, key: str, day_iso: str) -> bool:
        """Return True when a person already has a memory with *key* on *day_iso*.

        day_iso must be in YYYY-MM-DD form and is matched in local time so the
        post-greeting "what are you doing today / this weekend?" prompt does not
        repeat after the person already answered once that day.
        """
        row = self._conn.execute(
            """SELECT 1
               FROM memories
               WHERE person_id = ?
                 AND key = ?
                 AND date(created_at, 'localtime') = ?
               LIMIT 1""",
            (person_id, key, day_iso),
        ).fetchone()
        return row is not None

    def get_latest_memory_for_local_day(
        self, person_id: int, key: str, day_iso: str
    ) -> Optional[dict]:
        """Return the newest memory for *person_id*/*key* on the given local day."""
        row = self._conn.execute(
            """SELECT id, category, key, value, raw_quote, created_at,
                      expires_at, follow_up_after, followed_up
               FROM memories
               WHERE person_id = ?
                 AND key = ?
                 AND date(created_at, 'localtime') = ?
               ORDER BY created_at DESC
               LIMIT 1""",
            (person_id, key, day_iso),
        ).fetchone()
        return dict(row) if row is not None else None

    def mark_followed_up(self, memory_id: int) -> None:
        """Mark a memory as having been followed up on."""
        with self._conn:
            self._conn.execute(
                "UPDATE memories SET followed_up = TRUE WHERE id = ?", (memory_id,)
            )
        log.debug("FaceDB: marked memory id=%d as followed up", memory_id)

    def get_memories_as_context(self, person_id: int) -> str:
        """Return a single formatted string of all memories suitable for a GPT system prompt.

        Example: "Brett likes Italian food. Brett has a dog named Max."
        Returns "" if no memories exist.
        """
        person = self.get_person(person_id)
        if not person:
            return ""
        name = person["name"]
        memories = self.get_memories(person_id)
        if not memories:
            return ""
        sentences: list[str] = []
        for m in memories:
            value = m["value"].strip()
            if not value:
                continue
            # Prepend person's name if the sentence doesn't already start with it.
            if not value.lower().startswith(name.lower()):
                value = f"{name} {value}"
            if not value.endswith("."):
                value += "."
            sentences.append(value)
        return " ".join(sentences)

    def close(self) -> None:
        self._conn.close()
        log.debug("FaceDB: connection closed")
