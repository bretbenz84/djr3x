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

import difflib
import logging
import os
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

import config

log = logging.getLogger(__name__)


def _normalize_name(value: str) -> str:
    """Return a lowercase alphanumeric form suitable for name matching."""
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

_CREATE_PEOPLE = """
CREATE TABLE IF NOT EXISTS people (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    first_seen  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen   TIMESTAMP,
    visit_count INTEGER DEFAULT 1,
    daily_visit_date TEXT,
    daily_visit_count INTEGER DEFAULT 0,
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
    question_text    TEXT,
    answer_text      TEXT,
    tags             TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at       TIMESTAMP,
    follow_up_after  TIMESTAMP,
    followed_up      BOOLEAN DEFAULT FALSE
);
"""

_CREATE_INTERACTION_STATS = """
CREATE TABLE IF NOT EXISTS interaction_stats (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id              INTEGER REFERENCES people(id),
    day_iso                TEXT NOT NULL,
    interaction_count      INTEGER DEFAULT 0,
    repeated_command_count INTEGER DEFAULT 0,
    repeated_topic_count   INTEGER DEFAULT 0,
    last_command           TEXT,
    last_topic             TEXT,
    last_interaction_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(person_id, day_iso)
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
            self._conn.execute(_CREATE_INTERACTION_STATS)
        self._ensure_people_columns()
        self._ensure_memories_columns()

    def _ensure_people_columns(self) -> None:
        """Backfill newer people-table columns when upgrading an existing DB."""
        cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(people)").fetchall()
        }
        with self._conn:
            if "daily_visit_date" not in cols:
                self._conn.execute("ALTER TABLE people ADD COLUMN daily_visit_date TEXT")
            if "daily_visit_count" not in cols:
                self._conn.execute(
                    "ALTER TABLE people ADD COLUMN daily_visit_count INTEGER DEFAULT 0"
                )

    def _ensure_memories_columns(self) -> None:
        """Backfill newer memories-table columns when upgrading an existing DB."""
        cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        with self._conn:
            if "question_text" not in cols:
                self._conn.execute("ALTER TABLE memories ADD COLUMN question_text TEXT")
            if "answer_text" not in cols:
                self._conn.execute("ALTER TABLE memories ADD COLUMN answer_text TEXT")
            if "tags" not in cols:
                self._conn.execute("ALTER TABLE memories ADD COLUMN tags TEXT")

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

    def update_last_seen(self, person_id: int) -> int:
        """Stamp last_seen to now, increment visit_count, and return today's count."""
        with self._conn:
            self._conn.execute(
                """UPDATE people
                   SET last_seen   = CURRENT_TIMESTAMP,
                       visit_count = visit_count + 1,
                       daily_visit_count = CASE
                           WHEN daily_visit_date = date('now', 'localtime')
                           THEN daily_visit_count + 1
                           ELSE 1
                       END,
                       daily_visit_date = date('now', 'localtime')
                 WHERE id = ?""",
                (person_id,),
            )
        row = self._conn.execute(
            "SELECT daily_visit_count FROM people WHERE id = ?",
            (person_id,),
        ).fetchone()
        daily_count = int(row["daily_visit_count"]) if row is not None else 1
        log.debug(
            "FaceDB: updated last_seen for person id=%d (daily_visit_count=%d)",
            person_id,
            daily_count,
        )
        return daily_count

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
            """SELECT name, visit_count, first_seen, last_seen,
                      daily_visit_date, daily_visit_count
               FROM people
               WHERE id = ?""",
            (person_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_people(self) -> list[dict]:
        """Return all known people ordered by name."""
        rows = self._conn.execute(
            """SELECT id, name, visit_count, first_seen, last_seen,
                      daily_visit_date, daily_visit_count
               FROM people
               ORDER BY name"""
        ).fetchall()
        return [dict(r) for r in rows]

    def find_person_by_name(self, spoken_name: str) -> Optional[tuple[int, str, float]]:
        """Return the closest stored person for a spoken/display name.

        Matching is exact/substring first on a lightly normalized form, then a
        fuzzy fallback so transcribed names can still resolve when punctuation
        or a word is slightly off.
        """
        normalized_query = _normalize_name(spoken_name)
        if not normalized_query:
            return None

        rows = self._conn.execute(
            "SELECT id, name FROM people ORDER BY name"
        ).fetchall()
        if not rows:
            return None

        normalized_rows = [
            (row["id"], row["name"], _normalize_name(row["name"]))
            for row in rows
        ]

        for person_id, name, normalized_name in normalized_rows:
            if normalized_query == normalized_name:
                return person_id, name, 1.0

        for person_id, name, normalized_name in normalized_rows:
            if normalized_query in normalized_name or normalized_name in normalized_query:
                return person_id, name, 0.9

        choices = [normalized_name for _, _, normalized_name in normalized_rows if normalized_name]
        candidates = difflib.get_close_matches(normalized_query, choices, n=1, cutoff=0.72)
        if not candidates:
            return None

        best = candidates[0]
        for person_id, name, normalized_name in normalized_rows:
            if normalized_name == best:
                score = difflib.SequenceMatcher(None, normalized_query, normalized_name).ratio()
                return person_id, name, score
        return None

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
                "DELETE FROM interaction_stats WHERE person_id = ?", (person_id,)
            )
            self._conn.execute(
                "DELETE FROM memories WHERE person_id = ?", (person_id,)
            )
            self._conn.execute(
                "DELETE FROM face_encodings WHERE person_id = ?", (person_id,)
            )
            self._conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        log.info("FaceDB: deleted person id=%d", person_id)

    def delete_people_by_name(self, name: str) -> list[int]:
        """Remove every person row whose normalized name matches *name*."""
        normalized_query = _normalize_name(name)
        if not normalized_query:
            return []

        rows = self._conn.execute(
            "SELECT id, name FROM people ORDER BY id"
        ).fetchall()
        matching_ids = [
            row["id"]
            for row in rows
            if _normalize_name(row["name"]) == normalized_query
        ]
        if not matching_ids:
            return []

        with self._conn:
            for person_id in matching_ids:
                self._conn.execute(
                    "DELETE FROM interaction_stats WHERE person_id = ?", (person_id,)
                )
                self._conn.execute(
                    "DELETE FROM memories WHERE person_id = ?", (person_id,)
                )
                self._conn.execute(
                    "DELETE FROM face_encodings WHERE person_id = ?", (person_id,)
                )
                self._conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        log.info(
            "FaceDB: deleted %d person row(s) for name %r: ids=%s",
            len(matching_ids),
            name,
            matching_ids,
        )
        return matching_ids

    def delete_all_people(self) -> None:
        """Remove all people and all stored face encodings."""
        with self._conn:
            self._conn.execute("DELETE FROM interaction_stats")
            self._conn.execute("DELETE FROM memories")
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
        question_text: Optional[str] = None,
        answer_text: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> int:
        """Insert a new memory for a person. Returns the new memory id."""
        with self._conn:
            cur = self._conn.execute(
                """INSERT INTO memories
                   (
                       person_id, category, key, value, raw_quote,
                       question_text, answer_text, tags,
                       expires_at, follow_up_after
                   )
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    person_id, category, key, value, raw_quote,
                    question_text, answer_text, tags,
                    expires_at, follow_up_after,
                ),
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
            """SELECT id, category, key, value, raw_quote, question_text,
                      answer_text, tags, created_at,
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
            """SELECT id, category, key, value, raw_quote, question_text,
                      answer_text, tags, created_at,
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
            """SELECT id, category, key, value, raw_quote, question_text,
                      answer_text, tags, created_at,
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

    def get_latest_memory_for_local_range(
        self, person_id: int, key: str, start_day_iso: str, end_day_iso: str
    ) -> Optional[dict]:
        """Return the newest memory for *person_id*/*key* within a local date range."""
        row = self._conn.execute(
            """SELECT id, category, key, value, raw_quote, question_text,
                      answer_text, tags, created_at,
                      expires_at, follow_up_after, followed_up
               FROM memories
               WHERE person_id = ?
                 AND key = ?
                 AND date(created_at, 'localtime') >= ?
                 AND date(created_at, 'localtime') <= ?
               ORDER BY created_at DESC
               LIMIT 1""",
            (person_id, key, start_day_iso, end_day_iso),
        ).fetchone()
        return dict(row) if row is not None else None

    def mark_followed_up(self, memory_id: int) -> None:
        """Mark a memory as having been followed up on."""
        with self._conn:
            self._conn.execute(
                "UPDATE memories SET followed_up = TRUE WHERE id = ?", (memory_id,)
            )
        log.debug("FaceDB: marked memory id=%d as followed up", memory_id)

    def get_asked_interview_questions(self, person_id: int) -> set[str]:
        """Return the set of interview question texts already asked to a person.

        Questions are stamped with category='interview_question' when asked
        during enrollment or follow-up interview sessions.
        """
        rows = self._conn.execute(
            """SELECT key FROM memories
               WHERE person_id = ? AND category = 'interview_question'""",
            (person_id,),
        ).fetchall()
        return {row["key"] for row in rows}

    def stamp_interview_question(self, person_id: int, question: str) -> None:
        """Record that *question* has been asked to *person_id*.

        Idempotent — calling twice for the same question is harmless because
        the query in get_asked_interview_questions returns a set.
        """
        with self._conn:
            self._conn.execute(
                """INSERT INTO memories (person_id, category, key, value, raw_quote)
                   VALUES (?, 'interview_question', ?, 'asked', '')""",
                (person_id, question),
            )
        log.debug("FaceDB: stamped interview question for person_id=%d: %r", person_id, question)

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
            if m.get("category") == "curiosity" and m.get("answer_text"):
                question_text = str(m.get("question_text") or "").strip().rstrip("?!.")
                answer_text = str(m.get("answer_text") or "").strip()
                if answer_text:
                    if question_text:
                        value = f"once told Rex, when asked {question_text.lower()}, that {answer_text}"
                    else:
                        value = f"once told Rex that {answer_text}"
                else:
                    value = ""
            else:
                value = str(m.get("value") or "").strip()
            if not value:
                continue
            # Prepend person's name if the sentence doesn't already start with it.
            if not value.lower().startswith(name.lower()):
                value = f"{name} {value}"
            if not value.endswith("."):
                value += "."
            sentences.append(value)
        return " ".join(sentences)

    def record_interaction(
        self,
        person_id: int,
        *,
        command_key: Optional[str] = None,
        topic_key: Optional[str] = None,
    ) -> dict:
        """Update daily interaction counters for a known person.

        Returns a snapshot containing today's interaction counts and whether
        this turn repeated the previous command or topic.
        """
        day_iso = date.today().isoformat()
        existing = self._conn.execute(
            """SELECT interaction_count, repeated_command_count, repeated_topic_count,
                      last_command, last_topic
               FROM interaction_stats
               WHERE person_id = ? AND day_iso = ?""",
            (person_id, day_iso),
        ).fetchone()

        prev_last_command = str(existing["last_command"]) if existing and existing["last_command"] else None
        prev_last_topic = str(existing["last_topic"]) if existing and existing["last_topic"] else None
        is_repeated_command = bool(command_key and prev_last_command == command_key)
        is_repeated_topic = bool(topic_key and prev_last_topic == topic_key)

        if existing is None:
            interaction_count = 1
            repeated_command_count = 1 if is_repeated_command else 0
            repeated_topic_count = 1 if is_repeated_topic else 0
            new_last_command = command_key
            new_last_topic = topic_key
            with self._conn:
                self._conn.execute(
                    """INSERT INTO interaction_stats
                       (
                           person_id, day_iso, interaction_count,
                           repeated_command_count, repeated_topic_count,
                           last_command, last_topic
                       )
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        person_id,
                        day_iso,
                        interaction_count,
                        repeated_command_count,
                        repeated_topic_count,
                        new_last_command,
                        new_last_topic,
                    ),
                )
        else:
            interaction_count = int(existing["interaction_count"]) + 1
            repeated_command_count = int(existing["repeated_command_count"]) + (
                1 if is_repeated_command else 0
            )
            repeated_topic_count = int(existing["repeated_topic_count"]) + (
                1 if is_repeated_topic else 0
            )
            new_last_command = command_key or prev_last_command
            new_last_topic = topic_key or prev_last_topic
            with self._conn:
                self._conn.execute(
                    """UPDATE interaction_stats
                       SET interaction_count = ?,
                           repeated_command_count = ?,
                           repeated_topic_count = ?,
                           last_command = ?,
                           last_topic = ?,
                           last_interaction_at = CURRENT_TIMESTAMP
                       WHERE person_id = ? AND day_iso = ?""",
                    (
                        interaction_count,
                        repeated_command_count,
                        repeated_topic_count,
                        new_last_command,
                        new_last_topic,
                        person_id,
                        day_iso,
                    ),
                )

        snapshot = {
            "day_iso": day_iso,
            "interaction_count": interaction_count,
            "repeated_command_count": repeated_command_count,
            "repeated_topic_count": repeated_topic_count,
            "last_command": new_last_command,
            "last_topic": new_last_topic,
            "is_repeated_command": is_repeated_command,
            "is_repeated_topic": is_repeated_topic,
        }
        log.debug("FaceDB: interaction stats updated for person_id=%d: %s", person_id, snapshot)
        return snapshot

    def get_today_interaction_stats(self, person_id: int) -> Optional[dict]:
        """Return today's interaction stats snapshot for a known person."""
        day_iso = date.today().isoformat()
        row = self._conn.execute(
            """SELECT day_iso, interaction_count, repeated_command_count,
                      repeated_topic_count, last_command, last_topic,
                      last_interaction_at
               FROM interaction_stats
               WHERE person_id = ? AND day_iso = ?""",
            (person_id, day_iso),
        ).fetchone()
        return dict(row) if row is not None else None

    def close(self) -> None:
        self._conn.close()
        log.debug("FaceDB: connection closed")
