"""Transcripts kept in SQLite, each with the UTC time its audio was recorded."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY,
    recorded_at TEXT NOT NULL,
    text TEXT NOT NULL,
    language TEXT NOT NULL,
    duration REAL NOT NULL,
    words TEXT
);
CREATE INDEX IF NOT EXISTS transcripts_recorded_at ON transcripts (recorded_at);
"""


def timestamp(value: str | None = None) -> str:
    """An ISO 8601 time as UTC to the second, e.g. 2026-10-08T22:10:03Z, or now when value is None.

    A time without a zone is taken as UTC. Raises ValueError when value isn't ISO 8601.
    """
    moment = datetime.fromisoformat(value) if value else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record(row: sqlite3.Row) -> dict:
    record = dict(row)
    words = record.pop("words")
    if words is not None:
        record["words"] = json.loads(words)
    return record


class TranscriptStore:
    """Opens a connection per call, so it can be shared by threads and server processes."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, result: dict, recorded_at: str) -> dict:
        """Store a transcription result; returns it with its "id" and "recorded_at"."""
        words = result.get("words")
        with self._db() as db:
            cursor = db.execute(
                "INSERT INTO transcripts (recorded_at, text, language, duration, words) VALUES (?, ?, ?, ?, ?)",
                (recorded_at, result["text"], result["language"], result["duration"],
                 json.dumps(words, ensure_ascii=False) if words is not None else None))
        return {"id": cursor.lastrowid, "recorded_at": recorded_at, **result}

    def latest(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """The most recently recorded first."""
        with self._db() as db:
            rows = db.execute("SELECT * FROM transcripts ORDER BY recorded_at DESC, id DESC LIMIT ? OFFSET ?",
                              (limit, offset)).fetchall()
        return [_record(row) for row in rows]

    def get(self, transcript_id: int) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM transcripts WHERE id = ?", (transcript_id,)).fetchone()
        return _record(row) if row else None

    def delete(self, transcript_id: int) -> bool:
        with self._db() as db:
            return db.execute("DELETE FROM transcripts WHERE id = ?", (transcript_id,)).rowcount > 0
