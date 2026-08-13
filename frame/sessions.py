"""Persistent local question-answer sessions with LRU limits."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from .manifest import utc_now


DEFAULT_MAX_SESSIONS = 100
DEFAULT_MAX_SESSION_SIZE = 100 * 1024 ** 2


@dataclass(frozen=True)
class Session:
    id: str
    language: str
    fingerprints: list[str]
    top_k: int
    created_at: str
    last_used_at: str


class SessionStore:
    def __init__(self, path: Path, max_sessions: int = DEFAULT_MAX_SESSIONS,
                 max_size: int = DEFAULT_MAX_SESSION_SIZE):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.max_sessions = max_sessions
        self.max_size = max_size
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                fingerprints TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def create(self, language: str, fingerprints: list[str], top_k: int) -> Session:
        now = utc_now()
        session = Session(uuid.uuid4().hex[:12], language, fingerprints, top_k, now, now)
        with self.connection:
            self.connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
                (session.id, language, json.dumps(fingerprints), top_k, now, now),
            )
        self.enforce_limits(protected_id=session.id)
        return session

    def get(self, session_id: str) -> Session | None:
        row = self.connection.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_session(row) if row else None

    def list(self) -> list[Session]:
        return [_row_session(row) for row in self.connection.execute(
            "SELECT * FROM sessions ORDER BY last_used_at DESC, id"
        )]

    def delete(self, session_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    def recent_turns(self, session_id: str, limit: int = 6) -> list[tuple[str, str]]:
        rows = self.connection.execute("""
            SELECT question, answer FROM turns WHERE session_id = ?
            ORDER BY id DESC LIMIT ?
        """, (session_id, limit)).fetchall()
        return [(row["question"], row["answer"]) for row in reversed(rows)]

    def turn_count(self, session_id: str) -> int:
        return self.connection.execute(
            "SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,)
        ).fetchone()[0]

    def add_turn(self, session_id: str, question: str, answer: str):
        now = utc_now()
        with self.connection:
            self.connection.execute(
                "INSERT INTO turns(session_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
                (session_id, question, answer, now),
            )
            self.connection.execute(
                "UPDATE sessions SET last_used_at = ? WHERE id = ?", (now, session_id)
            )
        self.enforce_limits(protected_id=session_id)

    def enforce_limits(self, protected_id: str | None = None):
        while self._count() > self.max_sessions or self._database_size() > self.max_size:
            row = self.connection.execute(
                "SELECT id FROM sessions WHERE id != ? ORDER BY last_used_at, id LIMIT 1",
                (protected_id or "",),
            ).fetchone()
            if not row:
                row = self.connection.execute(
                    "SELECT id FROM sessions ORDER BY last_used_at, id LIMIT 1"
                ).fetchone()
            if not row:
                break
            self.delete(row["id"])
            self.connection.execute("VACUUM")
            self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _count(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    def _database_size(self) -> int:
        return sum(path.stat().st_size for path in (
            self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")
        ) if path.exists())


def _row_session(row) -> Session:
    return Session(
        id=row["id"], language=row["language"],
        fingerprints=json.loads(row["fingerprints"]), top_k=row["top_k"],
        created_at=row["created_at"], last_used_at=row["last_used_at"],
    )
