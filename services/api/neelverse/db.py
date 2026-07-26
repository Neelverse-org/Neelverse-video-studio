import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id),
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id),
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stats_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    output_path TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user_created
                    ON sessions(user_id, created_at DESC);
                """
            )

    def create_admin(self, user_id: str, username: str, password_hash: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users(id, username, password_hash, is_admin, created_at)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(username) DO NOTHING
                """,
                (user_id, username, password_hash, datetime.now(UTC).isoformat()),
            )

    def user_by_username(self, username: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, password_hash, is_admin FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        return dict(row) if row else None

    def user_by_id(self, user_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, is_admin FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def create_asset(self, asset: dict[str, Any]) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO assets(id, user_id, filename, content_type, size_bytes, path, created_at)
                VALUES (:id, :user_id, :filename, :content_type, :size_bytes, :path, :created_at)
                """,
                asset,
            )

    def asset_for_user(self, asset_id: UUID | str, user_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM assets WHERE id = ? AND user_id = ?",
                (str(asset_id), user_id),
            ).fetchone()
        return dict(row) if row else None

    def create_session(self, session_id: UUID, user_id: str, config: dict[str, Any]) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions(id, user_id, config_json, status, created_at)
                VALUES (?, ?, ?, 'queued', ?)
                """,
                (str(session_id), user_id, json.dumps(config), datetime.now(UTC).isoformat()),
            )

    def update_session(self, session_id: UUID | str, **fields: Any) -> None:
        allowed = {"status", "stats_json", "error", "output_path", "started_at", "ended_at", "config_json"}
        values = {key: value for key, value in fields.items() if key in allowed}
        if not values:
            return
        for key in ("stats_json", "config_json"):
            if key in values and not isinstance(values[key], str):
                values[key] = json.dumps(values[key])
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"UPDATE sessions SET {assignments} WHERE id = ?",  # noqa: S608
                (*values.values(), str(session_id)),
            )

    def session_for_user(self, session_id: UUID | str, user_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ? AND user_id = ?",
                (str(session_id), user_id),
            ).fetchone()
        return self._decode_session(row)

    def sessions_for_user(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [self._decode_session(row) for row in rows if row is not None]

    @staticmethod
    def _decode_session(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result["config"] = json.loads(result.pop("config_json"))
        result["stats"] = json.loads(result.pop("stats_json"))
        return result
