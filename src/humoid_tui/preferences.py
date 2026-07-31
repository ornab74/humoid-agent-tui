from __future__ import annotations

import sqlite3
from pathlib import Path


class PreferenceStore:
    """Small durable store for user choices, independent of agent memory."""

    def __init__(self, path: Path = Path(".humoid/preferences.sqlite3")) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as database:
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            database.commit()

    def values(self) -> dict[str, str]:
        with sqlite3.connect(self.path) as database:
            rows = database.execute("SELECT key, value FROM preferences").fetchall()
        return {str(key): str(value) for key, value in rows}

    def set(self, key: str, value: object) -> None:
        normalized = key.strip().upper()
        with sqlite3.connect(self.path) as database:
            database.execute(
                """
                INSERT INTO preferences (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (normalized, str(value)),
            )
            database.commit()
