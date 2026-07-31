from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import Settings
from .models import MemoryHit, utcnow


_FILE_PATTERN = re.compile(
    r"(?<![\w.-])(?:[\w.-]+/)*[\w.-]+\.(?:py|js|ts|tsx|jsx|dart|go|rs|java|kt|"
    r"c|cc|cpp|h|hpp|cs|rb|php|swift|scala|sh|sql|html|css|scss|json|ya?ml|toml|md)\b",
    re.IGNORECASE,
)


def _file_references(text: str) -> set[str]:
    return {match.group(0) for match in _FILE_PATTERN.finditer(text)}


def _rank_memory(query: str, hit: MemoryHit) -> float:
    """Provider-independent ranking tuned for coding and durable facts."""
    query_lower = query.lower()
    text_lower = hit.text.lower()
    score = float(hit.score) + (3.0 * _lexical_score(query, hit.text))

    query_files = _file_references(query)
    text_files = _file_references(hit.text)
    score += 2.5 * len(query_files & text_files)

    query_symbols = {
        token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)
        if "_" in token or any(character.isupper() for character in token[1:])
    }
    score += 0.8 * sum(symbol.lower() in text_lower for symbol in query_symbols)

    if hit.validation_status == "verified":
        score += 0.4
    elif hit.validation_status == "observed":
        score += 0.2
    elif hit.validation_status in {"invalid", "rejected"}:
        score -= 0.8

    failure = any(marker in text_lower for marker in (
        "provider stream failed", "validation errors", "internal_server_error"
    ))
    debugging = any(word in query_lower for word in ("error", "fail", "debug", "traceback"))
    if failure and not debugging:
        score -= 2.0

    if any(marker in text_lower for marker in ("tests pass", "fixed", "created", "wrote ")):
        score += 0.15

    try:
        created = datetime.fromisoformat(hit.created_at.replace("Z", "+00:00"))
        age_days = max(0.0, (datetime.now(timezone.utc) - created).total_seconds() / 86400)
        score += 0.25 / (1.0 + age_days / 30.0)
    except (TypeError, ValueError):
        pass

    return score


def _tokens(text: str) -> set[str]:
    return set(
        re.findall(
            r"[a-z0-9_]{2,}",
            text.lower(),
        )
    )


def _lexical_score(
    query: str,
    text: str,
) -> float:
    query_tokens = _tokens(query)
    text_tokens = _tokens(text)

    if not query_tokens or not text_tokens:
        return 0.0

    intersection = len(
        query_tokens & text_tokens
    )

    denominator = math.sqrt(
        len(query_tokens) * len(text_tokens)
    )

    return intersection / max(
        1.0,
        denominator,
    )


def _safe_metadata_json(
    value: Any,
) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if not value:
        return {}

    try:
        decoded = json.loads(str(value))

        if isinstance(decoded, dict):
            return decoded

    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    return {}


class SQLiteMemory:
    """
    Local SQLite memory fallback.

    This backend uses lexical retrieval and requires no external
    process.
    """

    def __init__(
        self,
        path: Path,
    ) -> None:
        self.path = path

    async def initialize(self) -> None:
        self._initialize_sync()

    def _initialize_sync(self) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with sqlite3.connect(self.path) as database:
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    memory_tier TEXT,
                    channel TEXT,
                    task_id TEXT,
                    validation_status TEXT,
                    created_at TEXT,
                    metadata TEXT
                )
                """
            )

            database.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_memories_created_at
                ON memories(created_at)
                """
            )

            database.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_memories_channel
                ON memories(channel)
                """
            )

            database.commit()

    async def add(
        self,
        text: str,
        **meta: Any,
    ) -> str:
        return self._add_sync(text, meta)

    def _add_sync(self, text: str, meta: dict[str, Any]) -> str:
        memory_id = str(uuid4())

        with sqlite3.connect(self.path) as database:
            database.execute(
                """
                INSERT INTO memories (
                    id,
                    text,
                    memory_tier,
                    channel,
                    task_id,
                    validation_status,
                    created_at,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    text,
                    meta.get(
                        "memory_tier",
                        "episodic",
                    ),
                    meta.get(
                        "channel",
                        "agent",
                    ),
                    meta.get(
                        "task_id",
                        "",
                    ),
                    meta.get(
                        "validation_status",
                        "unverified",
                    ),
                    utcnow(),
                    json.dumps(
                        meta.get(
                            "metadata",
                            {},
                        ),
                        ensure_ascii=False,
                    ),
                ),
            )

            database.commit()

        return memory_id

    async def search(
        self,
        query: str,
        limit: int = 8,
    ) -> list[MemoryHit]:
        return self._search_sync(query, limit)

    def _search_sync(self, query: str, limit: int) -> list[MemoryHit]:
        with sqlite3.connect(self.path) as database:
            cursor = database.execute(
                """
                SELECT
                    id,
                    text,
                    memory_tier,
                    channel,
                    task_id,
                    validation_status,
                    created_at,
                    metadata
                FROM memories
                """
            )

            rows = cursor.fetchall()

        hits: list[MemoryHit] = []

        for row in rows:
            score = _lexical_score(
                query,
                row[1],
            )

            if score <= 0:
                continue

            hits.append(
                MemoryHit(
                    memory_id=row[0],
                    text=row[1],
                    score=score,
                    memory_tier=(
                        row[2] or "episodic"
                    ),
                    channel=(
                        row[3] or "agent"
                    ),
                    task_id=(
                        row[4] or ""
                    ),
                    validation_status=(
                        row[5] or "unverified"
                    ),
                    created_at=(
                        row[6] or ""
                    ),
                    metadata=(
                        _safe_metadata_json(
                            row[7]
                        )
                    ),
                )
            )

        hits.sort(
            key=lambda hit: _rank_memory(query, hit),
            reverse=True,
        )

        return hits[:limit]

    async def temporal_neighbors(
        self,
        hit: MemoryHit,
        radius: int = 3,
    ) -> list[MemoryHit]:
        return self._temporal_neighbors_sync(hit, radius)

    def _temporal_neighbors_sync(self, hit: MemoryHit, radius: int) -> list[MemoryHit]:
        if radius <= 0:
            return []

        with sqlite3.connect(self.path) as database:
            cursor = database.execute(
                """
                SELECT
                    id,
                    text,
                    memory_tier,
                    channel,
                    task_id,
                    validation_status,
                    created_at,
                    metadata
                FROM memories
                ORDER BY created_at ASC
                """
            )

            rows = cursor.fetchall()

        memory_ids = [
            row[0]
            for row in rows
        ]

        if hit.memory_id not in memory_ids:
            return []

        index = memory_ids.index(
            hit.memory_id
        )

        start = max(
            0,
            index - radius,
        )

        end = min(
            len(rows),
            index + radius + 1,
        )

        selected_rows = [
            row
            for position, row in enumerate(
                rows[start:end],
                start=start,
            )
            if position != index
        ]

        return [
            MemoryHit(
                memory_id=row[0],
                text=row[1],
                score=0.0,
                memory_tier=(
                    row[2] or "episodic"
                ),
                channel=(
                    row[3] or "agent"
                ),
                task_id=(
                    row[4] or ""
                ),
                validation_status=(
                    row[5] or "unverified"
                ),
                created_at=(
                    row[6] or ""
                ),
                metadata=_safe_metadata_json(
                    row[7]
                ),
            )
            for row in selected_rows
        ]

    async def list_memories(self, limit: int = 200) -> list[MemoryHit]:
        return self._list_memories_sync(limit)

    def _list_memories_sync(self, limit: int) -> list[MemoryHit]:
        with sqlite3.connect(self.path) as database:
            cursor = database.execute(
                """SELECT id, text, memory_tier, channel, task_id,
                validation_status, created_at, metadata FROM memories
                ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            )
            rows = cursor.fetchall()
        return [
            MemoryHit(
                memory_id=row[0], text=row[1], memory_tier=row[2] or "episodic",
                channel=row[3] or "agent", task_id=row[4] or "",
                validation_status=row[5] or "unverified", created_at=row[6] or "",
                metadata=_safe_metadata_json(row[7]),
            )
            for row in rows
        ]

    async def update_memory(self, memory_id: str, **changes: Any) -> None:
        self._update_memory_sync(memory_id, changes)

    def _update_memory_sync(self, memory_id: str, changes: dict[str, Any]) -> None:
        allowed = {"text", "memory_tier", "channel", "task_id", "validation_status"}
        selected = {key: value for key, value in changes.items() if key in allowed}
        if not selected:
            return
        assignments = ", ".join(f"{key} = ?" for key in selected)
        with sqlite3.connect(self.path) as database:
            database.execute(
                f"UPDATE memories SET {assignments} WHERE id = ?",
                (*selected.values(), memory_id),
            )
            database.commit()

    async def delete_memory(self, memory_id: str) -> None:
        self._delete_memory_sync(memory_id)

    def _delete_memory_sync(self, memory_id: str) -> None:
        with sqlite3.connect(self.path) as database:
            database.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            database.commit()

    async def close(self) -> None:
        # Connections are opened per operation.
        return


class WeaviateMemory:
    """
    Remote Weaviate backend.
    """

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.s = settings
        self.client = None

    def _collection(self):
        if self.client is None:
            raise RuntimeError(
                "Weaviate client is not initialized"
            )

        return self.client.collections.get(
            self.s.weaviate_collection
        )

    async def initialize(self) -> None:
        await asyncio.to_thread(
            self._initialize_sync
        )

    def _initialize_sync(self) -> None:
        try:
            import weaviate
        except ImportError as exc:
            raise RuntimeError(
                "Install Weaviate support with: "
                "pip install -U weaviate-client"
            ) from exc

        self.client = weaviate.connect_to_custom(
            http_host=self.s.weaviate_http_host,
            http_port=self.s.weaviate_http_port,
            http_secure=self.s.weaviate_secure,
            grpc_host=self.s.weaviate_grpc_host,
            grpc_port=self.s.weaviate_grpc_port,
            grpc_secure=self.s.weaviate_secure,
        )

        if not self.client.is_ready():
            raise RuntimeError(
                "Remote Weaviate is not ready"
            )

        self._ensure_collection()

        self._crud_roundtrip()

    def _ensure_collection(self) -> None:
        try:
            from weaviate.classes.config import (
                Configure,
                DataType,
                Property,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Installed weaviate-client is missing "
                "the v4 configuration API"
            ) from exc

        if self.client is None:
            raise RuntimeError(
                "Weaviate client is not initialized"
            )

        if self.client.collections.exists(
            self.s.weaviate_collection
        ):
            return

        self.client.collections.create(
            self.s.weaviate_collection,
            vectorizer_config=(
                Configure.Vectorizer.none()
            ),
            properties=[
                Property(
                    name="text",
                    data_type=DataType.TEXT,
                ),
                Property(
                    name="memory_tier",
                    data_type=DataType.TEXT,
                ),
                Property(
                    name="channel",
                    data_type=DataType.TEXT,
                ),
                Property(
                    name="task_id",
                    data_type=DataType.TEXT,
                ),
                Property(
                    name="validation_status",
                    data_type=DataType.TEXT,
                ),
                Property(
                    name="created_at",
                    data_type=DataType.DATE,
                ),
                Property(
                    name="metadata_json",
                    data_type=DataType.TEXT,
                ),
            ],
        )

    def _crud_roundtrip(self) -> None:
        """
        Third readiness gate:

        1. Insert sentinel
        2. Fetch sentinel
        3. Search sentinel
        4. Delete sentinel
        """

        collection = self._collection()

        sentinel = (
            f"humoid-health-{uuid4()}"
        )

        inserted_id = None

        try:
            inserted_id = collection.data.insert(
                {
                    "text": sentinel,
                    "memory_tier": "system",
                    "channel": "health",
                    "task_id": "health",
                    "validation_status": (
                        "verified"
                    ),
                    "created_at": utcnow(),
                    "metadata_json": "{}",
                }
            )

            fetched = (
                collection.query
                .fetch_object_by_id(
                    inserted_id
                )
            )

            if (
                fetched is None
                or fetched.properties.get(
                    "text"
                ) != sentinel
            ):
                raise RuntimeError(
                    "Weaviate CRUD read-back failed"
                )

            queried = collection.query.bm25(
                query=sentinel,
                query_properties=[
                    "text"
                ],
                limit=1,
            )

            if not queried.objects:
                raise RuntimeError(
                    "Weaviate BM25 query "
                    "round-trip failed"
                )

        finally:
            if inserted_id is not None:
                try:
                    collection.data.delete_by_id(
                        inserted_id
                    )
                except Exception:
                    pass

    async def add(
        self,
        text: str,
        **meta: Any,
    ) -> str:
        collection = self._collection()

        inserted_id = collection.data.insert(
            {
                "text": text,
                "memory_tier": meta.get(
                    "memory_tier",
                    "episodic",
                ),
                "channel": meta.get(
                    "channel",
                    "agent",
                ),
                "task_id": meta.get(
                    "task_id",
                    "",
                ),
                "validation_status": meta.get(
                    "validation_status",
                    "unverified",
                ),
                "created_at": utcnow(),
                "metadata_json": json.dumps(
                    meta.get(
                        "metadata",
                        {},
                    ),
                    ensure_ascii=False,
                ),
            }
        )

        return str(inserted_id)

    async def search(
        self,
        query: str,
        limit: int = 8,
    ) -> list[MemoryHit]:
        try:
            from weaviate.classes.query import (
                MetadataQuery,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Installed weaviate-client is missing "
                "the v4 query API"
            ) from exc

        collection = self._collection()

        result = collection.query.bm25(
            query=query,
            query_properties=[
                "text"
            ],
            # Retrieve broadly, then apply the same coding-aware ranker used
            # by SQLite. This keeps provider switches behaviorally stable.
            limit=max(limit * 4, 24),
            return_metadata=MetadataQuery(
                score=True
            ),
        )

        hits: list[MemoryHit] = []

        for obj in result.objects:
            properties = obj.properties or {}
            metadata = obj.metadata

            score = 0.0

            if metadata is not None:
                score = float(
                    metadata.score or 0.0
                )

            hits.append(
                MemoryHit(
                    memory_id=str(obj.uuid),
                    text=str(
                        properties.get(
                            "text",
                            "",
                        )
                    ),
                    score=score,
                    memory_tier=str(
                        properties.get(
                            "memory_tier",
                            "episodic",
                        )
                    ),
                    channel=str(
                        properties.get(
                            "channel",
                            "agent",
                        )
                    ),
                    task_id=str(
                        properties.get(
                            "task_id",
                            "",
                        )
                    ),
                    validation_status=str(
                        properties.get(
                            "validation_status",
                            "unverified",
                        )
                    ),
                    created_at=str(
                        properties.get(
                            "created_at",
                            "",
                        )
                    ),
                    metadata=(
                        _safe_metadata_json(
                            properties.get(
                                "metadata_json",
                                "{}",
                            )
                        )
                    ),
                )
            )

        hits.sort(key=lambda hit: _rank_memory(query, hit), reverse=True)
        return hits[:limit]

    async def temporal_neighbors(
        self,
        hit: MemoryHit,
        radius: int = 3,
    ) -> list[MemoryHit]:
        """
        Portable fallback.

        Weaviate timestamp-neighbor querying can be added later without
        changing the MemoryRouter interface.
        """

        return []

    async def list_memories(self, limit: int = 200) -> list[MemoryHit]:
        return await asyncio.to_thread(self._list_memories_sync, limit)

    def _list_memories_sync(self, limit: int) -> list[MemoryHit]:
        result = self._collection().query.fetch_objects(limit=limit)
        hits: list[MemoryHit] = []
        for obj in result.objects:
            properties = obj.properties or {}
            hits.append(MemoryHit(
                memory_id=str(obj.uuid), text=str(properties.get("text", "")),
                memory_tier=str(properties.get("memory_tier", "episodic")),
                channel=str(properties.get("channel", "agent")),
                task_id=str(properties.get("task_id", "")),
                validation_status=str(properties.get("validation_status", "unverified")),
                created_at=str(properties.get("created_at", "")),
                metadata=_safe_metadata_json(properties.get("metadata_json", "{}")),
            ))
        return hits

    async def update_memory(self, memory_id: str, **changes: Any) -> None:
        allowed = {"text", "memory_tier", "channel", "task_id", "validation_status"}
        selected = {key: value for key, value in changes.items() if key in allowed}
        if selected:
            await asyncio.to_thread(self._collection().data.update, uuid=memory_id, properties=selected)

    async def delete_memory(self, memory_id: str) -> None:
        await asyncio.to_thread(self._collection().data.delete_by_id, memory_id)

    async def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None


class EmbeddedWeaviateMemory(
    WeaviateMemory
):
    """
    Embedded Weaviate backend.

    It starts a native Weaviate process from Python and stores data in
    the configured persistence directory. Docker is not required.
    """

    async def initialize(self) -> None:
        await asyncio.to_thread(
            self._initialize_sync
        )

    connection_mode = "embedded"

    def _adopt_existing(self, weaviate: Any) -> bool:
        """Adopt a healthy server on the embedded ports without owning it."""

        candidate = None
        try:
            candidate = weaviate.connect_to_local(
                host="127.0.0.1",
                port=self.s.weaviate_embedded_http_port,
                grpc_port=self.s.weaviate_embedded_grpc_port,
            )
            if not candidate.is_ready():
                candidate.close()
                return False
        except Exception:
            if candidate is not None:
                try:
                    candidate.close()
                except Exception:
                    pass
            return False

        self.client = candidate
        self.connection_mode = "adopted-existing"
        self._ensure_collection()
        self._crud_roundtrip()
        return True

    def _initialize_sync(self) -> None:
        try:
            import weaviate
        except ImportError as exc:
            raise RuntimeError(
                "Install Embedded Weaviate with: "
                "pip install -U weaviate-client"
            ) from exc

        # A prior Humoid process may have left a healthy embedded server
        # running. Validate and reuse it before attempting another launch.
        if self._adopt_existing(weaviate):
            return

        self.s.weaviate_embedded_data_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.s.weaviate_embedded_binary_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        environment_variables = {
            "LOG_LEVEL": (
                self.s.weaviate_embedded_log_level
            ),
            "DISABLE_TELEMETRY": (
                "true"
                if (
                    self.s
                    .weaviate_embedded_disable_telemetry
                )
                else "false"
            ),
            "QUERY_DEFAULTS_LIMIT": "25",
        }

        try:
            self.client = weaviate.connect_to_embedded(
                hostname="127.0.0.1",
                port=self.s.weaviate_embedded_http_port,
                grpc_port=self.s.weaviate_embedded_grpc_port,
                version=(
                    self.s
                    .weaviate_embedded_version
                ),
                persistence_data_path=str(
                    self.s
                    .weaviate_embedded_data_path
                    .resolve()
                ),
                binary_path=str(
                    self.s
                    .weaviate_embedded_binary_path
                    .resolve()
                ),
                environment_variables=(
                    environment_variables
                ),
            )
        except Exception:
            # Close the check/start race: another Humoid may have claimed
            # the endpoints after our first readiness probe.
            if self._adopt_existing(weaviate):
                return
            raise

        if not self.client.is_ready():
            raise RuntimeError(
                "Embedded Weaviate did not "
                "become ready"
            )

        self._ensure_collection()

        self._crud_roundtrip()


class MemoryRouter:
    """
    Select and manage the configured memory backend.

    Modes:

    embedded-weaviate
        Use Embedded Weaviate directly.

    weaviate
        Require a remote Weaviate server.

    sqlite
        Use SQLite directly.

    auto
        Try remote Weaviate, then Embedded Weaviate, then SQLite.
    """

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.s = settings
        self.backend: (
            SQLiteMemory
            | WeaviateMemory
            | EmbeddedWeaviateMemory
            | None
        ) = None

        self.status = "uninitialized"

    async def initialize(self) -> None:
        requested = (
            self.s.humoid_memory_backend
        )

        if requested == "embedded-weaviate":
            embedded = EmbeddedWeaviateMemory(
                self.s
            )

            try:
                await embedded.initialize()

            except Exception as exc:
                embedded_error = (
                    "embedded-weaviate failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                # Memory must never prevent the agent from starting. Keep the
                # durable SQLite store as a provider-independent fallback while
                # retaining the Weaviate data for the next healthy boot.
                sqlite = SQLiteMemory(
                    self.s.humoid_memory_db
                )
                await sqlite.initialize()
                self.backend = sqlite
                self.status = (
                    "sqlite: healthy fallback; "
                    f"{embedded_error}"
                )
                return

            self.backend = embedded

            self.status = (
                f"embedded-weaviate: {embedded.connection_mode} healthy "
                "(liveness + schema + CRUD)"
            )

            return

        if requested == "weaviate":
            remote = WeaviateMemory(
                self.s
            )

            try:
                await remote.initialize()

            except Exception as exc:
                self.status = (
                    "weaviate failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                raise RuntimeError(
                    self.status
                ) from exc

            self.backend = remote

            self.status = (
                "weaviate: healthy "
                "(liveness + schema + CRUD)"
            )

            return

        if requested == "sqlite":
            sqlite = SQLiteMemory(
                self.s.humoid_memory_db
            )

            await sqlite.initialize()

            self.backend = sqlite
            self.status = "sqlite: healthy"

            return

        if requested != "auto":
            raise ValueError(
                "Unsupported memory backend: "
                f"{requested}"
            )

        remote_error: Exception | None = None
        embedded_error: Exception | None = None

        try:
            remote = WeaviateMemory(
                self.s
            )

            await remote.initialize()

            self.backend = remote

            self.status = (
                "weaviate: healthy "
                "(liveness + schema + CRUD)"
            )

            return

        except Exception as exc:
            remote_error = exc

        try:
            embedded = EmbeddedWeaviateMemory(
                self.s
            )

            await embedded.initialize()

            self.backend = embedded

            self.status = (
                f"embedded-weaviate: {embedded.connection_mode} healthy "
                "(remote unavailable; "
                "liveness + schema + CRUD)"
            )

            return

        except Exception as exc:
            embedded_error = exc

        sqlite = SQLiteMemory(
            self.s.humoid_memory_db
        )

        await sqlite.initialize()

        self.backend = sqlite

        self.status = (
            "sqlite: healthy fallback; "
            "remote Weaviate failed="
            f"{type(remote_error).__name__}: "
            f"{remote_error}; "
            "embedded Weaviate failed="
            f"{type(embedded_error).__name__}: "
            f"{embedded_error}"
        )

    def _require_backend(
        self,
    ) -> (
        SQLiteMemory
        | WeaviateMemory
        | EmbeddedWeaviateMemory
    ):
        if self.backend is None:
            raise RuntimeError(
                "Memory backend has not been "
                "initialized"
            )

        return self.backend

    async def list_memories(self, limit: int = 200) -> list[MemoryHit]:
        return await self._require_backend().list_memories(limit)

    async def update_memory(self, memory_id: str, **changes: Any) -> None:
        await self._require_backend().update_memory(memory_id, **changes)

    async def delete_memory(self, memory_id: str) -> None:
        await self._require_backend().delete_memory(memory_id)

    async def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        backend = self._require_backend()

        return await backend.add(
            *args,
            **kwargs,
        )

    async def search(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> list[MemoryHit]:
        backend = self._require_backend()

        return await backend.search(
            *args,
            **kwargs,
        )

    async def context_packet(
        self,
        query: str,
        limit: int | None = None,
    ) -> str:
        backend = self._require_backend()

        if limit is None:
            limit = (
                self.s
                .humoid_memory_context_limit
            )

        hits = await self.search(
            query,
            self.s.humoid_memory_search_limit,
        )

        if not hits:
            return ""

        expanded: list[MemoryHit] = []
        seen: set[str] = set()

        for hit in hits:
            neighbors = (
                await backend.temporal_neighbors(
                    hit,
                    self.s
                    .humoid_memory_temporal_radius,
                )
            )

            for candidate in [
                hit,
                *neighbors,
            ]:
                if (
                    candidate.memory_id
                    in seen
                ):
                    continue

                expanded.append(
                    candidate
                )

                seen.add(
                    candidate.memory_id
                )

        selected: list[MemoryHit] = []

        for hit in sorted(
            expanded,
            key=lambda item: item.score,
            reverse=True,
        ):
            is_duplicate = any(
                _lexical_score(
                    hit.text,
                    existing.text,
                )
                >= 0.82
                for existing in selected
            )

            if is_duplicate:
                continue

            selected.append(
                hit
            )

            if len(selected) >= limit:
                break

        files = sorted({
            path
            for hit in selected
            for path in _file_references(hit.text)
        })
        sections = [
            "[RETRIEVED WORKING SET: use verified facts and successful evidence; "
            "ignore stale failures unless debugging them]",
            "ACTIVE FILE REFERENCES: " + (", ".join(files[:40]) or "none"),
        ]
        sections.extend(
            (
                f"[memory {index}] "
                f"tier={hit.memory_tier} "
                f"channel={hit.channel} "
                f"status="
                f"{hit.validation_status}\n"
                f"{hit.text[:2400]}"
            )
            for index, hit in enumerate(
                selected,
                start=1,
            )
        )
        packet = "\n\n".join(sections)
        return packet[:self.s.humoid_memory_packet_max_chars]

    async def close(self) -> None:
        if self.backend is None:
            return

        try:
            await self.backend.close()
        finally:
            self.backend = None
            self.status = "closed"
