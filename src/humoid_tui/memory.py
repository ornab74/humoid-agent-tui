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
from .rhomgaic_memory import (
    AgenticRagContextBuilder,
    ChromaticCodeEmbedding,
    ContextSurfaceCompiler,
    RhomgaicMemory,
)


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

        properties = [
            Property(name="text", data_type=DataType.TEXT),
            Property(name="memory_tier", data_type=DataType.TEXT),
            Property(name="channel", data_type=DataType.TEXT),
            Property(name="task_id", data_type=DataType.TEXT),
            Property(name="validation_status", data_type=DataType.TEXT),
            Property(name="created_at", data_type=DataType.DATE),
            Property(name="metadata_json", data_type=DataType.TEXT),
        ]

        # weaviate-client >=4.16 renamed the self-provided vector API. Keep
        # compatibility with the project's >=4.12 floor and current releases.
        vectors_api = getattr(Configure, "Vectors", None)
        if vectors_api is not None and hasattr(vectors_api, "self_provided"):
            self.client.collections.create(
                self.s.weaviate_collection,
                vector_config=vectors_api.self_provided(),
                properties=properties,
            )
        else:
            self.client.collections.create(
                self.s.weaviate_collection,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=properties,
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
            sentinel_embedding = ChromaticCodeEmbedding.encode(
                sentinel, dimensions=self.s.humoid_rhomgaic_dimensions
            )
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
                    "metadata_json": json.dumps({
                        "chromatic_action": sentinel_embedding.action,
                        "chromatic_language": sentinel_embedding.language,
                        "chromatic_address": sentinel_embedding.address.hex(),
                    }),
                },
                vector=list(sentinel_embedding.vector),
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

            queried = collection.query.hybrid(
                query=sentinel,
                vector=list(sentinel_embedding.vector),
                alpha=0.58,
                query_properties=["text"],
                limit=1,
            )

            if not queried.objects:
                raise RuntimeError(
                    "Weaviate chromatic hybrid query round-trip failed"
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
        chromatic = ChromaticCodeEmbedding.encode(
            text, dimensions=self.s.humoid_rhomgaic_dimensions
        )
        metadata = dict(meta.get("metadata", {}))
        metadata.update({
            "chromatic_action": chromatic.action,
            "chromatic_language": chromatic.language,
            "chromatic_address": chromatic.address.hex(),
            "chromatic_files": list(chromatic.files),
            "chromatic_symbols": list(chromatic.symbols),
        })

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
                    metadata, ensure_ascii=False,
                ),
            },
            vector=list(chromatic.vector),
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

        chromatic = ChromaticCodeEmbedding.encode(
            query, dimensions=self.s.humoid_rhomgaic_dimensions
        )
        result = collection.query.hybrid(
            query=query,
            vector=list(chromatic.vector),
            alpha=0.58,
            query_properties=["text"],
            # Weaviate contributes semantic/vector recall while the Rhomgaic
            # sidecar contributes chromatic/action/file/symbol recall.
            limit=max(limit * 4, 24),
            return_metadata=MetadataQuery(score=True),
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
        if not selected:
            return

        vector = None
        if "text" in selected:
            chromatic = ChromaticCodeEmbedding.encode(
                str(selected["text"]), dimensions=self.s.humoid_rhomgaic_dimensions
            )
            vector = list(chromatic.vector)
            current = await asyncio.to_thread(
                self._collection().query.fetch_object_by_id, memory_id
            )
            properties = current.properties if current is not None else {}
            metadata = _safe_metadata_json((properties or {}).get("metadata_json", "{}"))
            metadata.update({
                "chromatic_action": chromatic.action,
                "chromatic_language": chromatic.language,
                "chromatic_address": chromatic.address.hex(),
                "chromatic_files": list(chromatic.files),
                "chromatic_symbols": list(chromatic.symbols),
            })
            selected["metadata_json"] = json.dumps(metadata, ensure_ascii=False)

        kwargs: dict[str, Any] = {"uuid": memory_id, "properties": selected}
        if vector is not None:
            kwargs["vector"] = vector
        await asyncio.to_thread(self._collection().data.update, **kwargs)

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
        self.rhomgaic = RhomgaicMemory(
            self.s.humoid_rhomgaic_db,
            dimensions=self.s.humoid_rhomgaic_dimensions,
            candidate_limit=self.s.humoid_rhomgaic_candidate_limit,
            graph_max_edges=self.s.humoid_rhomgaic_graph_max_edges,
            importance_half_life_days=self.s.humoid_rhomgaic_importance_half_life_days,
            policy_adaptation=self.s.humoid_rhomgaic_policy_adaptation,
            semantic_capsules=self.s.humoid_rhomgaic_semantic_capsules,
        ) if self.s.humoid_rhomgaic_enabled else None
        self.rag = AgenticRagContextBuilder(self)
        self.surface_compiler = ContextSurfaceCompiler(
            dimensions=self.s.humoid_rhomgaic_dimensions
        )
        self.last_trace: dict[str, Any] = {}

    async def initialize(self) -> None:
        if self.rhomgaic is not None:
            await self.rhomgaic.initialize()
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
                await self._bootstrap_rhomgaic()
                return

            self.backend = embedded

            self.status = (
                f"embedded-weaviate: {embedded.connection_mode} healthy "
                "(liveness + schema + CRUD)"
            )
            await self._bootstrap_rhomgaic()

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
            await self._bootstrap_rhomgaic()

            return

        if requested == "sqlite":
            sqlite = SQLiteMemory(
                self.s.humoid_memory_db
            )

            await sqlite.initialize()

            self.backend = sqlite
            self.status = "sqlite: healthy"
            await self._bootstrap_rhomgaic()

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
            await self._bootstrap_rhomgaic()

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
            await self._bootstrap_rhomgaic()

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
        await self._bootstrap_rhomgaic()

    async def _bootstrap_rhomgaic(self) -> None:
        if self.rhomgaic is None or self.backend is None:
            return
        try:
            stats = await self.rhomgaic.stats()
            if int(stats.get("memories", 0)) > 0:
                return
            memories = await self.backend.list_memories(
                self.s.humoid_rhomgaic_bootstrap_limit
            )
            if memories:
                count = await self.rhomgaic.synchronize(memories)
                self.status += f"; rhomgaic bootstrapped={count}"
        except Exception as exc:
            # Provider memory is authoritative; an optional sidecar bootstrap
            # cannot make agent startup fail. Surface the degradation in status.
            self.status += f"; rhomgaic bootstrap degraded={type(exc).__name__}"

    async def surface_status(self) -> dict[str, Any]:
        rhomgaic = await self.rhomgaic.stats() if self.rhomgaic is not None else {"enabled": False}
        return {
            "provider": type(self.backend).__name__ if self.backend is not None else None,
            "status": self.status,
            "rhomgaic": rhomgaic,
            "last_recall": dict(self.last_trace),
        }

    async def consolidate_memory_surface(
        self,
        *,
        similarity: float = 0.92,
        max_aliases: int = 256,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        if self.rhomgaic is None:
            return {"enabled": False, "aliases_proposed": 0}
        result = await self.rhomgaic.consolidate(
            similarity=similarity,
            max_aliases=max_aliases,
            dry_run=dry_run,
        )
        result["provider_unchanged"] = True
        return result

    async def attach_code_surface(self, memory_id: str, path: str, content: str) -> dict[str, Any]:
        if self.rhomgaic is None:
            return {"indexed": False, "reason": "rhomgaic-disabled"}
        return await self.rhomgaic.attach_code_surface(memory_id, path, content)

    async def distill_memory_surface(
        self, *, min_support: int = 2, max_capsules: int = 128, dry_run: bool = False
    ) -> dict[str, Any]:
        if self.rhomgaic is None:
            return {"enabled": False, "capsules": 0}
        result = await self.rhomgaic.distill_semantic_capsules(
            min_support=min_support, max_capsules=max_capsules, dry_run=dry_run
        )
        result["provider_unchanged"] = True
        return result

    async def chromatic_surface(self, query: str, *, limit: int = 8) -> dict[str, Any]:
        if self.rhomgaic is None:
            return {"enabled": False, "query": query[:1000]}
        try:
            payload = await self.rhomgaic.chromatic_surface(
                query, limit=max(1, min(32, int(limit)))
            )
            payload["enabled"] = True
            return payload
        except Exception as exc:
            return {"enabled": False, "query": query[:1000], "error": type(exc).__name__}

    async def chromatic_surface_packet(
        self, query: str, *, limit: int = 8, max_chars: int = 12000
    ) -> str:
        if self.rhomgaic is None:
            return "[CHROMATIC CODE SURFACE]\nnone: rhomgaic-disabled"
        try:
            return await self.rhomgaic.chromatic_surface_packet(
                query, limit=max(1, min(32, int(limit))), max_chars=max_chars
            )
        except Exception as exc:
            return f"[CHROMATIC CODE SURFACE]\nunavailable={type(exc).__name__}"

    async def chromatic_change_playbooks(
        self, fingerprints: dict[str, Any], *, limit: int = 5
    ) -> list[dict[str, Any]]:
        if self.rhomgaic is None:
            return []
        try:
            return await self.rhomgaic.search_change_procedures(
                fingerprints, limit=max(1, min(12, int(limit)))
            )
        except Exception:
            return []

    async def chromatic_change_playbook_packet(
        self, fingerprints: dict[str, Any], *, limit: int = 5, max_chars: int = 8000
    ) -> str:
        if self.rhomgaic is None:
            return "[CHROMATIC CHANGE-HISTORY SURFACE]\n- none"
        try:
            return await self.rhomgaic.change_procedure_packet(
                fingerprints, limit=max(1, min(12, int(limit))), max_chars=max_chars
            )
        except Exception:
            return "[CHROMATIC CHANGE-HISTORY SURFACE]\n- none"

    async def coding_playbooks(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        if self.rhomgaic is None:
            return []
        try:
            return await self.rhomgaic.search_procedures(query, limit=max(1, min(24, int(limit))))
        except Exception:
            return []

    async def coding_playbook_packet(
        self, query: str, *, limit: int = 5, max_chars: int = 8000
    ) -> str:
        if self.rhomgaic is None:
            return "[CODING PROCEDURAL PLAYBOOKS]\n- none"
        try:
            return await self.rhomgaic.procedure_packet(
                query, limit=max(1, min(24, int(limit))), max_chars=max_chars
            )
        except Exception:
            return "[CODING PROCEDURAL PLAYBOOKS]\n- none"

    async def record_coding_trajectory(self, trajectory: dict[str, Any]) -> str:
        """Persist a compact procedural trajectory in the durable provider.

        Rhomgaic derives its reusable procedural index from the provider metadata,
        so a sidecar rebuild can recover these playbooks later.
        """
        request = str(trajectory.get("request") or "").strip()[:4000]
        changed = [str(value) for value in (trajectory.get("changed_paths") or []) if str(value)]
        validators = [str(value) for value in (trajectory.get("validators") or []) if str(value)]
        steps = []
        for item in trajectory.get("steps") or trajectory.get("ledger") or []:
            if isinstance(item, dict):
                value = str(item.get("kind") or item.get("tool") or "").strip()
            else:
                value = str(item).strip().split(":", 1)[0]
            if value and value not in steps:
                steps.append(value)
            if len(steps) >= 32:
                break
        validated = bool(trajectory.get("validated"))
        failures = max(0, int(trajectory.get("failures", 0) or 0))
        outcome = "verified" if validated else ("failed" if failures else "observed")
        text = (
            "CODING PROCEDURE MEMORY\n"
            f"request={request}\n"
            f"operation={trajectory.get('operation', 'edit')}\n"
            f"files={','.join(changed)}\n"
            f"steps={' > '.join(steps)}\n"
            f"validators={' ; '.join(validators)}\n"
            f"outcome={outcome}\n"
            f"summary={str(trajectory.get('summary') or '')[:1200]}"
        )[:12000]
        return await self.add(
            text,
            memory_tier="procedural",
            channel="code_procedure",
            task_id=str(trajectory.get("task_id") or ""),
            validation_status="verified" if validated else "observed",
            metadata={"coding_trajectory": dict(trajectory)},
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
        if self.rhomgaic is not None:
            try:
                await self.rhomgaic.update(memory_id, **changes)
            except Exception:
                pass

    async def delete_memory(self, memory_id: str) -> None:
        await self._require_backend().delete_memory(memory_id)
        if self.rhomgaic is not None:
            await self.rhomgaic.delete(memory_id)

    async def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        backend = self._require_backend()

        memory_id = await backend.add(
            *args,
            **kwargs,
        )
        if self.rhomgaic is not None and args:
            try:
                await self.rhomgaic.add(memory_id, str(args[0]), **kwargs)
            except Exception:
                # The provider remains authoritative. The sidecar is rebuildable
                # and must never block an agent memory write.
                pass
        return memory_id

    async def _fused_search(
        self, query: str, limit: int, *, breadth: int = 4
    ) -> list[MemoryHit]:
        backend = self._require_backend()
        provider_hits = await backend.search(query, max(limit * breadth, limit))
        chromatic_hits = (
            await self.rhomgaic.search(query, max(limit * breadth, limit))
            if self.rhomgaic is not None else []
        )
        fused: dict[str, MemoryHit] = {}
        # Reciprocal-rank fusion lets Weaviate and the chromatic surface vote
        # without requiring their raw score scales to be comparable.
        for source_weight, hits in ((1.0, provider_hits), (0.92, chromatic_hits)):
            for rank, hit in enumerate(hits, start=1):
                vote = source_weight / (24.0 + rank)
                if hit.memory_id not in fused:
                    fused[hit.memory_id] = MemoryHit(
                        memory_id=hit.memory_id, text=hit.text, score=0.0,
                        memory_tier=hit.memory_tier, channel=hit.channel,
                        task_id=hit.task_id, validation_status=hit.validation_status,
                        created_at=hit.created_at, metadata=dict(hit.metadata),
                    )
                else:
                    fused[hit.memory_id].metadata.update(hit.metadata)
                fused[hit.memory_id].score += vote
        ranked = list(fused.values())
        for hit in ranked:
            hit.score += 0.035 * _rank_memory(query, hit)
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    async def _graph_expand(
        self,
        anchors: list[MemoryHit],
        *,
        limit: int,
        hops: int,
    ) -> list[MemoryHit]:
        if self.rhomgaic is None or not anchors:
            return []
        try:
            return await self.rhomgaic.expand_graph(
                anchors, limit=limit, hops=hops
            )
        except Exception:
            return []

    async def search(
        self, query: str, limit: int = 8,
    ) -> list[MemoryHit]:
        return await self._fused_search(query, limit)

    async def context_packet(
        self,
        query: str,
        limit: int | None = None,
        *,
        depth: str = "auto",
        max_chars: int | None = None,
    ) -> str:
        backend = self._require_backend()

        if limit is None:
            limit = self.s.humoid_memory_context_limit
        limit = max(1, min(64, int(limit)))
        if depth not in {"auto", "reflex", "working", "subsurface"}:
            raise ValueError("depth must be auto, reflex, working, or subsurface")

        packet_budget = max_chars or self.s.humoid_memory_packet_max_chars
        if depth == "reflex" and self.s.humoid_rag_agentic_growth:
            packet_budget = min(packet_budget, self.s.humoid_rag_reflex_max_chars)
        elif depth == "working":
            packet_budget = min(packet_budget, self.s.humoid_rag_working_max_chars)
        packet_budget = max(256, int(packet_budget))

        if self.rhomgaic is not None and self.s.humoid_rag_context_cache:
            cached = await self.rhomgaic.get_cached_context(
                query,
                depth,
                packet_budget,
                ttl_seconds=self.s.humoid_rag_context_cache_ttl_seconds,
            )
            if cached is not None:
                self.last_trace = dict(cached.get("trace") or {})
                return str(cached.get("packet") or "")

        rag_stages = (
            await self.rag.gather(query, limit, max_depth=depth)
            if self.s.humoid_rag_progressive
            else []
        )
        if self.rhomgaic is not None and rag_stages:
            await self.rhomgaic.record_recall_outcome(
                query, rag_stages, forced=(depth != "auto")
            )
        hits = (
            rag_stages[-1].hits
            if rag_stages
            else await self.search(query, self.s.humoid_memory_search_limit)
        )
        if not hits:
            self.last_trace = {
                "query": query[:240],
                "depth": depth,
                "stages": [],
                "selected": 0,
            }
            return ""

        # Reflex retrieval stays tiny. Temporal expansion becomes available only
        # once the agent intentionally opens a larger context surface.
        temporal_radius = 0 if depth == "reflex" else self.s.humoid_memory_temporal_radius
        expanded: list[MemoryHit] = []
        seen: set[str] = set()
        for hit in hits:
            neighbors = (
                await backend.temporal_neighbors(hit, temporal_radius)
                if temporal_radius > 0
                else []
            )
            for candidate in [hit, *neighbors]:
                if candidate.memory_id in seen:
                    continue
                expanded.append(candidate)
                seen.add(candidate.memory_id)

        selected: list[MemoryHit] = []
        for hit in sorted(
            expanded,
            key=lambda item: (_rank_memory(query, item), item.score),
            reverse=True,
        ):
            is_duplicate = any(
                _lexical_score(hit.text, existing.text) >= 0.86
                for existing in selected
            )
            if is_duplicate:
                continue
            selected.append(hit)
            if len(selected) >= limit:
                break

        if not selected:
            return ""

        if self.rhomgaic is not None:
            await self.rhomgaic.record_access([hit.memory_id for hit in selected])

        procedures: list[dict[str, Any]] = []
        if self.rhomgaic is not None:
            encoded_query = ChromaticCodeEmbedding.encode(
                query, dimensions=self.s.humoid_rhomgaic_dimensions
            )
            if encoded_query.files or encoded_query.symbols or encoded_query.language != "text" or encoded_query.action in {"edit", "debug", "test", "create"}:
                procedures = await self.coding_playbooks(query, limit=1 if depth == "reflex" else 4)

        packet = self.surface_compiler.compile(
            query,
            rag_stages,
            selected,
            max_chars=packet_budget,
            procedures=procedures,
        )
        self.last_trace = {
            "query": query[:240],
            "depth": depth,
            "packet_chars": len(packet),
            "selected": len(selected),
            "procedural_playbooks": len(procedures),
            "cache_hit": False,
            "retrieval_plan": (
                self.rag.last_plan.to_dict()
                if self.rag.last_plan is not None
                else None
            ),
            "chromatic_candidate_trace": (
                dict(self.rhomgaic.last_chromatic_trace)
                if self.rhomgaic is not None
                else {}
            ),
            "stages": [
                {
                    "name": stage.name,
                    "hits": len(stage.hits),
                    "entropy_bits": stage.entropy_bits,
                    "novelty": stage.novelty,
                    "reason": stage.reason,
                    "breadth": stage.breadth,
                }
                for stage in rag_stages
            ],
        }
        if self.rhomgaic is not None and self.s.humoid_rag_context_cache:
            await self.rhomgaic.put_cached_context(
                query,
                depth,
                packet_budget,
                packet,
                self.last_trace,
                max_entries=self.s.humoid_rag_context_cache_max_entries,
            )
        return packet

    async def close(self) -> None:
        if self.backend is None:
            return

        try:
            await self.backend.close()
        finally:
            self.backend = None
            self.status = "closed"
