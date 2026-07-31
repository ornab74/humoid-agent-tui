from __future__ import annotations

"""Task-scoped repository perspective with temporary Weaviate acceleration."""

import atexit
import asyncio
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

TEXT_SUFFIXES = {
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".dart",
    ".go", ".rs", ".java", ".kt", ".kts", ".c", ".cc", ".cpp", ".h",
    ".hpp", ".cs", ".rb", ".php", ".swift", ".scala", ".sh", ".bash",
    ".zsh", ".fish", ".ps1", ".sql", ".html", ".htm", ".css", ".scss",
    ".sass", ".less", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf", ".env", ".md", ".rst", ".txt", ".xml", ".graphql",
    ".proto", ".gradle", ".properties",
}
SPECIAL_FILES = {"dockerfile", "makefile", "justfile", "procfile", "gemfile", "rakefile", "license", "readme", "changelog", "authors", "notice"}
IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", ".venv", "venv", "env",
    "node_modules", "vendor", "target", "build", "dist", "out", ".next", ".nuxt",
    ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__", ".tox",
    ".nox", "coverage", "htmlcov", ".gradle", ".dart_tool", ".terraform", ".humoid",
}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|\d+(?:\.\d+)?")
SYMBOL_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|function|interface|type|struct|enum|trait|impl|fn|func|"
    r"public\s+class|private\s+class|export\s+(?:default\s+)?(?:class|function|const|let|var))"
    r"\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def safe_path(root: Path, value: Path) -> Path:
    resolved = value.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Perspective path escapes workspace: {value}")
    return resolved


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def lexical_score(query: str, text: str) -> float:
    left, right = Counter(tokens(query)), Counter(tokens(text))
    if not left or not right:
        return 0.0
    overlap = sum(min(count, right[token]) for token, count in left.items())
    return overlap / math.sqrt(sum(left.values()) * sum(right.values()))


class HashingEmbedder:
    """Private dependency-free vectors over identifiers, subtokens and bigrams."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = max(64, dimensions)

    def features(self, text: str) -> Iterable[str]:
        expanded: list[str] = []
        for token in tokens(text):
            expanded.append(token)
            expanded.extend(part for part in re.split(r"[_\-.]+", token) if len(part) > 1)
            expanded.extend(part.lower() for part in re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token).split() if len(part) > 1)
        yield from expanded
        yield from (f"{expanded[index]}::{expanded[index + 1]}" for index in range(len(expanded) - 1))

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature in self.features(text):
            number = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
            vector[number % self.dimensions] += -1.0 if (number >> 8) & 1 else 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


@dataclass(slots=True)
class PerspectiveChunk:
    chunk_id: str
    path: str
    line_start: int
    line_end: int
    chunk_index: int
    text: str
    digest: str
    symbols: tuple[str, ...] = ()
    vector: list[float] = field(default_factory=list, repr=False)

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.line_start}-{self.line_end}"


@dataclass(slots=True)
class PerspectiveHit:
    chunk: PerspectiveChunk
    score: float
    vector_score: float
    lexical_score: float


class ProjectPerspectiveIndex:
    """Disposable repository map, independent from durable conversational memory."""

    def __init__(self, settings: Any, root: Path) -> None:
        self.settings, self.root = settings, root.resolve()
        self.enabled = env_bool("HUMOID_PERSPECTIVE_ENABLED", True)
        self.backend_preference = os.getenv("HUMOID_PERSPECTIVE_BACKEND", "auto").strip().lower()
        self.chunk_chars = env_int("HUMOID_PERSPECTIVE_CHUNK_CHARS", 5600, 512)
        self.chunk_overlap = min(self.chunk_chars - 1, env_int("HUMOID_PERSPECTIVE_CHUNK_OVERLAP", 600, 0))
        self.max_files = env_int("HUMOID_PERSPECTIVE_MAX_FILES", 4000)
        self.max_file_bytes = env_int("HUMOID_PERSPECTIVE_MAX_FILE_BYTES", 1_000_000, 1024)
        self.default_limit = env_int("HUMOID_PERSPECTIVE_SEARCH_LIMIT", 12)
        self.packet_max_chars = env_int("HUMOID_PERSPECTIVE_PACKET_MAX_CHARS", 14000, 1000)
        self.embedder = HashingEmbedder(env_int("HUMOID_PERSPECTIVE_VECTOR_DIMS", 384, 64))
        self.collection_name = f"HumoidPerspective{uuid4().hex[:16]}"
        self.objective, self.backend = "", "unbuilt"
        self.manifest: dict[str, Any] = {}
        self.chunks: dict[str, PerspectiveChunk] = {}
        self.path_chunks: dict[str, list[str]] = defaultdict(list)
        self.dirty_paths: set[str] = set()
        self._client = self._collection = None
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._closed = False
        atexit.register(self._close_sync)

    @staticmethod
    def is_text(path: Path) -> bool:
        name = path.name.lower()
        return path.suffix.lower() in TEXT_SUFFIXES or name in SPECIAL_FILES or name.startswith("readme.")

    def scan(self, paths: list[str] | None) -> list[Path]:
        roots = [self.root] if not paths else [safe_path(self.root, self.root / (str(value).strip() or ".")) for value in paths]
        found: dict[str, Path] = {}
        for candidate_root in roots:
            candidates = [candidate_root] if candidate_root.is_file() else candidate_root.rglob("*")
            for path in candidates:
                if len(found) >= self.max_files:
                    break
                if not path.is_file() or not self.is_text(path):
                    continue
                relative = path.relative_to(self.root)
                if any(part in IGNORED_DIRS for part in relative.parts[:-1]):
                    continue
                try:
                    if path.stat().st_size > self.max_file_bytes or b"\x00" in path.read_bytes()[:4096]:
                        continue
                except OSError:
                    continue
                found[relative.as_posix()] = path
        return [found[key] for key in sorted(found)]

    def chunk_file(self, path: Path, content: str | None = None) -> list[PerspectiveChunk]:
        relative = path.relative_to(self.root).as_posix()
        try:
            source = content if content is not None else path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        lines, output, start, chunk_index = source.splitlines(keepends=True), [], 0, 0
        while start < len(lines):
            end, size = start, 0
            while end < len(lines) and (size < self.chunk_chars or end == start):
                size += len(lines[end]); end += 1
            text = "".join(lines[start:end]).rstrip()
            if text:
                digest = hashlib.sha256(text.encode(errors="replace")).hexdigest()[:16]
                chunk_id = hashlib.sha256(f"{relative}:{start + 1}:{end}:{digest}".encode()).hexdigest()[:24]
                symbols = tuple(dict.fromkeys(SYMBOL_RE.findall(text)))[:24]
                output.append(PerspectiveChunk(
                    chunk_id, relative, start + 1, end, chunk_index, text, digest, symbols,
                    self.embedder.embed(f"file {relative}\nidentifiers {' '.join(symbols)}\n{text}"),
                ))
                chunk_index += 1
            if end >= len(lines):
                break
            overlap, next_start = 0, end
            while next_start > start and overlap < self.chunk_overlap:
                next_start -= 1; overlap += len(lines[next_start])
            start = max(start + 1, next_start)
        return output

    def build_manifest(self, files: list[Path], chunks: list[PerspectiveChunk]) -> dict[str, Any]:
        suffixes = Counter(path.suffix.lower() or path.name.lower() for path in files)
        top_dirs = Counter((path.relative_to(self.root).parts[0] if len(path.relative_to(self.root).parts) > 1 else ".") for path in files)
        entry_names = {"main.py", "app.py", "server.py", "cli.py", "index.js", "index.ts", "main.dart", "main.go", "main.rs", "package.json", "pyproject.toml", "cargo.toml", "go.mod", "build.gradle", "pom.xml"}
        relatives = [(path, path.relative_to(self.root).as_posix()) for path in files]
        return {
            "files": len(files), "chunks": len(chunks), "characters": sum(len(chunk.text) for chunk in chunks),
            "languages": dict(suffixes.most_common(12)), "top_directories": dict(top_dirs.most_common(12)),
            "entrypoints": [relative for path, relative in relatives if path.name.lower() in entry_names][:30],
            "tests": [relative for path, relative in relatives if "test" in path.name.lower() or "tests" in path.parts][:30],
            "documentation": [relative for path, relative in relatives if path.name.lower().startswith("readme") or path.name.lower() in {"changelog.md", "contributing.md", "architecture.md"}][:20],
        }

    def connect_weaviate(self) -> None:
        if self.backend_preference in {"local", "memory", "off", "disabled"}:
            return
        try:
            import weaviate
        except ImportError:
            return
        client = None
        if str(getattr(self.settings, "humoid_memory_backend", "auto")) == "weaviate":
            try:
                client = weaviate.connect_to_custom(
                    http_host=getattr(self.settings, "weaviate_http_host", "127.0.0.1"),
                    http_port=int(getattr(self.settings, "weaviate_http_port", 8080)),
                    http_secure=bool(getattr(self.settings, "weaviate_secure", False)),
                    grpc_host=getattr(self.settings, "weaviate_grpc_host", "127.0.0.1"),
                    grpc_port=int(getattr(self.settings, "weaviate_grpc_port", 50051)),
                    grpc_secure=bool(getattr(self.settings, "weaviate_secure", False)),
                )
                if not client.is_ready(): client.close(); client = None
            except Exception:
                client = None
        if client is None:
            try:
                client = weaviate.connect_to_local(
                    host="127.0.0.1", port=int(getattr(self.settings, "weaviate_embedded_http_port", 8079)),
                    grpc_port=int(getattr(self.settings, "weaviate_embedded_grpc_port", 50050)),
                )
                if not client.is_ready(): client.close(); client = None
            except Exception:
                client = None
        if client is None and self.backend_preference in {"auto", "weaviate", "embedded-weaviate"}:
            try:
                self._tempdir = tempfile.TemporaryDirectory(prefix="humoid-perspective-")
                binary_path = Path(getattr(self.settings, "weaviate_embedded_binary_path", Path("./.humoid/weaviate-bin")))
                binary_path.mkdir(parents=True, exist_ok=True)
                client = weaviate.connect_to_embedded(
                    hostname="127.0.0.1", port=env_int("HUMOID_PERSPECTIVE_HTTP_PORT", 8089),
                    grpc_port=env_int("HUMOID_PERSPECTIVE_GRPC_PORT", 50059),
                    version=str(getattr(self.settings, "weaviate_embedded_version", "1.37.0")),
                    persistence_data_path=self._tempdir.name, binary_path=str(binary_path.resolve()),
                    environment_variables={"LOG_LEVEL": "error", "DISABLE_TELEMETRY": "true"},
                )
                if not client.is_ready(): client.close(); client = None
            except Exception:
                client = None
                if self._tempdir: self._tempdir.cleanup(); self._tempdir = None
        self._client = client

    def create_collection(self, chunks: list[PerspectiveChunk]) -> None:
        if self._client is None:
            return
        try:
            from weaviate.classes.config import Configure, DataType, Property
            if self._client.collections.exists(self.collection_name): self._client.collections.delete(self.collection_name)
            self._client.collections.create(
                self.collection_name, vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="chunk_id", data_type=DataType.TEXT), Property(name="path", data_type=DataType.TEXT),
                    Property(name="line_start", data_type=DataType.INT), Property(name="line_end", data_type=DataType.INT),
                    Property(name="chunk_index", data_type=DataType.INT), Property(name="digest", data_type=DataType.TEXT),
                    Property(name="symbols", data_type=DataType.TEXT_ARRAY), Property(name="text", data_type=DataType.TEXT),
                ],
            )
            self._collection = self._client.collections.get(self.collection_name)
            for chunk in chunks:
                self._collection.data.insert(
                    properties={"chunk_id": chunk.chunk_id, "path": chunk.path, "line_start": chunk.line_start,
                                "line_end": chunk.line_end, "chunk_index": chunk.chunk_index, "digest": chunk.digest,
                                "symbols": list(chunk.symbols), "text": chunk.text},
                    vector=chunk.vector,
                )
        except Exception:
            self.drop_collection(); self._collection = None

    def reset_transport(self) -> None:
        self.drop_collection()
        if self._client is not None:
            try: self._client.close()
            except Exception: pass
        self._client = None
        if self._tempdir is not None:
            try: self._tempdir.cleanup()
            except Exception: pass
            self._tempdir = None

    def build_sync(self, objective: str, paths: list[str] | None, force: bool) -> str:
        if not self.enabled:
            return "Project perspective indexing is disabled by HUMOID_PERSPECTIVE_ENABLED."
        if self.chunks and not force:
            return self.status_text("Perspective already built; reuse search_project_perspective.")
        self.reset_transport()
        files = self.scan(paths)
        chunks = [chunk for path in files for chunk in self.chunk_file(path)]
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.path_chunks = defaultdict(list)
        for chunk in chunks: self.path_chunks[chunk.path].append(chunk.chunk_id)
        self.objective, self.manifest = objective.strip(), self.build_manifest(files, chunks)
        self.dirty_paths.clear(); self.connect_weaviate(); self.create_collection(chunks)
        self.backend = "temporary-weaviate" if self._collection is not None else "local-vector-fallback"
        return self.status_text("Built project perspective.")

    async def build(self, objective: str, paths: list[str] | None = None, force: bool = False) -> str:
        return await asyncio.to_thread(self.build_sync, objective, paths, force)

    def local_candidates(self, query: str) -> list[PerspectiveHit]:
        query_vector, output = self.embedder.embed(query), []
        for chunk in self.chunks.values():
            vector = max(0.0, sum(a * b for a, b in zip(query_vector, chunk.vector, strict=False)))
            lexical = lexical_score(query, f"{chunk.path} {' '.join(chunk.symbols)} {chunk.text}")
            score = 0.68 * vector + 0.32 * lexical + (0.08 if any(token in chunk.path.lower() for token in tokens(query)) else 0.0)
            output.append(PerspectiveHit(chunk, score, vector, lexical))
        return output

    def weaviate_candidates(self, query: str, limit: int) -> list[PerspectiveHit]:
        if self._collection is None:
            return []
        try:
            from weaviate.classes.query import MetadataQuery
            result = self._collection.query.near_vector(
                near_vector=self.embedder.embed(query), limit=max(limit * 4, 24),
                return_metadata=MetadataQuery(distance=True),
            )
        except Exception:
            return []
        output = []
        for obj in result.objects:
            props = obj.properties or {}; path, chunk_id = str(props.get("path", "")), str(props.get("chunk_id", ""))
            if path in self.dirty_paths or chunk_id not in self.chunks: continue
            chunk = self.chunks[chunk_id]
            distance = float(getattr(obj.metadata, "distance", 1.0) or 1.0)
            vector, lexical = 1.0 / (1.0 + max(0.0, distance)), lexical_score(query, f"{chunk.path} {' '.join(chunk.symbols)} {chunk.text}")
            output.append(PerspectiveHit(chunk, 0.68 * vector + 0.32 * lexical, vector, lexical))
        return output

    @staticmethod
    def select_diverse(candidates: list[PerspectiveHit], limit: int) -> list[PerspectiveHit]:
        unique: dict[str, PerspectiveHit] = {}
        for hit in candidates:
            if hit.chunk.chunk_id not in unique or hit.score > unique[hit.chunk.chunk_id].score: unique[hit.chunk.chunk_id] = hit
        pool, selected, per_file = sorted(unique.values(), key=lambda hit: hit.score, reverse=True), [], Counter()
        while pool and len(selected) < limit:
            def adjusted(hit: PerspectiveHit) -> float:
                penalty = 0.08 * per_file[hit.chunk.path]
                penalty += sum(0.18 for prior in selected if prior.chunk.path == hit.chunk.path and abs(prior.chunk.chunk_index - hit.chunk.chunk_index) <= 1)
                return hit.score - penalty
            chosen = max(pool, key=adjusted); pool.remove(chosen); selected.append(chosen); per_file[chosen.chunk.path] += 1
        return selected

    def render_packet(self, query: str, hits: list[PerspectiveHit], budget: int) -> str:
        header = (
            "[PROJECT PERSPECTIVE: task-scoped repository evidence]\n"
            f"objective={self.objective or 'unspecified'}\nquery={query}\nbackend={self.backend}\n"
            f"manifest={json.dumps(self.manifest, ensure_ascii=False, sort_keys=True)}\n"
            "Use chunk IDs with expand_project_perspective. Read files directly only when editing or exact syntax is required."
        )
        parts, folded, used = [header], [], len(header)
        for hit in hits:
            chunk = hit.chunk
            block = (
                f"\n\n[PERSPECTIVE CHUNK {chunk.chunk_id} | {chunk.citation} | score={hit.score:.3f} "
                f"vector={hit.vector_score:.3f} lexical={hit.lexical_score:.3f}]\n"
                f"symbols={', '.join(chunk.symbols) or 'none'}\n{chunk.text}"
            )
            if used + len(block) <= budget: parts.append(block); used += len(block)
            else: folded.append(hit)
        if folded:
            grouped: dict[str, list[PerspectiveHit]] = defaultdict(list)
            for hit in folded: grouped[hit.chunk.path].append(hit)
            lines = ["\n\n[CONTEXT ACCORDION: lower-ranked evidence folded to protect model context]", "Expand only evidence needed for the next decision:"]
            for path, path_hits in sorted(grouped.items(), key=lambda item: max(hit.score for hit in item[1]), reverse=True)[:30]:
                ids = ",".join(hit.chunk.chunk_id for hit in path_hits[:4]); symbols = sorted({symbol for hit in path_hits for symbol in hit.chunk.symbols})[:8]
                lines.append(f"- {path}: {len(path_hits)} chunk(s); ids={ids}; symbols={','.join(symbols) or 'none'}")
            parts.append("\n".join(lines))
        return "".join(parts)[:budget]

    def search_sync(self, query: str, limit: int | None, max_chars: int | None) -> str:
        if not self.chunks: return "Perspective index is empty. Call build_project_perspective first."
        selected_limit = max(1, min(50, limit or self.default_limit)); budget = max(1000, min(100_000, max_chars or self.packet_max_chars))
        candidates = self.weaviate_candidates(query, selected_limit) + self.local_candidates(query)
        return self.render_packet(query, self.select_diverse(candidates, selected_limit), budget)

    async def search(self, query: str, limit: int | None = None, max_chars: int | None = None) -> str:
        return await asyncio.to_thread(self.search_sync, query, limit, max_chars)

    def expand_sync(self, chunk_ids: list[str], radius: int, max_chars: int | None) -> str:
        if not self.chunks: return "Perspective index is empty. Call build_project_perspective first."
        selected: dict[str, PerspectiveChunk] = {}; radius = max(0, min(4, radius))
        for chunk_id in chunk_ids[:20]:
            chunk = self.chunks.get(chunk_id)
            if not chunk: continue
            ids = self.path_chunks.get(chunk.path, []); position = ids.index(chunk_id) if chunk_id in ids else 0
            for neighbor in ids[max(0, position - radius):position + radius + 1]: selected[neighbor] = self.chunks[neighbor]
        if not selected: return "No requested perspective chunk IDs were found. Search again for current IDs."
        packet = "[EXPANDED PROJECT PERSPECTIVE]" + "".join(
            f"\n\n[{chunk.chunk_id} | {chunk.citation}]\nsymbols={', '.join(chunk.symbols) or 'none'}\n{chunk.text}"
            for chunk in sorted(selected.values(), key=lambda item: (item.path, item.chunk_index))
        )
        budget = max(1000, min(100_000, max_chars or self.packet_max_chars))
        return packet if len(packet) <= budget else packet[:budget] + "\n\n[expanded context truncated]"

    async def expand(self, chunk_ids: list[str], radius: int = 1, max_chars: int | None = None) -> str:
        return await asyncio.to_thread(self.expand_sync, chunk_ids, radius, max_chars)

    async def notify_file_changed(self, relative: str, content: str) -> None:
        if not self.chunks: return
        path = safe_path(self.root, self.root / relative); normalized = path.relative_to(self.root).as_posix()
        for chunk_id in self.path_chunks.pop(normalized, []): self.chunks.pop(chunk_id, None)
        for chunk in self.chunk_file(path, content): self.chunks[chunk.chunk_id] = chunk; self.path_chunks[normalized].append(chunk.chunk_id)
        self.dirty_paths.add(normalized); self.manifest["chunks"] = len(self.chunks)
        self.backend = "temporary-weaviate+local-delta" if self._collection is not None else "local-vector-fallback"

    async def notify_file_deleted(self, relative: str) -> None:
        if not self.chunks: return
        normalized = safe_path(self.root, self.root / relative).relative_to(self.root).as_posix()
        for chunk_id in self.path_chunks.pop(normalized, []): self.chunks.pop(chunk_id, None)
        self.dirty_paths.add(normalized); self.manifest["chunks"] = len(self.chunks)

    def status_text(self, prefix: str = "Perspective status.") -> str:
        return f"{prefix} backend={self.backend}; collection={self.collection_name}; files={self.manifest.get('files', 0)}; chunks={len(self.chunks)}; dirty_files={len(self.dirty_paths)}; objective={self.objective or 'none'}"

    def status_json(self) -> str:
        return json.dumps({
            "enabled": self.enabled, "backend": self.backend, "backend_preference": self.backend_preference,
            "collection": self.collection_name, "objective": self.objective, "manifest": self.manifest,
            "dirty_paths": sorted(self.dirty_paths), "chunk_chars": self.chunk_chars,
            "chunk_overlap": self.chunk_overlap, "packet_max_chars": self.packet_max_chars,
        }, indent=2, ensure_ascii=False)

    def drop_collection(self) -> None:
        if self._client is not None and self._collection is not None:
            try:
                if self._client.collections.exists(self.collection_name): self._client.collections.delete(self.collection_name)
            except Exception: pass
        self._collection = None

    def clear_sync(self) -> None:
        self.reset_transport(); self.chunks.clear(); self.path_chunks.clear(); self.dirty_paths.clear()
        self.manifest, self.objective = {}, ""; self.collection_name = f"HumoidPerspective{uuid4().hex[:16]}"; self.backend = "cleared"

    def _close_sync(self) -> None:
        if self._closed: return
        self._closed = True; self.clear_sync()

    async def clear(self) -> str:
        await asyncio.to_thread(self.clear_sync)
        return "Cleared temporary project perspective collection and local chunk cache."
