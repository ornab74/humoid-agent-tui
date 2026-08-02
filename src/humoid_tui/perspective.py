from __future__ import annotations

"""Task-scoped code search using constraints, a symbol graph, and failure memory."""

import atexit
import asyncio
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

TEXT_SUFFIXES = {
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".dart",
    ".go", ".rs", ".java", ".kt", ".kts", ".c", ".cc", ".cpp", ".h",
    ".hpp", ".cs", ".rb", ".php", ".swift", ".scala", ".sh", ".sql",
    ".html", ".css", ".scss", ".json", ".yaml", ".yml", ".toml", ".md",
}
IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "vendor", "target", "build",
    "dist", ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "__pycache__", ".tox", ".nox", ".gradle", ".dart_tool", ".humoid",
}
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "code", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "where", "which", "with", "work", "works",
}
ROLE_HINTS = {
    "test": {"test", "tests", "spec", "assert", "fixture", "mock"},
    "config": {"config", "settings", "environment", "env", "yaml", "toml"},
    "storage": {"database", "sqlite", "weaviate", "store", "storage", "persist", "cache"},
    "network": {"api", "http", "request", "response", "client", "server", "socket"},
    "auth": {"auth", "token", "session", "login", "credential", "permission"},
    "ui": {"ui", "tui", "view", "widget", "screen", "render", "component"},
    "error": {"error", "failure", "bug", "exception", "traceback", "regression", "crash"},
}
FAILURE_MARKERS = {
    "bug", "crash", "error", "exception", "fail", "failed", "failure", "fix",
    "fixed", "regression", "stack trace", "traceback", "validation error",
}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|\d+(?:\.\d+)?")
FILE_RE = re.compile(
    r"(?<![\w.-])(?:[\w.-]+/)*[\w.-]+\."
    r"(?:py|pyi|js|mjs|cjs|ts|tsx|jsx|dart|go|rs|java|kt|kts|c|cc|cpp|h|hpp|"
    r"cs|rb|php|swift|scala|sh|sql|html|css|scss|json|ya?ml|toml|md)\b",
    re.IGNORECASE,
)
SYMBOL_DEF_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|function|interface|type|struct|enum|trait|fn|func|"
    r"public\s+class|private\s+class|export\s+(?:default\s+)?(?:class|function|const|let|var))"
    r"\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)
CALL_RE = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")
IMPORT_RE = re.compile(
    r"(?:^\s*(?:from|import)\s+([\w.]+)|\bfrom\s+[\"']([^\"']+)[\"']|"
    r"\brequire\(\s*[\"']([^\"']+)[\"']\s*\))",
    re.MULTILINE,
)
CALL_KEYWORDS = {"if", "for", "while", "return", "assert", "class", "def", "function", "with"}


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


def trigrams(term: str) -> set[str]:
    padded = f"  {term.lower()}  "
    return {padded[index:index + 3] for index in range(max(1, len(padded) - 2))}


class HashingEmbedder:
    """Cheap local vectors used only as a tie-breaker and cold-start fallback."""

    def __init__(self, dimensions: int = 192) -> None:
        self.dimensions = max(64, dimensions)

    def features(self, text: str) -> Iterable[str]:
        values = tokens(text)
        yield from values
        yield from (f"{values[index]}::{values[index + 1]}" for index in range(len(values) - 1))

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
    references: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    vector: list[float] = field(default_factory=list, repr=False)

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.line_start}-{self.line_end}"


@dataclass(slots=True)
class GraphEdge:
    source: str
    target: str
    kind: str
    label: str
    weight: float


@dataclass(slots=True)
class PerspectiveHit:
    chunk: PerspectiveChunk
    score: float
    vector_score: float
    lexical_score: float
    constraint_score: float = 0.0
    graph_score: float = 0.0
    failure_score: float = 0.0
    reasons: tuple[str, ...] = ()
    anchor: bool = False


@dataclass(slots=True)
class FailureBias:
    path_scores: dict[str, float] = field(default_factory=dict)
    symbol_scores: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    status: str = "unavailable"


class ProjectPerspectiveIndex:
    """Return the smallest connected code subgraph that explains a behavior."""

    def __init__(self, settings: Any, root: Path, memory: Any | None = None) -> None:
        self.settings, self.root = settings, root.resolve()
        self.enabled = env_bool("HUMOID_PERSPECTIVE_ENABLED", True)
        self.chunk_chars = env_int("HUMOID_PERSPECTIVE_CHUNK_CHARS", 5600, 512)
        self.chunk_overlap = min(self.chunk_chars - 1, env_int("HUMOID_PERSPECTIVE_CHUNK_OVERLAP", 600, 0))
        self.max_files = env_int("HUMOID_PERSPECTIVE_MAX_FILES", 4000)
        self.max_file_bytes = env_int("HUMOID_PERSPECTIVE_MAX_FILE_BYTES", 1_000_000, 1024)
        self.default_limit = env_int("HUMOID_PERSPECTIVE_SEARCH_LIMIT", 12)
        self.packet_max_chars = env_int("HUMOID_PERSPECTIVE_PACKET_MAX_CHARS", 14000, 1000)
        self.graph_max_hops = env_int("HUMOID_PERSPECTIVE_GRAPH_MAX_HOPS", 3, 1)
        self.graph_fanout = env_int("HUMOID_PERSPECTIVE_GRAPH_FANOUT", 20, 2)
        self.failure_memory_limit = env_int("HUMOID_PERSPECTIVE_FAILURE_MEMORY_LIMIT", 16, 1)
        self.failure_memory_enabled = env_bool("HUMOID_PERSPECTIVE_FAILURE_MEMORY", True)
        self.embedder = HashingEmbedder(env_int("HUMOID_PERSPECTIVE_VECTOR_DIMS", 192, 64))
        self.collection_name = f"HumoidPerspective{uuid4().hex[:16]}"
        self.objective, self.backend = "", "unbuilt"
        self.manifest: dict[str, Any] = {}
        self.chunks: dict[str, PerspectiveChunk] = {}
        self.path_chunks: dict[str, list[str]] = defaultdict(list)
        self.dirty_paths: set[str] = set()
        self.token_chunks: dict[str, set[str]] = defaultdict(set)
        self.symbol_definitions: dict[str, set[str]] = defaultdict(set)
        self.symbol_references: dict[str, set[str]] = defaultdict(set)
        self.trigram_terms: dict[str, set[str]] = defaultdict(set)
        self.adjacency: dict[str, dict[str, GraphEdge]] = defaultdict(dict)
        self._memory = memory
        self._owns_memory = memory is None
        self._memory_initialized = False
        self._closed = False
        atexit.register(self._close_sync)

    @staticmethod
    def is_text(path: Path) -> bool:
        return path.suffix.lower() in TEXT_SUFFIXES or path.name.lower() in {"dockerfile", "makefile", "readme"}

    def scan(self, paths: list[str] | None) -> list[Path]:
        roots = [self.root] if not paths else [safe_path(self.root, self.root / value) for value in paths]
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

    @staticmethod
    def _roles(path: str, text: str) -> tuple[str, ...]:
        haystack = f"{path} {text[:2000]}".lower()
        return tuple(role for role, hints in ROLE_HINTS.items() if any(hint in haystack for hint in hints))

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
                symbols = tuple(dict.fromkeys(SYMBOL_DEF_RE.findall(text)))[:40]
                defined = {symbol.lower() for symbol in symbols}
                references = tuple(dict.fromkeys(
                    value for value in CALL_RE.findall(text)
                    if value.lower() not in defined and value.lower() not in CALL_KEYWORDS
                ))[:80]
                imports = []
                for match in IMPORT_RE.finditer(text):
                    value = next((group for group in match.groups() if group), "")
                    if value:
                        imports.append(value.replace(".", "/").lstrip("./"))
                digest = hashlib.sha256(text.encode(errors="replace")).hexdigest()[:16]
                chunk_id = hashlib.sha256(f"{relative}:{start + 1}:{end}:{digest}".encode()).hexdigest()[:24]
                roles = self._roles(relative, text)
                vector_text = f"{relative} {' '.join(symbols)} {' '.join(references)} {' '.join(roles)} {text}"
                output.append(PerspectiveChunk(
                    chunk_id, relative, start + 1, end, chunk_index, text, digest,
                    symbols, references, tuple(dict.fromkeys(imports))[:40], roles,
                    self.embedder.embed(vector_text),
                ))
                chunk_index += 1
            if end >= len(lines):
                break
            overlap, next_start = 0, end
            while next_start > start and overlap < self.chunk_overlap:
                next_start -= 1; overlap += len(lines[next_start])
            start = max(start + 1, next_start)
        return output

    def _add_edge(self, source: str, target: str, kind: str, label: str, weight: float) -> None:
        if source == target or source not in self.chunks or target not in self.chunks:
            return
        edge = GraphEdge(source, target, kind, label, weight)
        current = self.adjacency[source].get(target)
        if current is None or weight > current.weight:
            self.adjacency[source][target] = edge
            self.adjacency[target][source] = GraphEdge(target, source, kind, label, weight)

    def _rebuild_indexes(self) -> None:
        self.token_chunks = defaultdict(set)
        self.symbol_definitions = defaultdict(set)
        self.symbol_references = defaultdict(set)
        self.trigram_terms = defaultdict(set)
        self.adjacency = defaultdict(dict)
        path_aliases: dict[str, str] = {}
        for path, ids in self.path_chunks.items():
            stem = Path(path).stem.lower()
            path_aliases[stem] = path
            path_aliases[str(Path(path).with_suffix("")).replace("\\", "/").lower()] = path
            for left, right in zip(ids, ids[1:]):
                self._add_edge(left, right, "neighbor", "adjacent chunk", 0.25)
        for chunk in self.chunks.values():
            indexed = set(tokens(f"{chunk.path} {' '.join(chunk.symbols)} {' '.join(chunk.references)} {' '.join(chunk.roles)} {chunk.text}"))
            for term in indexed:
                self.token_chunks[term].add(chunk.chunk_id)
                for trigram in trigrams(term):
                    self.trigram_terms[trigram].add(term)
            for symbol in chunk.symbols:
                self.symbol_definitions[symbol.lower()].add(chunk.chunk_id)
            for reference in chunk.references:
                self.symbol_references[reference.lower()].add(chunk.chunk_id)
        for symbol, definitions in self.symbol_definitions.items():
            references = self.symbol_references.get(symbol, set())
            for definition in definitions:
                ordered = sorted(references, key=lambda item: self.chunks[item].path != self.chunks[definition].path)
                for reference in ordered[:self.graph_fanout]:
                    self._add_edge(definition, reference, "symbol", symbol, 1.0)
        for chunk in self.chunks.values():
            for imported in chunk.imports:
                stem = imported.rsplit("/", 1)[-1].lower()
                target_path = path_aliases.get(imported.lower()) or path_aliases.get(stem)
                if target_path and self.path_chunks[target_path]:
                    self._add_edge(chunk.chunk_id, self.path_chunks[target_path][0], "import", imported, 0.85)
        source_stems = {Path(path).stem.lower(): path for path in self.path_chunks if "test" not in Path(path).stem.lower()}
        for path, ids in self.path_chunks.items():
            stem = Path(path).stem.lower().removeprefix("test_").removesuffix("_test")
            if "test" in Path(path).stem.lower() and stem in source_stems:
                self._add_edge(ids[0], self.path_chunks[source_stems[stem]][0], "test", stem, 0.9)
        self.manifest["graph_edges"] = sum(len(edges) for edges in self.adjacency.values()) // 2
        self.manifest["indexed_terms"] = len(self.token_chunks)

    def build_sync(self, objective: str, paths: list[str] | None, force: bool) -> str:
        if not self.enabled:
            return "Project perspective indexing is disabled by HUMOID_PERSPECTIVE_ENABLED."
        if self.chunks and not force:
            return self.status_text("Perspective already built; reuse search_project_perspective.")
        files = self.scan(paths)
        chunks = [chunk for path in files for chunk in self.chunk_file(path)]
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.path_chunks = defaultdict(list)
        for chunk in chunks:
            self.path_chunks[chunk.path].append(chunk.chunk_id)
        self.objective = objective.strip()
        self.manifest = {
            "files": len(files), "chunks": len(chunks),
            "characters": sum(len(chunk.text) for chunk in chunks),
            "symbols": sum(len(chunk.symbols) for chunk in chunks),
            "tests": [path.relative_to(self.root).as_posix() for path in files if "test" in path.name.lower()][:30],
        }
        self.dirty_paths.clear()
        self._rebuild_indexes()
        self.backend = "local-vector-fallback+constraint-graph"
        return self.status_text("Built project perspective.")

    async def build(self, objective: str, paths: list[str] | None = None, force: bool = False) -> str:
        return await asyncio.to_thread(self.build_sync, objective, paths, force)

    async def _failure_bias(self, query: str) -> FailureBias:
        if not self.failure_memory_enabled or not any(marker in query.lower() for marker in FAILURE_MARKERS):
            return FailureBias(status="skipped: non-debug query")
        if self._memory is None:
            try:
                from .memory import MemoryRouter
                self._memory = MemoryRouter(self.settings)
            except Exception as exc:
                return FailureBias(status=f"unavailable: {type(exc).__name__}: {exc}")
        if not self._memory_initialized:
            try:
                await self._memory.initialize()
                self._memory_initialized = True
            except Exception as exc:
                return FailureBias(status=f"unavailable: {type(exc).__name__}: {exc}")
        try:
            hits = await self._memory.search(
                f"{query} bug failure regression traceback exception fixed error",
                self.failure_memory_limit,
            )
        except Exception as exc:
            return FailureBias(status=f"unavailable: {type(exc).__name__}: {exc}")
        path_scores: dict[str, float] = defaultdict(float)
        symbol_scores: dict[str, float] = defaultdict(float)
        evidence: list[str] = []
        for hit in hits:
            text = str(getattr(hit, "text", ""))
            metadata = getattr(hit, "metadata", {}) or {}
            paths = [match.group(0).lower() for match in FILE_RE.finditer(text)]
            metadata_paths = metadata.get("files") or metadata.get("paths") or []
            if isinstance(metadata_paths, str):
                metadata_paths = [metadata_paths]
            paths.extend(str(value).lower() for value in metadata_paths)
            symbols = [token.lower() for token in TOKEN_RE.findall(text) if "_" in token or any(c.isupper() for c in token[1:])]
            metadata_symbols = metadata.get("symbols") or []
            if isinstance(metadata_symbols, str):
                metadata_symbols = [metadata_symbols]
            symbols.extend(str(value).lower() for value in metadata_symbols)
            base = 0.5 + min(1.5, max(0.0, float(getattr(hit, "score", 0.0))))
            if str(getattr(hit, "validation_status", "")) == "verified":
                base += 0.35
            for path in paths:
                path_scores[path] = max(path_scores[path], base)
            for symbol in symbols:
                symbol_scores[symbol] = max(symbol_scores[symbol], base * 0.7)
            if paths or symbols:
                evidence.append(f"score={base:.2f}; paths={','.join(dict.fromkeys(paths)) or 'none'}; symbols={','.join(dict.fromkeys(symbols)) or 'none'}; excerpt={text.replace(chr(10), ' ')[:240]}")
        return FailureBias(dict(path_scores), dict(symbol_scores), evidence[:4], str(getattr(self._memory, "status", "ready")))

    def _fuzzy_terms(self, term: str) -> list[tuple[str, float]]:
        grams = trigrams(term)
        candidates: Counter[str] = Counter()
        for gram in grams:
            candidates.update(self.trigram_terms.get(gram, set()))
        output = []
        for candidate, overlap in candidates.most_common(30):
            similarity = overlap / max(1, len(grams | trigrams(candidate)))
            if candidate != term and similarity >= 0.42:
                output.append((candidate, similarity))
        return output[:4]

    def _failure_score(self, chunk: PerspectiveChunk, bias: FailureBias) -> tuple[float, list[str]]:
        score, reasons = 0.0, []
        for path, weight in bias.path_scores.items():
            if chunk.path.lower() == path or chunk.path.lower().endswith(f"/{path}") or Path(chunk.path).name.lower() == Path(path).name:
                score = max(score, weight); reasons.append(f"past failure touched {path}")
        symbols = {value.lower() for value in (*chunk.symbols, *chunk.references)}
        for symbol, weight in bias.symbol_scores.items():
            if symbol in symbols:
                score = max(score, weight); reasons.append(f"past failure mentioned {symbol}")
        return min(2.0, score), reasons[:3]

    def _score(self, query: str, bias: FailureBias) -> dict[str, PerspectiveHit]:
        query_terms = tuple(dict.fromkeys(term for term in tokens(query) if term not in STOP_WORDS))[:24]
        query_symbols = {token.lower() for token in TOKEN_RE.findall(query) if "_" in token or any(c.isupper() for c in token[1:])}
        query_roles = {role for role, hints in ROLE_HINTS.items() if any(hint in query.lower() for hint in hints)}
        candidate_ids: set[str] = set()
        reasons: dict[str, list[str]] = defaultdict(list)
        for term in query_terms:
            exact = self.token_chunks.get(term, set())
            for chunk_id in exact:
                candidate_ids.add(chunk_id); reasons[chunk_id].append(f"exact term {term}")
            if not exact:
                for fuzzy, similarity in self._fuzzy_terms(term):
                    for chunk_id in self.token_chunks.get(fuzzy, set()):
                        candidate_ids.add(chunk_id); reasons[chunk_id].append(f"identifier-near {term}≈{fuzzy} ({similarity:.2f})")
        for symbol in query_symbols:
            for chunk_id in self.symbol_definitions.get(symbol, set()):
                candidate_ids.add(chunk_id); reasons[chunk_id].append(f"defines {symbol}")
            for chunk_id in self.symbol_references.get(symbol, set()):
                candidate_ids.add(chunk_id); reasons[chunk_id].append(f"references {symbol}")
        for path in bias.path_scores:
            for indexed_path, ids in self.path_chunks.items():
                if indexed_path.lower().endswith(path) or Path(indexed_path).name.lower() == Path(path).name:
                    candidate_ids.update(ids)
        query_vector = self.embedder.embed(query)
        if not candidate_ids:
            candidate_ids = {
                chunk.chunk_id for chunk in sorted(
                    self.chunks.values(),
                    key=lambda chunk: sum(a * b for a, b in zip(query_vector, chunk.vector, strict=False)),
                    reverse=True,
                )[:max(24, self.default_limit * 4)]
            }
        hits: dict[str, PerspectiveHit] = {}
        labels = max(1, len(query_terms) + len(query_symbols) + len(query_roles))
        for chunk_id in candidate_ids:
            chunk = self.chunks[chunk_id]
            haystack = set(tokens(f"{chunk.path} {' '.join(chunk.symbols)} {' '.join(chunk.references)} {' '.join(chunk.roles)} {chunk.text}"))
            matched = sum(term in haystack for term in query_terms)
            matched += sum(symbol in {value.lower() for value in (*chunk.symbols, *chunk.references)} for symbol in query_symbols)
            matched += len(query_roles & set(chunk.roles))
            constraint = matched / labels
            lexical = lexical_score(query, f"{chunk.path} {' '.join(chunk.symbols)} {' '.join(chunk.references)} {chunk.text}")
            vector = max(0.0, sum(a * b for a, b in zip(query_vector, chunk.vector, strict=False)))
            failure, failure_reasons = self._failure_score(chunk, bias)
            score = 0.5 * constraint + 0.25 * lexical + 0.1 * vector + 0.2 * min(1.0, failure)
            hits[chunk_id] = PerspectiveHit(chunk, score, vector, lexical, constraint, failure_score=failure, reasons=tuple(dict.fromkeys([*reasons[chunk_id], *failure_reasons]))[:6])
        return hits

    def _shortest_path(self, starts: set[str], target: str) -> tuple[list[str], list[GraphEdge]] | None:
        if target in starts:
            return [target], []
        queue: deque[tuple[str, int]] = deque((start, 0) for start in starts)
        parent: dict[str, tuple[str, GraphEdge] | None] = {start: None for start in starts}
        while queue:
            node, depth = queue.popleft()
            if depth >= self.graph_max_hops:
                continue
            for neighbor, edge in sorted(self.adjacency.get(node, {}).items(), key=lambda item: item[1].weight, reverse=True)[:self.graph_fanout]:
                if neighbor in parent:
                    continue
                parent[neighbor] = (node, edge)
                if neighbor == target:
                    nodes, edges, current = [target], [], target
                    while parent[current] is not None:
                        previous, path_edge = parent[current]
                        nodes.append(previous); edges.append(path_edge); current = previous
                    return list(reversed(nodes)), list(reversed(edges))
                queue.append((neighbor, depth + 1))
        return None

    def _minimal_subgraph(self, hits: dict[str, PerspectiveHit], limit: int) -> tuple[list[PerspectiveHit], list[GraphEdge]]:
        if not hits:
            return [], []
        ranked = sorted(hits, key=lambda chunk_id: hits[chunk_id].score, reverse=True)
        selected = {ranked[0]}; hits[ranked[0]].anchor = True
        selected_edges: dict[tuple[str, str, str], GraphEdge] = {}
        while len(selected) < limit:
            best = None
            for candidate in ranked:
                if candidate in selected:
                    continue
                path = self._shortest_path(selected, candidate)
                nodes, edges = path if path else ([candidate], [])
                additions = [node for node in nodes if node not in selected]
                if not additions or len(selected) + len(additions) > limit:
                    continue
                edge_strength = sum(edge.weight for edge in edges) / max(1, len(edges))
                gain = hits[candidate].score + 0.12 * edge_strength - 0.12 * max(0, len(additions) - 1) - (0.25 if not path else 0)
                if best is None or gain > best[0]:
                    best = gain, candidate, nodes, edges
            if best is None:
                break
            _, candidate, nodes, edges = best
            hits[candidate].anchor = True
            for node in nodes:
                if node not in hits:
                    hits[node] = PerspectiveHit(self.chunks[node], 0.05, 0.0, 0.0, graph_score=0.35, reasons=("bridge node in minimal behavior path",))
                selected.add(node)
            for edge in edges:
                selected_edges[tuple(sorted((edge.source, edge.target))) + (edge.kind,)] = edge
        ordered = sorted((hits[node] for node in selected), key=lambda hit: (not hit.anchor, -hit.score, hit.chunk.path, hit.chunk.chunk_index))
        return ordered, list(selected_edges.values())

    def render_packet(self, query: str, hits: list[PerspectiveHit], edges: list[GraphEdge], bias: FailureBias, budget: int) -> str:
        parts = [
            "[PROJECT PERSPECTIVE: minimal connected behavior subgraph]\n"
            f"objective={self.objective or 'unspecified'}\nquery={query}\nbackend={self.backend}\n"
            f"failure_memory={bias.status}\nmanifest={json.dumps(self.manifest, sort_keys=True)}\n"
            "Ranking is constraint-first; vectors are tie-breakers or cold-start fallback."
        ]
        used = len(parts[0])
        if edges:
            block = "\n\n[BEHAVIOR GRAPH EDGES]\n" + "\n".join(
                f"- {self.chunks[edge.source].citation} --{edge.kind}:{edge.label}--> {self.chunks[edge.target].citation} weight={edge.weight:.2f}"
                for edge in sorted(edges, key=lambda item: item.weight, reverse=True)[:30]
            )
            parts.append(block); used += len(block)
        if bias.evidence:
            block = "\n\n[FAILURE MEMORY: historical bug-zone evidence]\n" + "\n".join(f"- {item}" for item in bias.evidence)
            if used + len(block) <= budget:
                parts.append(block); used += len(block)
        folded = []
        for hit in hits:
            chunk = hit.chunk
            block = (
                f"\n\n[PERSPECTIVE NODE {chunk.chunk_id} | {chunk.citation} | anchor={str(hit.anchor).lower()} | "
                f"score={hit.score:.3f} constraint={hit.constraint_score:.3f} graph={hit.graph_score:.3f} "
                f"failure={hit.failure_score:.3f} vector={hit.vector_score:.3f} lexical={hit.lexical_score:.3f}]\n"
                f"definitions={', '.join(chunk.symbols) or 'none'}\nreferences={', '.join(chunk.references[:20]) or 'none'}\n"
                f"roles={', '.join(chunk.roles) or 'none'}\nreasons={'; '.join(hit.reasons) or 'graph-selected'}\n{chunk.text}"
            )
            if used + len(block) <= budget:
                parts.append(block); used += len(block)
            else:
                folded.append(hit)
        if folded:
            parts.append("\n\n[CONTEXT ACCORDION: lower-value nodes folded]\n" + "\n".join(
                f"- {hit.chunk.path}: ids={hit.chunk.chunk_id}; symbols={','.join(hit.chunk.symbols) or 'none'}"
                for hit in folded
            ))
        return "".join(parts)[:budget]

    def search_sync(self, query: str, limit: int | None, max_chars: int | None, bias: FailureBias) -> str:
        if not self.chunks:
            return "Perspective index is empty. Call build_project_perspective first."
        hits = self._score(query, bias)
        selected, edges = self._minimal_subgraph(hits, max(1, min(50, limit or self.default_limit)))
        return self.render_packet(query, selected, edges, bias, max(1000, min(100_000, max_chars or self.packet_max_chars)))

    async def search(self, query: str, limit: int | None = None, max_chars: int | None = None) -> str:
        bias = await self._failure_bias(query)
        return await asyncio.to_thread(self.search_sync, query, limit, max_chars, bias)

    def expand_sync(self, chunk_ids: list[str], radius: int, max_chars: int | None) -> str:
        selected: dict[str, PerspectiveChunk] = {}
        radius = max(0, min(4, radius))
        for chunk_id in chunk_ids[:20]:
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue
            ids = self.path_chunks.get(chunk.path, [])
            position = ids.index(chunk_id) if chunk_id in ids else 0
            for neighbor in ids[max(0, position - radius):position + radius + 1]:
                selected[neighbor] = self.chunks[neighbor]
            for neighbor in list(self.adjacency.get(chunk_id, {}))[:radius * 2]:
                selected[neighbor] = self.chunks[neighbor]
        if not selected:
            return "No requested perspective chunk IDs were found. Search again for current IDs."
        packet = "[EXPANDED PROJECT PERSPECTIVE]" + "".join(
            f"\n\n[{chunk.chunk_id} | {chunk.citation}]\ndefinitions={', '.join(chunk.symbols) or 'none'}\n"
            f"references={', '.join(chunk.references[:20]) or 'none'}\n{chunk.text}"
            for chunk in sorted(selected.values(), key=lambda item: (item.path, item.chunk_index))
        )
        budget = max(1000, min(100_000, max_chars or self.packet_max_chars))
        return packet if len(packet) <= budget else packet[:budget] + "\n\n[expanded context truncated]"

    async def expand(self, chunk_ids: list[str], radius: int = 1, max_chars: int | None = None) -> str:
        return await asyncio.to_thread(self.expand_sync, chunk_ids, radius, max_chars)

    async def notify_file_changed(self, relative: str, content: str) -> None:
        if not self.chunks:
            return
        path = safe_path(self.root, self.root / relative)
        normalized = path.relative_to(self.root).as_posix()
        for chunk_id in self.path_chunks.pop(normalized, []):
            self.chunks.pop(chunk_id, None)
        for chunk in self.chunk_file(path, content):
            self.chunks[chunk.chunk_id] = chunk; self.path_chunks[normalized].append(chunk.chunk_id)
        self.dirty_paths.add(normalized)
        self.manifest["chunks"] = len(self.chunks)
        self._rebuild_indexes()

    async def notify_file_deleted(self, relative: str) -> None:
        if not self.chunks:
            return
        normalized = safe_path(self.root, self.root / relative).relative_to(self.root).as_posix()
        for chunk_id in self.path_chunks.pop(normalized, []):
            self.chunks.pop(chunk_id, None)
        self.dirty_paths.add(normalized)
        self.manifest["chunks"] = len(self.chunks)
        self._rebuild_indexes()

    def status_text(self, prefix: str = "Perspective status.") -> str:
        return (
            f"{prefix} backend={self.backend}; collection={self.collection_name}; files={self.manifest.get('files', 0)}; "
            f"chunks={len(self.chunks)}; symbols={self.manifest.get('symbols', 0)}; edges={self.manifest.get('graph_edges', 0)}; "
            f"dirty_files={len(self.dirty_paths)}; objective={self.objective or 'none'}"
        )

    def status_json(self) -> str:
        return json.dumps({
            "enabled": self.enabled, "backend": self.backend, "collection": self.collection_name,
            "objective": self.objective, "manifest": self.manifest, "dirty_paths": sorted(self.dirty_paths),
            "graph_max_hops": self.graph_max_hops, "graph_fanout": self.graph_fanout,
            "failure_memory_enabled": self.failure_memory_enabled,
            "failure_memory_status": str(getattr(self._memory, "status", "not-initialized")),
        }, indent=2)

    def clear_sync(self) -> None:
        self.chunks.clear(); self.path_chunks.clear(); self.dirty_paths.clear()
        self.token_chunks.clear(); self.symbol_definitions.clear(); self.symbol_references.clear()
        self.trigram_terms.clear(); self.adjacency.clear(); self.manifest, self.objective = {}, ""
        self.collection_name = f"HumoidPerspective{uuid4().hex[:16]}"; self.backend = "cleared"

    def _close_sync(self) -> None:
        if self._closed:
            return
        self._closed = True; self.clear_sync()

    async def clear(self) -> str:
        await asyncio.to_thread(self.clear_sync)
        if self._owns_memory and self._memory is not None and self._memory_initialized:
            try:
                await self._memory.close()
            except Exception:
                pass
            self._memory = None; self._memory_initialized = False
        return "Cleared temporary project perspective collection and local chunk/graph cache."
