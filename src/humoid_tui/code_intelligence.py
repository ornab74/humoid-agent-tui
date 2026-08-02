from __future__ import annotations

"""Incremental repository intelligence for impact-aware code localization.

The constraint graph answers "where is behavior likely implemented?".  This
module answers the next questions before an edit:

* which concrete symbols participate;
* which callers, subclasses, imports, and tests form the change frontier;
* what is the smallest likely blast radius;
* which files should be retrieved to prove or falsify the hypothesis.

Python receives a real AST pass. Other text languages use conservative symbol,
call, import, and test heuristics. Results are cached by content digest so a
large repository only reparses changed files.
"""

import ast
import hashlib
import json
import os
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

CODE_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go",
    ".rs", ".java", ".kt", ".kts", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".scala", ".dart", ".sh", ".sql",
}
IGNORED = {
    ".git", ".venv", "venv", "node_modules", "vendor", "target", "build",
    "dist", ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "__pycache__", ".tox", ".nox", ".gradle", ".dart_tool", ".humoid",
}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")
GENERIC_DEF_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|function|interface|type|struct|enum|trait|fn|func|"
    r"public\s+class|private\s+class|export\s+(?:default\s+)?(?:class|function|const|let|var))"
    r"\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)
GENERIC_CALL_RE = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")
GENERIC_IMPORT_RE = re.compile(
    r"(?:^\s*(?:from|import)\s+([\w.]+)|\bfrom\s+[\"']([^\"']+)[\"']|"
    r"\brequire\(\s*[\"']([^\"']+)[\"']\s*\))",
    re.MULTILINE,
)
STOP = {
    "and", "are", "change", "code", "edit", "file", "fix", "for", "from",
    "how", "implement", "in", "is", "of", "on", "or", "the", "this", "to",
    "update", "what", "where", "with",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode(errors="replace")).hexdigest()


def _tokens(text: str) -> set[str]:
    return {value.lower() for value in TOKEN_RE.findall(text) if value.lower() not in STOP}


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _qualified_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


@dataclass(slots=True)
class SymbolRecord:
    symbol_id: str
    name: str
    qualified_name: str
    kind: str
    path: str
    line_start: int
    line_end: int
    parent: str = ""
    bases: tuple[str, ...] = ()
    decorators: tuple[str, ...] = ()
    calls: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    signature: str = ""


@dataclass(slots=True)
class IntelligenceEdge:
    source: str
    target: str
    kind: str
    confidence: float
    evidence: str


@dataclass(slots=True)
class FileRecord:
    path: str
    digest: str
    language: str
    symbols: list[SymbolRecord] = field(default_factory=list)
    imports: tuple[str, ...] = ()
    parse_error: str = ""


@dataclass(slots=True)
class ImpactNode:
    symbol_id: str
    path: str
    symbol: str
    kind: str
    distance: int
    score: float
    reasons: tuple[str, ...]


@dataclass(slots=True)
class ImpactReport:
    query: str
    anchors: list[ImpactNode]
    frontier: list[ImpactNode]
    edges: list[IntelligenceEdge]
    files: list[str]
    tests: list[str]
    risk_score: float
    risk_reasons: list[str]
    uncertainty: list[str]

    def render(self, max_chars: int = 12000) -> str:
        lines = [
            "[STATIC IMPACT ANALYSIS: incremental AST + conservative fallback]",
            f"query={self.query}",
            f"risk_score={self.risk_score:.2f}",
            f"files={','.join(self.files) or 'none'}",
            f"tests={','.join(self.tests) or 'none'}",
            f"risk_reasons={'; '.join(self.risk_reasons) or 'none'}",
            f"uncertainty={'; '.join(self.uncertainty) or 'none'}",
            "",
            "ANCHORS",
        ]
        lines.extend(
            f"- {node.path}:{node.symbol} kind={node.kind} score={node.score:.3f} reasons={'; '.join(node.reasons)}"
            for node in self.anchors
        )
        lines.append("")
        lines.append("CHANGE FRONTIER")
        lines.extend(
            f"- d={node.distance} {node.path}:{node.symbol} kind={node.kind} score={node.score:.3f} reasons={'; '.join(node.reasons)}"
            for node in self.frontier
        )
        lines.append("")
        lines.append("TYPED EDGES")
        lines.extend(
            f"- {edge.source} --{edge.kind}:{edge.confidence:.2f}--> {edge.target} evidence={edge.evidence}"
            for edge in self.edges[:80]
        )
        return "\n".join(lines)[:max_chars]


class PythonAnalyzer(ast.NodeVisitor):
    def __init__(self, path: str, source: str) -> None:
        self.path, self.source = path, source
        self.stack: list[str] = []
        self.symbols: list[SymbolRecord] = []
        self.module_imports: list[str] = []

    def _record(self, node: ast.AST, name: str, kind: str, *, bases: Iterable[str] = (), decorators: Iterable[str] = (), signature: str = "") -> None:
        parent = ".".join(self.stack)
        qualified = f"{parent}.{name}" if parent else name
        calls = []
        imports = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                value = _qualified_name(child.func)
                if value:
                    calls.append(value)
            elif isinstance(child, ast.Import):
                imports.extend(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom):
                module = child.module or ""
                imports.extend(f"{module}.{alias.name}".strip(".") for alias in child.names)
        symbol_id = f"{self.path}::{qualified}"
        self.symbols.append(SymbolRecord(
            symbol_id=symbol_id,
            name=name,
            qualified_name=qualified,
            kind=kind,
            path=self.path,
            line_start=int(getattr(node, "lineno", 1)),
            line_end=int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
            parent=parent,
            bases=tuple(dict.fromkeys(bases)),
            decorators=tuple(dict.fromkeys(decorators)),
            calls=tuple(dict.fromkeys(calls)),
            imports=tuple(dict.fromkeys(imports)),
            signature=signature,
        ))

    @staticmethod
    def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        args = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        return f"{node.name}({', '.join(args)})"

    def visit_Import(self, node: ast.Import) -> Any:
        self.module_imports.extend(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = node.module or ""
        self.module_imports.extend(f"{module}.{alias.name}".strip(".") for alias in node.names)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._record(
            node, node.name, "class",
            bases=(_qualified_name(base) for base in node.bases),
            decorators=(_qualified_name(item) for item in node.decorator_list),
        )
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._record(
            node, node.name, "method" if self.stack else "function",
            decorators=(_qualified_name(item) for item in node.decorator_list),
            signature=self._signature(node),
        )
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.visit_FunctionDef(node)


class CodeIntelligenceIndex:
    def __init__(self, root: Path, cache_path: Path | None = None) -> None:
        self.root = root.resolve()
        self.cache_path = cache_path or self.root / ".humoid" / "code-intelligence-v2.json"
        self.files: dict[str, FileRecord] = {}
        self.symbols: dict[str, SymbolRecord] = {}
        self.name_index: dict[str, set[str]] = defaultdict(set)
        self.path_index: dict[str, set[str]] = defaultdict(set)
        self.edges: list[IntelligenceEdge] = []
        self.forward: dict[str, list[IntelligenceEdge]] = defaultdict(list)
        self.reverse: dict[str, list[IntelligenceEdge]] = defaultdict(list)
        self.stats: dict[str, int] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        try:
            payload = json.loads(self.cache_path.read_text())
        except (OSError, ValueError, TypeError):
            return
        for item in payload.get("files", []):
            symbols = [SymbolRecord(**symbol) for symbol in item.get("symbols", [])]
            record = FileRecord(
                path=item["path"], digest=item["digest"], language=item.get("language", "generic"),
                symbols=symbols, imports=tuple(item.get("imports", [])), parse_error=item.get("parse_error", ""),
            )
            self.files[record.path] = record
        self._rebuild_graph()

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "files": [
                {
                    "path": record.path,
                    "digest": record.digest,
                    "language": record.language,
                    "symbols": [asdict(symbol) for symbol in record.symbols],
                    "imports": list(record.imports),
                    "parse_error": record.parse_error,
                }
                for record in sorted(self.files.values(), key=lambda value: value.path)
            ],
        }
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        temporary.replace(self.cache_path)

    def _scan(self, paths: list[str] | None = None) -> list[Path]:
        roots = [self.root] if not paths else [(self.root / value).resolve() for value in paths]
        output: dict[str, Path] = {}
        for start in roots:
            candidates = [start] if start.is_file() else start.rglob("*")
            for path in candidates:
                if not path.is_file() or path.suffix.lower() not in CODE_SUFFIXES:
                    continue
                relative = path.relative_to(self.root)
                if any(part in IGNORED for part in relative.parts[:-1]):
                    continue
                output[relative.as_posix()] = path
        return [output[key] for key in sorted(output)]

    def _analyze_python(self, path: str, source: str, digest: str) -> FileRecord:
        try:
            tree = ast.parse(source, filename=path, type_comments=True)
        except SyntaxError as exc:
            return FileRecord(path, digest, "python", [], (), f"{exc.msg} at {exc.lineno}:{exc.offset}")
        analyzer = PythonAnalyzer(path, source)
        analyzer.visit(tree)
        return FileRecord(path, digest, "python", analyzer.symbols, tuple(dict.fromkeys(analyzer.module_imports)))

    def _analyze_generic(self, path: str, source: str, digest: str) -> FileRecord:
        symbols = []
        calls = tuple(dict.fromkeys(value for value in GENERIC_CALL_RE.findall(source) if value not in {"if", "for", "while", "return"}))[:200]
        imports = []
        for match in GENERIC_IMPORT_RE.finditer(source):
            value = next((item for item in match.groups() if item), "")
            if value:
                imports.append(value)
        lines = source.splitlines()
        for match in GENERIC_DEF_RE.finditer(source):
            name = match.group(1)
            line = source.count("\n", 0, match.start()) + 1
            symbols.append(SymbolRecord(
                symbol_id=f"{path}::{name}", name=name, qualified_name=name, kind="symbol",
                path=path, line_start=line, line_end=min(len(lines), line + 80), calls=calls,
                imports=tuple(dict.fromkeys(imports)),
            ))
        if not symbols:
            symbols.append(SymbolRecord(
                symbol_id=f"{path}::<module>", name="<module>", qualified_name="<module>", kind="module",
                path=path, line_start=1, line_end=max(1, len(lines)), calls=calls,
                imports=tuple(dict.fromkeys(imports)),
            ))
        return FileRecord(path, digest, path.suffix.lower().lstrip(".") or "generic", symbols, tuple(dict.fromkeys(imports)))

    def build(self, paths: list[str] | None = None) -> dict[str, int]:
        scanned = self._scan(paths)
        seen = set()
        parsed = reused = errors = 0
        for file_path in scanned:
            relative = file_path.relative_to(self.root).as_posix()
            seen.add(relative)
            try:
                source = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            digest = _sha(source)
            prior = self.files.get(relative)
            if prior and prior.digest == digest:
                reused += 1
                continue
            record = self._analyze_python(relative, source, digest) if file_path.suffix.lower() in {".py", ".pyi"} else self._analyze_generic(relative, source, digest)
            self.files[relative] = record
            parsed += 1
            errors += bool(record.parse_error)
        if not paths:
            for stale in set(self.files) - seen:
                self.files.pop(stale, None)
        self._rebuild_graph()
        self._save_cache()
        self.stats = {
            "files": len(self.files), "symbols": len(self.symbols), "edges": len(self.edges),
            "parsed": parsed, "reused": reused, "parse_errors": errors,
        }
        return dict(self.stats)

    def _rebuild_graph(self) -> None:
        self.symbols = {}
        self.name_index = defaultdict(set)
        self.path_index = defaultdict(set)
        self.edges = []
        self.forward = defaultdict(list)
        self.reverse = defaultdict(list)
        for record in self.files.values():
            for symbol in record.symbols:
                self.symbols[symbol.symbol_id] = symbol
                self.name_index[symbol.name.lower()].add(symbol.symbol_id)
                self.name_index[symbol.qualified_name.lower()].add(symbol.symbol_id)
                self.path_index[symbol.path].add(symbol.symbol_id)
        for symbol in self.symbols.values():
            for called in symbol.calls:
                target_name = called.rsplit(".", 1)[-1].lower()
                for target in sorted(self.name_index.get(target_name, set()))[:30]:
                    self._edge(symbol.symbol_id, target, "calls", 0.95 if called == target_name else 0.82, called)
            for base in symbol.bases:
                target_name = base.rsplit(".", 1)[-1].lower()
                for target in sorted(self.name_index.get(target_name, set()))[:20]:
                    self._edge(symbol.symbol_id, target, "inherits", 0.98, base)
            if symbol.parent:
                parent_name = symbol.parent.split(".")[-1].lower()
                for target in sorted(self.name_index.get(parent_name, set()))[:5]:
                    if target != symbol.symbol_id and self.symbols[target].path == symbol.path:
                        self._edge(target, symbol.symbol_id, "contains", 1.0, symbol.parent)
        module_alias: dict[str, str] = {}
        for path in self.files:
            module_alias[Path(path).stem.lower()] = path
            module_alias[str(Path(path).with_suffix("")).replace("/", ".").lower()] = path
        for record in self.files.values():
            source_symbols = sorted(self.path_index.get(record.path, set()))
            if not source_symbols:
                continue
            for imported in record.imports:
                key = imported.lower()
                target_path = module_alias.get(key) or module_alias.get(key.rsplit(".", 1)[0]) or module_alias.get(key.rsplit(".", 1)[-1])
                if target_path:
                    for target in sorted(self.path_index.get(target_path, set()))[:3]:
                        self._edge(source_symbols[0], target, "imports", 0.78, imported)
        source_by_stem = {Path(path).stem.lower(): path for path in self.files if "test" not in Path(path).stem.lower()}
        for path in self.files:
            stem = Path(path).stem.lower().removeprefix("test_").removesuffix("_test")
            if "test" in Path(path).stem.lower() and stem in source_by_stem:
                test_ids = sorted(self.path_index.get(path, set()))
                source_ids = sorted(self.path_index.get(source_by_stem[stem], set()))
                if test_ids and source_ids:
                    self._edge(test_ids[0], source_ids[0], "tests", 0.92, stem)

    def _edge(self, source: str, target: str, kind: str, confidence: float, evidence: str) -> None:
        if source == target or source not in self.symbols or target not in self.symbols:
            return
        edge = IntelligenceEdge(source, target, kind, confidence, evidence)
        self.edges.append(edge)
        self.forward[source].append(edge)
        self.reverse[target].append(edge)

    def _anchor_scores(self, query: str) -> dict[str, tuple[float, list[str]]]:
        terms = _tokens(query)
        scores: dict[str, tuple[float, list[str]]] = {}
        for symbol_id, symbol in self.symbols.items():
            name_terms = _tokens(f"{symbol.name} {symbol.qualified_name} {symbol.signature}")
            path_terms = _tokens(symbol.path.replace("/", " "))
            exact = len(terms & name_terms)
            path_overlap = len(terms & path_terms)
            call_overlap = len(terms & _tokens(" ".join(symbol.calls)))
            score = 1.8 * exact + 0.7 * path_overlap + 0.4 * call_overlap
            reasons = []
            if exact:
                reasons.append(f"symbol terms={exact}")
            if path_overlap:
                reasons.append(f"path terms={path_overlap}")
            if call_overlap:
                reasons.append(f"call terms={call_overlap}")
            if any(word in symbol.path.lower() for word in ("test", "spec")) and any(word in query.lower() for word in ("test", "verify", "regression")):
                score += 0.8
                reasons.append("test intent")
            if score > 0:
                scores[symbol_id] = (score, reasons)
        if not scores:
            for symbol_id, symbol in list(self.symbols.items())[:50]:
                overlap = len(terms & _tokens(f"{symbol.path} {' '.join(symbol.calls)}"))
                if overlap:
                    scores[symbol_id] = (float(overlap), ["fallback lexical overlap"])
        return scores

    def impact(self, query: str, *, max_depth: int = 3, max_nodes: int = 30) -> ImpactReport:
        if not self.symbols:
            self.build()
        scored = self._anchor_scores(query)
        ranked = sorted(scored, key=lambda key: scored[key][0], reverse=True)[:8]
        anchors = [
            ImpactNode(key, self.symbols[key].path, self.symbols[key].qualified_name, self.symbols[key].kind, 0, scored[key][0], tuple(scored[key][1]))
            for key in ranked
        ]
        queue: deque[tuple[str, int, float, tuple[str, ...]]] = deque(
            (node.symbol_id, 0, node.score, node.reasons) for node in anchors
        )
        best: dict[str, ImpactNode] = {node.symbol_id: node for node in anchors}
        used_edges: dict[tuple[str, str, str], IntelligenceEdge] = {}
        while queue and len(best) < max_nodes:
            current, depth, inherited, inherited_reasons = queue.popleft()
            if depth >= max_depth:
                continue
            neighbors = [*self.forward.get(current, []), *self.reverse.get(current, [])]
            for edge in sorted(neighbors, key=lambda value: value.confidence, reverse=True):
                neighbor = edge.target if edge.source == current else edge.source
                distance = depth + 1
                propagation = inherited * edge.confidence * (0.72 ** distance)
                reasons = tuple(dict.fromkeys((*inherited_reasons, f"{edge.kind}:{edge.evidence}")))[:5]
                prior = best.get(neighbor)
                if prior and prior.score >= propagation:
                    continue
                symbol = self.symbols[neighbor]
                best[neighbor] = ImpactNode(neighbor, symbol.path, symbol.qualified_name, symbol.kind, distance, propagation, reasons)
                used_edges[(edge.source, edge.target, edge.kind)] = edge
                queue.append((neighbor, distance, propagation, reasons))
                if len(best) >= max_nodes:
                    break
        frontier = sorted((node for node in best.values() if node.distance > 0), key=lambda value: (value.distance, -value.score))
        files = list(dict.fromkeys(node.path for node in [*anchors, *frontier]))
        tests = [path for path in files if "test" in Path(path).name.lower() or "tests" in Path(path).parts]
        kinds = Counter(edge.kind for edge in used_edges.values())
        risk = min(10.0, 1.0 + len(files) * 0.35 + kinds["inherits"] * 0.8 + kinds["calls"] * 0.2 + kinds["imports"] * 0.15)
        risk_reasons = []
        if len(files) > 6:
            risk_reasons.append(f"cross-file frontier={len(files)}")
        if kinds["inherits"]:
            risk_reasons.append(f"inheritance edges={kinds['inherits']}")
        if kinds["calls"] > 8:
            risk_reasons.append(f"call fanout={kinds['calls']}")
        if not tests:
            risk += 1.0
            risk_reasons.append("no linked tests found")
        uncertainty = []
        parse_errors = [record.path for record in self.files.values() if record.parse_error]
        if parse_errors:
            uncertainty.append(f"parse errors in {','.join(parse_errors[:6])}")
        if any(record.language != "python" for record in self.files.values()):
            uncertainty.append("non-Python edges use conservative heuristics")
        if not anchors:
            uncertainty.append("no strong symbol anchor")
        return ImpactReport(
            query=query,
            anchors=anchors,
            frontier=frontier,
            edges=sorted(used_edges.values(), key=lambda value: value.confidence, reverse=True),
            files=files,
            tests=tests,
            risk_score=min(10.0, risk),
            risk_reasons=risk_reasons,
            uncertainty=uncertainty,
        )

    def benchmark(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        results = []
        precision_values = []
        recall_values = []
        reciprocal_ranks = []
        for case in cases:
            query = str(case.get("query", ""))
            expected = {str(value).replace("\\", "/") for value in case.get("expected_files", [])}
            report = self.impact(query, max_nodes=int(case.get("max_nodes", 30)))
            predicted = report.files
            predicted_set = set(predicted)
            true_positive = len(expected & predicted_set)
            precision = true_positive / max(1, len(predicted_set))
            recall = true_positive / max(1, len(expected))
            rank = next((index for index, path in enumerate(predicted, 1) if path in expected), None)
            reciprocal = 1.0 / rank if rank else 0.0
            precision_values.append(precision)
            recall_values.append(recall)
            reciprocal_ranks.append(reciprocal)
            results.append({
                "query": query, "expected_files": sorted(expected), "predicted_files": predicted,
                "precision": precision, "recall": recall, "reciprocal_rank": reciprocal,
            })
        return {
            "cases": len(results),
            "mean_precision": sum(precision_values) / max(1, len(precision_values)),
            "mean_recall": sum(recall_values) / max(1, len(recall_values)),
            "mean_reciprocal_rank": sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)),
            "results": results,
        }
