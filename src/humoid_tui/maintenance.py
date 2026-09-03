from __future__ import annotations

"""Maintenance intelligence for impact-aware development and long-lived codebases.

This module deliberately separates three kinds of evidence:

* current working-tree structure (authoritative for the code that exists now),
* semantic change fingerprints (what a proposed/committed edit structurally changed),
* historical procedures (hints from previously validated edit trajectories).

None of the heuristic scores below are claims of code quality.  They are bounded
maintenance-pressure signals intended to tell an agent where inspection and
validation effort is most likely to pay off.
"""

import ast
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .code_intelligence import CodeIntelligenceIndex, ImpactReport

_TODO_RE = re.compile(r"\b(?:TODO|FIXME|HACK|XXX)\b", re.I)
_TEST_RE = re.compile(r"(?:^|/)(?:tests?|specs?)(?:/|$)|(?:^|[_-])test(?:[_-]|\.)|(?:^|[_-])spec(?:[_-]|\.)", re.I)


def _stable_hash(value: str, size: int = 16) -> str:
    return hashlib.blake2s(value.encode("utf-8", "ignore"), digest_size=size).hexdigest()


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _qualified_name(node.func)
    return ""


class _PythonSurface(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.definitions: dict[str, str] = {}
        self.signatures: dict[str, str] = {}
        self.calls: set[str] = set()
        self.imports: set[str] = set()
        self.raises: set[str] = set()
        self.branch_nodes = 0
        self.broad_excepts = 0

    @staticmethod
    def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        args = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        return f"{node.name}({', '.join(args)})"

    def _record_definition(self, node: ast.AST, name: str, signature: str) -> None:
        qualified = ".".join([*self.stack, name])
        payload = ast.dump(node, annotate_fields=True, include_attributes=False)
        self.definitions[qualified] = _stable_hash(payload, 12)
        self.signatures[qualified] = signature

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        bases = ",".join(filter(None, (_qualified_name(base) for base in node.bases)))
        self._record_definition(node, node.name, f"class {node.name}({bases})")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._record_definition(node, node.name, self._signature(node))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> Any:
        name = _qualified_name(node.func)
        if name:
            self.calls.add(name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        self.imports.update(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = node.module or ""
        self.imports.update(f"{module}.{alias.name}".strip(".") for alias in node.names)

    def visit_Raise(self, node: ast.Raise) -> Any:
        if node.exc is not None:
            name = _qualified_name(node.exc)
            if name:
                self.raises.add(name)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> Any:
        self.branch_nodes += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        self.branch_nodes += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        self.branch_nodes += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> Any:
        self.branch_nodes += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> Any:
        self.branch_nodes += len(node.handlers)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> Any:
        self.branch_nodes += len(node.cases)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Any:
        if node.type is None:
            self.broad_excepts += 1
        elif isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
            self.broad_excepts += 1
        self.generic_visit(node)


@dataclass(frozen=True, slots=True)
class SemanticChangeFingerprint:
    path: str
    language: str
    parse_ok_before: bool
    parse_ok_after: bool
    added_definitions: tuple[str, ...]
    removed_definitions: tuple[str, ...]
    modified_definitions: tuple[str, ...]
    signature_changes: tuple[str, ...]
    added_imports: tuple[str, ...]
    removed_imports: tuple[str, ...]
    added_calls: tuple[str, ...]
    removed_calls: tuple[str, ...]
    added_raises: tuple[str, ...]
    removed_raises: tuple[str, ...]
    line_delta: int
    branch_delta: int
    broad_except_delta: int
    public_api_change: bool
    risk_score: float
    risk_reasons: tuple[str, ...]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        return (
            f"path={self.path} lang={self.language} risk={self.risk_score:.2f} "
            f"api_change={str(self.public_api_change).lower()} line_delta={self.line_delta:+d} "
            f"defs(+{len(self.added_definitions)}/-{len(self.removed_definitions)}/~{len(self.modified_definitions)}) "
            f"imports(+{len(self.added_imports)}/-{len(self.removed_imports)}) "
            f"calls(+{len(self.added_calls)}/-{len(self.removed_calls)}) "
            f"reasons={';'.join(self.risk_reasons) or 'none'} fp={self.fingerprint}"
        )


def _python_surface(source: str) -> tuple[_PythonSurface | None, bool]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return None, False
    visitor = _PythonSurface()
    visitor.visit(tree)
    return visitor, True


def semantic_change_fingerprint(path: str, before: str, after: str) -> SemanticChangeFingerprint:
    """Create a compact structural diff without persisting either source body."""

    normalized = path.replace("\\", "/").lstrip("./")
    language = "python" if normalized.lower().endswith((".py", ".pyi")) else "text"
    before_lines = before.count("\n") + (1 if before else 0)
    after_lines = after.count("\n") + (1 if after else 0)

    if language != "python":
        changed = before != after
        material = f"{normalized}|{_stable_hash(before)}|{_stable_hash(after)}"
        return SemanticChangeFingerprint(
            path=normalized,
            language=language,
            parse_ok_before=True,
            parse_ok_after=True,
            added_definitions=(),
            removed_definitions=(),
            modified_definitions=(),
            signature_changes=(),
            added_imports=(),
            removed_imports=(),
            added_calls=(),
            removed_calls=(),
            added_raises=(),
            removed_raises=(),
            line_delta=after_lines - before_lines,
            branch_delta=0,
            broad_except_delta=0,
            public_api_change=False,
            risk_score=0.2 if changed else 0.0,
            risk_reasons=("non-Python semantic diff uses text-level fallback",) if changed else (),
            fingerprint=_stable_hash(material, 12),
        )

    old, old_ok = _python_surface(before)
    new, new_ok = _python_surface(after)
    old = old or _PythonSurface()
    new = new or _PythonSurface()
    old_defs, new_defs = set(old.definitions), set(new.definitions)
    added_defs = sorted(new_defs - old_defs)
    removed_defs = sorted(old_defs - new_defs)
    modified_defs = sorted(
        key for key in (old_defs & new_defs)
        if old.definitions.get(key) != new.definitions.get(key)
    )
    signature_changes = sorted(
        key for key in (old_defs & new_defs)
        if old.signatures.get(key) != new.signatures.get(key)
    )
    public_changes = [
        name for name in (*added_defs, *removed_defs, *signature_changes)
        if not name.rsplit(".", 1)[-1].startswith("_")
    ]

    added_imports = sorted(new.imports - old.imports)
    removed_imports = sorted(old.imports - new.imports)
    added_calls = sorted(new.calls - old.calls)
    removed_calls = sorted(old.calls - new.calls)
    added_raises = sorted(new.raises - old.raises)
    removed_raises = sorted(old.raises - new.raises)
    branch_delta = new.branch_nodes - old.branch_nodes
    broad_delta = new.broad_excepts - old.broad_excepts

    risk = 0.15
    reasons: list[str] = []
    if not new_ok:
        risk += 5.0
        reasons.append("post-edit Python parse failed")
    if removed_defs:
        risk += min(2.5, 0.75 * len(removed_defs))
        reasons.append(f"definitions removed={len(removed_defs)}")
    if signature_changes:
        risk += min(2.5, 0.8 * len(signature_changes))
        reasons.append(f"signatures changed={len(signature_changes)}")
    if public_changes:
        risk += min(2.0, 0.4 * len(public_changes))
        reasons.append(f"public surface changes={len(public_changes)}")
    dependency_delta = len(added_imports) + len(removed_imports) + len(added_calls) + len(removed_calls)
    if dependency_delta:
        risk += min(1.8, 0.12 * dependency_delta)
        reasons.append(f"dependency edges changed={dependency_delta}")
    if added_raises or removed_raises:
        risk += min(1.0, 0.2 * (len(added_raises) + len(removed_raises)))
        reasons.append("exception behavior changed")
    if broad_delta > 0:
        risk += min(1.5, 0.6 * broad_delta)
        reasons.append(f"broad exception handlers added={broad_delta}")
    if branch_delta > 6:
        risk += min(1.0, branch_delta * 0.08)
        reasons.append(f"control-flow branches increased={branch_delta}")

    structural = {
        "path": normalized,
        "added": added_defs,
        "removed": removed_defs,
        "modified": modified_defs,
        "signatures": signature_changes,
        "imports+": added_imports,
        "imports-": removed_imports,
        "calls+": added_calls,
        "calls-": removed_calls,
        "raises+": added_raises,
        "raises-": removed_raises,
        "line_delta": after_lines - before_lines,
        "branch_delta": branch_delta,
        "broad_delta": broad_delta,
    }
    return SemanticChangeFingerprint(
        path=normalized,
        language=language,
        parse_ok_before=old_ok,
        parse_ok_after=new_ok,
        added_definitions=tuple(added_defs[:64]),
        removed_definitions=tuple(removed_defs[:64]),
        modified_definitions=tuple(modified_defs[:64]),
        signature_changes=tuple(signature_changes[:64]),
        added_imports=tuple(added_imports[:64]),
        removed_imports=tuple(removed_imports[:64]),
        added_calls=tuple(added_calls[:96]),
        removed_calls=tuple(removed_calls[:96]),
        added_raises=tuple(added_raises[:32]),
        removed_raises=tuple(removed_raises[:32]),
        line_delta=after_lines - before_lines,
        branch_delta=branch_delta,
        broad_except_delta=broad_delta,
        public_api_change=bool(public_changes),
        risk_score=min(10.0, risk),
        risk_reasons=tuple(reasons[:12]),
        fingerprint=_stable_hash(json.dumps(structural, sort_keys=True), 12),
    )


@dataclass(frozen=True, slots=True)
class MaintenanceFinding:
    kind: str
    path: str
    symbol: str
    pressure: float
    evidence: str
    suggested_inspection: str


class MaintenanceHealthAnalyzer:
    """Compute inspectable maintenance-pressure signals over current code."""

    def __init__(self, root: Path, index: CodeIntelligenceIndex | None = None) -> None:
        self.root = root.resolve()
        self.index = index or CodeIntelligenceIndex(self.root)

    def analyze(self, paths: Sequence[str] | None = None, *, max_findings: int = 40) -> dict[str, Any]:
        self.index.build(list(paths) if paths else None)
        findings: list[MaintenanceFinding] = []
        tested_paths: set[str] = set()
        reverse_counts: dict[str, int] = {}
        for edge in self.index.edges:
            if edge.kind == "tests":
                source = self.index.symbols.get(edge.source)
                target = self.index.symbols.get(edge.target)
                if source and target:
                    tested_paths.add(target.path)
            reverse_counts[edge.target] = reverse_counts.get(edge.target, 0) + 1

        for record in self.index.files.values():
            if record.parse_error:
                findings.append(MaintenanceFinding(
                    "parse-error", record.path, "<module>", 10.0,
                    record.parse_error[:240], "repair parseability before structural refactors",
                ))
            path = self.root / record.path
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                source = ""
            todo_count = len(_TODO_RE.findall(source))
            if todo_count:
                findings.append(MaintenanceFinding(
                    "deferred-work", record.path, "<module>", min(5.0, 0.8 + math.log2(todo_count + 1)),
                    f"TODO/FIXME/HACK markers={todo_count}", "inspect whether deferred work intersects active change surfaces",
                ))
            if record.language == "python" and source:
                surface, ok = _python_surface(source)
                if ok and surface and surface.broad_excepts:
                    findings.append(MaintenanceFinding(
                        "broad-exception-surface", record.path, "<module>", min(7.0, 1.5 + surface.broad_excepts),
                        f"broad handlers={surface.broad_excepts}", "verify exception boundaries and failure observability",
                    ))
            for symbol in record.symbols:
                span = max(1, symbol.line_end - symbol.line_start + 1)
                fanout = len(symbol.calls)
                fanin = reverse_counts.get(symbol.symbol_id, 0)
                if span >= 80:
                    findings.append(MaintenanceFinding(
                        "large-symbol", record.path, symbol.qualified_name,
                        min(8.0, 1.0 + span / 70.0), f"line span={span}",
                        "inspect cohesion and isolate the smallest safe edit surface",
                    ))
                if fanout >= 14:
                    findings.append(MaintenanceFinding(
                        "dependency-fanout", record.path, symbol.qualified_name,
                        min(8.0, 1.0 + fanout / 8.0), f"distinct calls={fanout}",
                        "treat as a dependency hub; validate downstream behavior",
                    ))
                if fanin >= 10:
                    findings.append(MaintenanceFinding(
                        "change-centrality", record.path, symbol.qualified_name,
                        min(9.0, 1.0 + fanin / 6.0), f"incoming graph edges={fanin}",
                        "changes may propagate widely; inspect callers/subclasses/tests first",
                    ))
            if (
                record.symbols
                and record.path not in tested_paths
                and not _TEST_RE.search(record.path)
                and len(record.symbols) >= 3
            ):
                findings.append(MaintenanceFinding(
                    "no-linked-test-surface", record.path, "<module>", 1.2,
                    "static graph found no direct test edge", "search for integration/behavioral tests before changing this file",
                ))

        findings.sort(key=lambda item: (-item.pressure, item.path, item.symbol, item.kind))
        selected = findings[: max(1, min(200, int(max_findings)))]
        return {
            "basis": "heuristic maintenance-pressure signals; not a quality score",
            "files": len(self.index.files),
            "symbols": len(self.index.symbols),
            "graph_edges": len(self.index.edges),
            "findings": [asdict(item) for item in selected],
            "pressure_total": round(sum(item.pressure for item in selected), 3),
        }

    @staticmethod
    def render(report: dict[str, Any], max_chars: int = 14000) -> str:
        lines = [
            "[MAINTENANCE PRESSURE REPORT: heuristic inspection priorities, not a quality grade]",
            f"files={report.get('files', 0)} symbols={report.get('symbols', 0)} edges={report.get('graph_edges', 0)}",
            f"pressure_total={report.get('pressure_total', 0)}",
        ]
        for item in report.get("findings", []):
            lines.append(
                f"- pressure={float(item.get('pressure', 0)):.2f} kind={item.get('kind')} "
                f"{item.get('path')}:{item.get('symbol')} evidence={item.get('evidence')} "
                f"inspect={item.get('suggested_inspection')}"
            )
        return "\n".join(lines)[:max_chars]


def validator_candidates(root: Path, impact: ImpactReport) -> list[str]:
    """Return likely validation commands based only on files actually present."""

    candidates: list[str] = []
    linked_tests = impact.tests[:6]
    if (root / "pyproject.toml").is_file() or (root / "pytest.ini").is_file() or (root / "tests").is_dir():
        candidates.append("python -m pytest " + " ".join(linked_tests) if linked_tests else "python -m pytest")
    package = root / "package.json"
    if package.is_file():
        try:
            payload = json.loads(package.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError, TypeError):
            payload = {}
        scripts = payload.get("scripts") if isinstance(payload, dict) else {}
        if isinstance(scripts, dict):
            for name in ("test", "typecheck", "lint"):
                if name in scripts:
                    candidates.append(f"npm run {name}")
    if (root / "Cargo.toml").is_file():
        candidates.append("cargo test")
    if (root / "go.mod").is_file():
        candidates.append("go test ./...")
    if (root / "pubspec.yaml").is_file():
        candidates.append("dart test")
    return list(dict.fromkeys(candidates))[:8]


class MaintenancePlanner:
    """Fuse current impact evidence with historical procedural memory."""

    def __init__(self, root: Path, index: CodeIntelligenceIndex | None = None) -> None:
        self.root = root.resolve()
        self.index = index or CodeIntelligenceIndex(self.root)

    async def plan(self, query: str, memory: Any | None = None, *, max_chars: int = 16000) -> str:
        self.index.build()
        impact = self.index.impact(query, max_depth=4, max_nodes=48)
        playbooks: list[dict[str, Any]] = []
        if memory is not None and hasattr(memory, "coding_playbooks"):
            try:
                playbooks = list(await memory.coding_playbooks(query, limit=5))
            except Exception:
                playbooks = []
        validators = validator_candidates(self.root, impact)
        lines = [
            "[MAINTENANCE EXECUTION CONTRACT: current code beats historical procedure]",
            f"query={query}",
            f"blast_radius_files={','.join(impact.files) or 'none'}",
            f"linked_tests={','.join(impact.tests) or 'none'}",
            f"static_risk={impact.risk_score:.2f}",
            f"risk_reasons={'; '.join(impact.risk_reasons) or 'none'}",
            f"uncertainty={'; '.join(impact.uncertainty) or 'none'}",
            "",
            "CURRENT CHANGE FRONTIER",
        ]
        for node in [*impact.anchors, *impact.frontier][:36]:
            lines.append(
                f"- d={node.distance} {node.path}:{node.symbol} kind={node.kind} "
                f"score={node.score:.3f} reasons={'; '.join(node.reasons)}"
            )
        lines.extend(["", "HISTORICAL PROCEDURAL HINTS (non-authoritative)"])
        if playbooks:
            for item in playbooks:
                lines.append(
                    f"- score={float(item.get('score', 0)):.3f} confidence={float(item.get('confidence', 0)):.3f} "
                    f"support={item.get('support_count', 0)} success_rate={float(item.get('success_rate', 0)):.3f} "
                    f"action={item.get('action')} lang={item.get('language')} "
                    f"steps={' > '.join(item.get('step_kinds') or []) or 'none'} "
                    f"validators={'; '.join(item.get('validators') or []) or 'none'}"
                )
        else:
            lines.append("- none")
        lines.extend(["", "VALIDATION LADDER (candidates discovered from repository files)"])
        if validators:
            lines.extend(f"- {value}" for value in validators)
        else:
            lines.append("- no validator inferred; inspect project configuration before claiming validation")
        lines.extend([
            "",
            "EXECUTION RULES",
            "- Localize from current AST/graph evidence before editing.",
            "- Read exact targets and preserve digest preconditions.",
            "- Use historical playbooks only as strategy hints; never replay old patches blindly.",
            "- After every edit, inspect the semantic change fingerprint and expand validation when public/dependency surfaces changed.",
            "- A validator failure creates a new hypothesis; do not increase patch size merely to silence the failure.",
        ])
        return "\n".join(lines)[: max(1000, int(max_chars))]
