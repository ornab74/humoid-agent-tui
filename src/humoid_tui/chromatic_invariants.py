from __future__ import annotations

"""Chromatic invariant compilation and lightweight change proofs.

The invariant compiler converts a coding request and the current repository into
bounded, inspectable proof obligations.  It does not attempt formal verification
and it never treats an invariant address as truth.  Static evidence, semantic
change fingerprints, and validator outcomes remain the authority.
"""

import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .chromatic_system import ChromaticInvariantAddress, encode_invariant_address
from .code_intelligence import CodeIntelligenceIndex, ImpactReport

_PRESERVE_RE = re.compile(r"\b(?:preserve|keep|without breaking|do not change|don't change|must remain|compatible|compatibility)\b", re.I)
_API_CHANGE_RE = re.compile(r"\b(?:breaking change|allow breaking|break api|change (?:the )?api|public api|api migration|migration|migrate|change signature|new signature)\b", re.I)
_VALIDATION_RE = re.compile(r"\b(?:test|tests|pytest|validate|validation|verify|verification|lint|typecheck|benchmark)\b", re.I)
_ERROR_CHANGE_RE = re.compile(r"\b(?:exception|exceptions|raise|raises|error contract|failure mode|failure modes)\b", re.I)
_DEP_CHANGE_RE = re.compile(r"\b(?:dependency|dependencies|import|imports|move module|module migration|package migration)\b", re.I)


def _norm_path(value: str) -> str:
    return value.strip().replace("\\", "/").lstrip("./")


def _digest(value: str, size: int = 12) -> str:
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


@dataclass(frozen=True, slots=True)
class _PythonSnapshot:
    parse_ok: bool
    definitions: tuple[str, ...]
    public_signatures: tuple[tuple[str, str], ...]
    imports: tuple[str, ...]
    raises: tuple[str, ...]
    digest: str


class _SnapshotVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.definitions: list[str] = []
        self.public_signatures: dict[str, str] = {}
        self.imports: set[str] = set()
        self.raises: set[str] = set()

    @staticmethod
    def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        # ast.unparse(arguments) preserves defaults, keyword-only markers and
        # annotations, which makes the invariant materially stronger than a
        # name-only parameter list while remaining independent of whitespace.
        try:
            arguments = ast.unparse(node.args)
        except Exception:
            arguments = ", ".join(
                arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            )
        returns = ""
        if node.returns is not None:
            try:
                returns = f" -> {ast.unparse(node.returns)}"
            except Exception:
                returns = " -> <annotation>"
        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        return f"{prefix}{node.name}({arguments}){returns}"

    def _record(self, name: str, signature: str) -> None:
        qualified = ".".join([*self.stack, name])
        self.definitions.append(qualified)
        if not name.startswith("_"):
            self.public_signatures[qualified] = signature

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        bases = ",".join(filter(None, (_qualified_name(base) for base in node.bases)))
        self._record(node.name, f"class {node.name}({bases})")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._record(node.name, self._signature(node))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.visit_FunctionDef(node)

    def visit_Import(self, node: ast.Import) -> Any:
        self.imports.update(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = node.module or ""
        self.imports.update(f"{module}.{alias.name}".strip(".") for alias in node.names)

    def visit_Raise(self, node: ast.Raise) -> Any:
        if node.exc is not None:
            value = _qualified_name(node.exc)
            if value:
                self.raises.add(value)
        self.generic_visit(node)


def _snapshot_python(source: str) -> _PythonSnapshot:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return _PythonSnapshot(False, (), (), (), (), _digest(source, 16))
    visitor = _SnapshotVisitor()
    visitor.visit(tree)
    return _PythonSnapshot(
        True,
        tuple(sorted(dict.fromkeys(visitor.definitions))),
        tuple(sorted(visitor.public_signatures.items())),
        tuple(sorted(visitor.imports)),
        tuple(sorted(visitor.raises)),
        _digest(source, 16),
    )


@dataclass(frozen=True, slots=True)
class ChromaticInvariant:
    invariant_id: str
    kind: str
    statement: str
    severity: str
    source: str
    path: str = ""
    symbol: str = ""
    expected: str = ""
    requires_validation: bool = False
    address_hex: str = ""

    @property
    def address(self) -> ChromaticInvariantAddress:
        return ChromaticInvariantAddress(bytes.fromhex(self.address_hex))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InvariantContract:
    contract_id: str
    request: str
    operation: str
    files: tuple[str, ...]
    symbols: tuple[str, ...]
    impact_files: tuple[str, ...]
    linked_tests: tuple[str, ...]
    baseline: tuple[tuple[str, dict[str, Any]], ...]
    invariants: tuple[ChromaticInvariant, ...]
    surface_key: str

    def baseline_map(self) -> dict[str, dict[str, Any]]:
        return {path: dict(value) for path, value in self.baseline}

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "request": self.request,
            "operation": self.operation,
            "files": list(self.files),
            "symbols": list(self.symbols),
            "impact_files": list(self.impact_files),
            "linked_tests": list(self.linked_tests),
            "baseline": {path: value for path, value in self.baseline},
            "invariants": [item.to_dict() for item in self.invariants],
            "surface_key": self.surface_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InvariantContract":
        invariants = tuple(ChromaticInvariant(**dict(item)) for item in value.get("invariants", []) if isinstance(item, dict))
        baseline_obj = value.get("baseline") if isinstance(value.get("baseline"), dict) else {}
        return cls(
            contract_id=str(value.get("contract_id") or ""),
            request=str(value.get("request") or ""),
            operation=str(value.get("operation") or "edit"),
            files=tuple(str(item) for item in value.get("files", []) if item),
            symbols=tuple(str(item) for item in value.get("symbols", []) if item),
            impact_files=tuple(str(item) for item in value.get("impact_files", []) if item),
            linked_tests=tuple(str(item) for item in value.get("linked_tests", []) if item),
            baseline=tuple(sorted((str(path), dict(payload)) for path, payload in baseline_obj.items() if isinstance(payload, dict))),
            invariants=invariants,
            surface_key=str(value.get("surface_key") or ""),
        )


@dataclass(frozen=True, slots=True)
class InvariantCheck:
    invariant_id: str
    kind: str
    status: str
    severity: str
    path: str
    symbol: str
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InvariantProofReport:
    contract_id: str
    status: str
    gate: bool
    proof_score: float
    passed: int
    warned: int
    failed: int
    unverified: int
    checks: tuple[InvariantCheck, ...]
    obligations: tuple[str, ...]
    proof_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "status": self.status,
            "gate": self.gate,
            "proof_score": self.proof_score,
            "passed": self.passed,
            "warned": self.warned,
            "failed": self.failed,
            "unverified": self.unverified,
            "checks": [item.to_dict() for item in self.checks],
            "obligations": list(self.obligations),
            "proof_key": self.proof_key,
        }


class ChromaticInvariantCompiler:
    """Compile and evaluate bounded static/validation invariants for an edit."""

    def __init__(self, root: Path, index: CodeIntelligenceIndex | None = None) -> None:
        self.root = root.resolve()
        self.index = index or CodeIntelligenceIndex(self.root)

    def _make_invariant(
        self,
        *,
        kind: str,
        statement: str,
        severity: str,
        source: str,
        path: str = "",
        symbol: str = "",
        expected: str = "",
        requires_validation: bool = False,
        operation: str = "edit",
    ) -> ChromaticInvariant:
        address = encode_invariant_address(
            kind=kind,
            severity=severity,
            source=source,
            path=path,
            symbol=symbol,
            statement=statement,
            operation=operation,
            requires_validation=requires_validation,
        )
        stable = json.dumps(
            [kind, statement, severity, source, path, symbol, expected, requires_validation],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return ChromaticInvariant(
            invariant_id=_digest(stable, 10),
            kind=kind,
            statement=statement,
            severity=severity,
            source=source,
            path=path,
            symbol=symbol,
            expected=expected,
            requires_validation=requires_validation,
            address_hex=address.address.hex(),
        )

    def compile(
        self,
        request: str,
        *,
        operation: str = "edit",
        files: Sequence[str] = (),
        symbols: Sequence[str] = (),
        constraints: Sequence[str] = (),
        acceptance: Sequence[str] = (),
    ) -> InvariantContract:
        normalized_files = tuple(dict.fromkeys(_norm_path(value) for value in files if _norm_path(value)))[:24]
        normalized_symbols = tuple(dict.fromkeys(str(value) for value in symbols if str(value).strip()))[:32]

        self.index.build()
        try:
            impact: ImpactReport = self.index.impact(request, max_depth=4, max_nodes=64)
        except Exception:
            impact = ImpactReport(request, [], [], [], list(normalized_files), [], 0.0, [], ["impact analysis unavailable"])

        relevant_files = tuple(dict.fromkeys([*normalized_files, *impact.files]))[:32]
        baseline: list[tuple[str, dict[str, Any]]] = []
        invariants: list[ChromaticInvariant] = []
        request_lower = request.lower()
        destructive_symbol_request = bool(
            re.search(r"\b(?:rename|remove|delete)\b", request, re.I)
            and any(symbol.lower() in request_lower for symbol in normalized_symbols)
        )
        allow_api_change = bool(_API_CHANGE_RE.search(request)) or destructive_symbol_request
        allow_error_change = bool(_ERROR_CHANGE_RE.search(request))
        allow_dependency_change = bool(_DEP_CHANGE_RE.search(request))

        for relative in relevant_files:
            path = self.root / relative
            if not path.is_file() or path.suffix.lower() not in {".py", ".pyi"}:
                continue
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(source) > 2_000_000:
                continue
            snapshot = _snapshot_python(source)
            payload = {
                "parse_ok": snapshot.parse_ok,
                "definitions": list(snapshot.definitions),
                "public_signatures": dict(snapshot.public_signatures),
                "imports": list(snapshot.imports),
                "raises": list(snapshot.raises),
                "digest": snapshot.digest,
            }
            baseline.append((relative, payload))
            invariants.append(self._make_invariant(
                kind="parseability",
                statement=f"{relative} remains parseable Python after the edit",
                severity="critical",
                source="baseline",
                path=relative,
                expected="parse_ok=true",
                operation=operation,
            ))
            if not allow_api_change:
                for symbol, signature in snapshot.public_signatures[:64]:
                    if normalized_symbols and not any(
                        requested.lower() in symbol.lower() or symbol.lower().endswith(requested.lower())
                        for requested in normalized_symbols
                    ):
                        continue
                    invariants.append(self._make_invariant(
                        kind="public-signature",
                        statement=f"preserve public signature {relative}:{symbol}",
                        severity="critical" if _PRESERVE_RE.search(request) else "high",
                        source="baseline",
                        path=relative,
                        symbol=symbol,
                        expected=signature,
                        operation=operation,
                    ))
            for requested in normalized_symbols:
                matching = [name for name in snapshot.definitions if name.lower().endswith(requested.lower())]
                if matching and not re.search(r"\b(?:remove|delete|rename)\b", request_lower):
                    invariants.append(self._make_invariant(
                        kind="symbol-presence",
                        statement=f"preserve requested symbol {relative}:{matching[0]}",
                        severity="critical",
                        source="request+baseline",
                        path=relative,
                        symbol=matching[0],
                        expected="present",
                        operation=operation,
                    ))
            if snapshot.imports and not allow_dependency_change and operation in {"debug", "refactor", "edit"}:
                invariants.append(self._make_invariant(
                    kind="dependency-continuity",
                    statement=f"review dependency drift in {relative}; baseline imports should not disappear accidentally",
                    severity="medium",
                    source="baseline",
                    path=relative,
                    expected=json.dumps(snapshot.imports),
                    operation=operation,
                ))
            if snapshot.raises and not allow_error_change:
                invariants.append(self._make_invariant(
                    kind="exception-continuity",
                    statement=f"review exception-contract drift in {relative}",
                    severity="medium",
                    source="baseline",
                    path=relative,
                    expected=json.dumps(snapshot.raises),
                    operation=operation,
                ))

        for constraint in constraints[:16]:
            invariants.append(self._make_invariant(
                kind="explicit-constraint",
                statement=constraint,
                severity="critical" if _PRESERVE_RE.search(constraint) else "high",
                source="user-constraint",
                requires_validation=True,
                operation=operation,
            ))
        acceptance_text = "\n".join(acceptance)
        if acceptance or _VALIDATION_RE.search(request) or (operation == "debug" and impact.tests):
            invariants.append(self._make_invariant(
                kind="validation-obligation",
                statement="execute a repository-supported validator covering the changed behavior",
                severity="critical" if acceptance or _VALIDATION_RE.search(acceptance_text) else "high",
                source="acceptance/impact",
                expected="validator observed",
                requires_validation=True,
                operation=operation,
            ))
        if impact.files:
            invariants.append(self._make_invariant(
                kind="blast-radius",
                statement="changed files should remain inside the grounded impact frontier unless new evidence expands it",
                severity="medium",
                source="current-impact-graph",
                expected=json.dumps(sorted(impact.files)),
                operation=operation,
            ))

        # De-duplicate structurally identical obligations.
        unique: dict[str, ChromaticInvariant] = {}
        for item in invariants:
            unique.setdefault(item.invariant_id, item)
        selected = tuple(unique.values())[:160]
        surface_key = _digest("|".join(sorted(item.address_hex for item in selected)), 12)
        contract_id = _digest(json.dumps([
            request,
            operation,
            list(relevant_files),
            list(normalized_symbols),
            surface_key,
        ], sort_keys=True), 12)
        return InvariantContract(
            contract_id=contract_id,
            request=request,
            operation=operation,
            files=normalized_files,
            symbols=normalized_symbols,
            impact_files=tuple(impact.files[:64]),
            linked_tests=tuple(impact.tests[:32]),
            baseline=tuple(baseline),
            invariants=selected,
            surface_key=surface_key,
        )

    def evaluate(
        self,
        contract: InvariantContract | dict[str, Any],
        *,
        change_fingerprints: dict[str, dict[str, Any]] | None = None,
        changed_paths: Sequence[str] = (),
        validators: Sequence[str] = (),
        validated: bool = False,
    ) -> InvariantProofReport:
        if isinstance(contract, dict):
            contract = InvariantContract.from_dict(contract)
        fingerprints = change_fingerprints or {}
        baseline = contract.baseline_map()
        changed = {_norm_path(value) for value in changed_paths if value}
        current: dict[str, _PythonSnapshot] = {}
        for path in baseline:
            target = self.root / path
            try:
                source = target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                source = ""
            current[path] = _snapshot_python(source) if target.exists() else _PythonSnapshot(False, (), (), (), (), "missing")

        checks: list[InvariantCheck] = []
        obligations: list[str] = []
        severity_weight = {"critical": 4.0, "high": 2.5, "medium": 1.5, "low": 1.0}
        earned = 0.0
        possible = 0.0

        for item in contract.invariants:
            status = "unverified"
            evidence = "no direct evidence"
            if item.kind == "parseability":
                snap = current.get(item.path)
                status = "pass" if snap and snap.parse_ok else "fail"
                evidence = f"current_parse_ok={bool(snap and snap.parse_ok)}"
            elif item.kind == "public-signature":
                snap = current.get(item.path)
                signatures = dict(snap.public_signatures) if snap else {}
                observed = signatures.get(item.symbol)
                status = "pass" if observed == item.expected else "fail"
                evidence = f"expected={item.expected!r} observed={observed!r}"
            elif item.kind == "symbol-presence":
                snap = current.get(item.path)
                present = bool(snap and item.symbol in snap.definitions)
                status = "pass" if present else "fail"
                evidence = f"present={present}"
            elif item.kind == "dependency-continuity":
                payload = fingerprints.get(item.path) or {}
                removed = tuple(payload.get("removed_imports") or ())
                added = tuple(payload.get("added_imports") or ())
                status = "warn" if removed or added else "pass"
                evidence = f"imports_added={len(added)} removed={len(removed)}"
            elif item.kind == "exception-continuity":
                payload = fingerprints.get(item.path) or {}
                removed = tuple(payload.get("removed_raises") or ())
                added = tuple(payload.get("added_raises") or ())
                status = "warn" if removed or added else "pass"
                evidence = f"raises_added={len(added)} removed={len(removed)}"
            elif item.kind == "blast-radius":
                allowed = set(contract.impact_files) | set(contract.files)
                escaped = sorted(path for path in changed if path not in allowed)
                status = "warn" if escaped else "pass"
                evidence = f"outside_frontier={','.join(escaped) or 'none'}"
            elif item.kind == "validation-obligation":
                status = "pass" if validated and validators else "unverified"
                evidence = f"validated={validated} validators={len(validators)}"
            elif item.kind == "explicit-constraint":
                if validated and validators:
                    status = "pass"
                    evidence = "validator evidence observed; semantic constraint still requires human/model interpretation"
                else:
                    status = "unverified"
                    evidence = "explicit constraint requires validation evidence"
            else:
                status = "unverified"

            weight = severity_weight.get(item.severity, 1.0)
            possible += weight
            if status == "pass":
                earned += weight
            elif status == "warn":
                earned += weight * 0.55
            elif status == "unverified":
                earned += weight * 0.20
            if status in {"warn", "unverified", "fail"}:
                obligations.append(f"{item.severity}:{item.kind}:{item.statement} [{status}]")
            checks.append(InvariantCheck(
                invariant_id=item.invariant_id,
                kind=item.kind,
                status=status,
                severity=item.severity,
                path=item.path,
                symbol=item.symbol,
                evidence=evidence,
            ))

        passed = sum(item.status == "pass" for item in checks)
        warned = sum(item.status == "warn" for item in checks)
        failed = sum(item.status == "fail" for item in checks)
        unverified = sum(item.status == "unverified" for item in checks)
        hard_fail = any(item.status == "fail" and item.severity in {"critical", "high"} for item in checks)
        critical_unverified = any(
            item.status == "unverified" and item.severity == "critical" and item.kind in {"validation-obligation", "explicit-constraint"}
            for item in checks
        )
        gate = not hard_fail and not critical_unverified
        status = "fail" if hard_fail else "unverified" if critical_unverified else "warn" if warned or unverified else "pass"
        proof_score = round(earned / possible if possible else 1.0, 4)
        proof_material = json.dumps(
            [[item.invariant_id, item.status, item.evidence] for item in checks],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return InvariantProofReport(
            contract_id=contract.contract_id,
            status=status,
            gate=gate,
            proof_score=proof_score,
            passed=passed,
            warned=warned,
            failed=failed,
            unverified=unverified,
            checks=tuple(checks),
            obligations=tuple(obligations[:64]),
            proof_key=_digest(proof_material, 12),
        )

    @staticmethod
    def render_contract(contract: InvariantContract, max_chars: int = 12000) -> str:
        lines = [
            "[CHROMATIC INVARIANT CONTRACT: static obligations, not formal proof]",
            f"contract_id={contract.contract_id} surface={contract.surface_key}",
            f"impact_files={','.join(contract.impact_files) or 'none'}",
            f"linked_tests={','.join(contract.linked_tests) or 'none'}",
        ]
        for item in contract.invariants:
            lines.append(
                f"- {item.severity}:{item.kind} id={item.invariant_id} chromatic={item.address_hex[:16]} "
                f"target={item.path + (':' + item.symbol if item.symbol else '') or 'task'} statement={item.statement}"
            )
        return "\n".join(lines)[:max_chars]

    @staticmethod
    def render_proof(report: InvariantProofReport, max_chars: int = 12000) -> str:
        lines = [
            "[CHROMATIC INVARIANT PROOF: lightweight static + validation evidence]",
            f"contract_id={report.contract_id} proof_key={report.proof_key}",
            f"status={report.status} gate={str(report.gate).lower()} score={report.proof_score:.4f}",
            f"checks pass={report.passed} warn={report.warned} fail={report.failed} unverified={report.unverified}",
        ]
        for item in report.checks:
            lines.append(
                f"- {item.status}:{item.severity}:{item.kind} {item.path + (':' + item.symbol if item.symbol else '') or 'task'} evidence={item.evidence}"
            )
        if report.obligations:
            lines.append("OPEN OBLIGATIONS")
            lines.extend(f"- {value}" for value in report.obligations)
        return "\n".join(lines)[:max_chars]
