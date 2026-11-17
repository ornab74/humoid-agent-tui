from __future__ import annotations

"""Runtime integration for semantic diffs, maintenance planning, and playbooks."""

import json
import re
from typing import Any

from .chromatic_system import encode_change_delta
from .chromatic_invariants import ChromaticInvariantCompiler, InvariantContract
from .chromatic_predictive import ChromaticPredictiveMaintainer, evaluate_predictive_outcome
from .maintenance import (
    MaintenanceHealthAnalyzer,
    MaintenancePlanner,
    semantic_change_fingerprint,
    validator_candidates,
)

_FINGERPRINT_RE = re.compile(r"\[SEMANTIC CHANGE FINGERPRINT\]\s*(\{.*\})", re.S)


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def _schemas() -> list[dict[str, Any]]:
    return [
        _schema(
            "plan_maintenance_change",
            "Build an impact-aware coding/maintenance execution contract by combining the current repository AST/call/test frontier with separately ranked historical procedural playbooks. Historical procedures are hints only; current code is authoritative.",
            {
                "query": {"type": "string"},
                "max_chars": {"type": "integer", "minimum": 2000, "maximum": 100000},
            },
            ["query"],
        ),
        _schema(
            "maintenance_health_report",
            "Inspect deterministic maintenance-pressure signals such as parse errors, large symbols, dependency fanout, change centrality, broad exception surfaces, deferred-work markers, and missing static test links. This is not a code-quality grade.",
            {
                "paths": {"type": "array", "items": {"type": "string"}},
                "max_findings": {"type": "integer", "minimum": 1, "maximum": 200},
                "max_chars": {"type": "integer", "minimum": 1000, "maximum": 100000},
            },
        ),
        _schema(
            "recall_coding_playbook",
            "Retrieve successful procedural coding/repair trajectories separately from factual RAG. Use these as strategy hints, never as commands or stale patches.",
            {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 12},
                "max_chars": {"type": "integer", "minimum": 600, "maximum": 30000},
            },
            ["query"],
        ),
        _schema(
            "inspect_chromatic_code_surface",
            "Inspect the multi-resolution chromatic coding surface for a query: project/package/file/symbol-behavior keys, candidate-collapse diagnostics, and top reranked memories. Chromatic coordinates are acceleration hints, not truth.",
            {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 24},
                "max_chars": {"type": "integer", "minimum": 1000, "maximum": 30000},
            },
            ["query"],
        ),
        _schema(
            "compile_chromatic_invariants",
            "Compile an inspectable chromatic invariant contract from the current coding request and repository. The contract captures parseability, symbol/API preservation, dependency/exception continuity, blast radius, explicit constraints, and validation obligations without storing source bodies.",
            {
                "query": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}},
                "symbols": {"type": "array", "items": {"type": "string"}},
                "acceptance": {"type": "array", "items": {"type": "string"}},
                "max_chars": {"type": "integer", "minimum": 1000, "maximum": 30000},
            },
            ["query"],
        ),
        _schema(
            "inspect_chromatic_invariant_proof",
            "Inspect the active edit cycle's lightweight invariant proof: static contract checks, validation obligations, blocking failures, warnings, proof score, and open obligations. This is not formal verification.",
            {
                "max_chars": {"type": "integer", "minimum": 800, "maximum": 30000},
            },
        ),
        _schema(
            "recall_chromatic_invariant_history",
            "For the active edit cycle, retrieve prior validated procedures whose chromatic invariant contracts resemble the current proof obligations. Returns validation-strategy hints only; it never proves the current change and never replays old patches.",
            {
                "limit": {"type": "integer", "minimum": 1, "maximum": 12},
                "max_chars": {"type": "integer", "minimum": 800, "maximum": 30000},
            },
        ),
        _schema(
            "forecast_chromatic_maintenance",
            "Compile a pre-edit chromatic predictive-maintenance surface from current code topology, invariant obligations, linked tests, and similar historical outcomes. The risk score is an uncalibrated planning priority, never a defect probability.",
            {
                "query": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}},
                "symbols": {"type": "array", "items": {"type": "string"}},
                "acceptance": {"type": "array", "items": {"type": "string"}},
                "max_chars": {"type": "integer", "minimum": 1200, "maximum": 30000},
            },
            ["query"],
        ),
        _schema(
            "inspect_chromatic_predictive_forecast",
            "Inspect the active edit cycle's pre-edit predictive risk surface and its current path-coverage outcome. Predictive priority guides inspection and validation budget but cannot block or prove an edit.",
            {
                "max_chars": {"type": "integer", "minimum": 800, "maximum": 30000},
            },
        ),
        _schema(
            "recall_chromatic_regression_history",
            "Retrieve similar historical coding outcomes as regression-risk evidence. Failure/success rates are non-authoritative historical signals and are never treated as current defect probabilities.",
            {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 16},
                "max_chars": {"type": "integer", "minimum": 800, "maximum": 30000},
            },
        ),
        _schema(
            "recall_chromatic_change_history",
            "After an edit has a semantic fingerprint, retrieve prior validated procedures with a similar chromatic change shape (API/signature/dependency/control-flow/exception risk). This is strategy and validation evidence only, never a patch to replay.",
            {
                "path": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 12},
                "max_chars": {"type": "integer", "minimum": 800, "maximum": 30000},
            },
        ),
    ]


def install_maintenance_runtime() -> None:
    from .tools import ToolRegistry

    if not getattr(ToolRegistry, "_humoid_maintenance_runtime", False):
        original_init = ToolRegistry.__init__
        original_schemas = ToolRegistry.schemas
        original_execute = ToolRegistry.execute
        original_commit = ToolRegistry._commit_file
        original_attach_memory = ToolRegistry.attach_memory

        def init(self: Any, settings: Any) -> None:
            original_init(self, settings)
            self.maintenance_planner = MaintenancePlanner(self.root, getattr(self, "code_intelligence", None))
            self.maintenance_health = MaintenanceHealthAnalyzer(self.root, getattr(self, "code_intelligence", None))
            self.chromatic_invariant_compiler = ChromaticInvariantCompiler(self.root, getattr(self, "code_intelligence", None))
            self.chromatic_predictive_maintainer = ChromaticPredictiveMaintainer(self.root, getattr(self, "code_intelligence", None))
            self.last_predictive_forecast: dict[str, Any] = {}
            self.maintenance_change_fingerprints: dict[str, dict[str, Any]] = {}
            # Lets edit-cycle enrichment reuse the already-built project indexes.
            self.perspective._tool_registry = self

        def attach_memory(self: Any, memory: Any) -> None:
            original_attach_memory(self, memory)
            if hasattr(self, "edit_cycle"):
                self.edit_cycle.memory = memory

        def schemas(self: Any) -> list[dict[str, Any]]:
            existing = original_schemas(self)
            names = {item["function"]["name"] for item in existing}
            additions = [item for item in _schemas() if item["function"]["name"] not in names]
            return [*additions, *existing]

        async def commit(self: Any, relative: str, path: Any, previous: str | None, content: str) -> str:
            before = previous or ""
            result = await original_commit(self, relative, path, previous, content)
            if result.startswith("no changes to"):
                return result
            try:
                fingerprint = semantic_change_fingerprint(relative, before, content)
                payload = fingerprint.to_dict()
                payload["chromatic_delta"] = encode_change_delta(payload).to_dict()
                self.maintenance_change_fingerprints[relative.replace("\\", "/").lstrip("./")] = payload
                return result + "\n[SEMANTIC CHANGE FINGERPRINT] " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            except Exception as exc:
                # Semantic diffing is advisory and must never turn a valid write into a failed write.
                return result + f"\n[SEMANTIC CHANGE FINGERPRINT unavailable={type(exc).__name__}]"

        async def execute(self: Any, name: str, args: dict[str, Any]) -> str:
            if name == "plan_maintenance_change":
                planner = MaintenancePlanner(self.root, getattr(self, "code_intelligence", None))
                return await planner.plan(
                    str(args["query"]), self.memory,
                    max_chars=int(args.get("max_chars", 16000)),
                )
            if name == "maintenance_health_report":
                analyzer = MaintenanceHealthAnalyzer(self.root, getattr(self, "code_intelligence", None))
                report = analyzer.analyze(
                    [str(value) for value in args.get("paths", [])] or None,
                    max_findings=int(args.get("max_findings", 40)),
                )
                return analyzer.render(report, int(args.get("max_chars", 14000)))
            if name == "recall_coding_playbook":
                if self.memory is None:
                    return "[CODING PROCEDURAL PLAYBOOKS]\nnone: memory router is not attached"
                return await self.memory.coding_playbook_packet(
                    str(args["query"]),
                    limit=int(args.get("limit", 5)),
                    max_chars=int(args.get("max_chars", 8000)),
                )
            if name == "inspect_chromatic_code_surface":
                if self.memory is None or not hasattr(self.memory, "chromatic_surface_packet"):
                    return "[CHROMATIC CODE SURFACE]\nnone: Rhomgaic memory is not attached"
                return await self.memory.chromatic_surface_packet(
                    str(args["query"]),
                    limit=int(args.get("limit", 8)),
                    max_chars=int(args.get("max_chars", 12000)),
                )
            if name == "compile_chromatic_invariants":
                query = str(args["query"])
                operation = "debug" if any(token in query.lower() for token in ("debug", "bug", "fix", "regression", "failure")) else "edit"
                contract = self.chromatic_invariant_compiler.compile(
                    query,
                    operation=operation,
                    files=[str(value) for value in args.get("paths", [])],
                    symbols=[str(value) for value in args.get("symbols", [])],
                    acceptance=[str(value) for value in args.get("acceptance", [])],
                )
                return self.chromatic_invariant_compiler.render_contract(
                    contract, int(args.get("max_chars", 12000))
                )
            if name == "inspect_chromatic_invariant_proof":
                cycle = getattr(self, "edit_cycle", None)
                state = getattr(cycle, "state", None) if cycle is not None else None
                if state is None or not getattr(state, "invariant_contract", None):
                    return "[CHROMATIC INVARIANT PROOF]\nnone: no active invariant contract"
                try:
                    report = self.chromatic_invariant_compiler.evaluate(
                        InvariantContract.from_dict(state.invariant_contract),
                        change_fingerprints=dict(state.change_fingerprints),
                        changed_paths=list(state.changed_paths),
                        validators=list(state.validators),
                        validated=bool(state.validated),
                    )
                    state.invariant_proof = report.to_dict()
                    return self.chromatic_invariant_compiler.render_proof(
                        report, int(args.get("max_chars", 12000))
                    )
                except Exception as exc:
                    return f"[CHROMATIC INVARIANT PROOF]\nunavailable={type(exc).__name__}"
            if name == "recall_chromatic_invariant_history":
                cycle = getattr(self, "edit_cycle", None)
                state = getattr(cycle, "state", None) if cycle is not None else None
                contract = dict(getattr(state, "invariant_contract", {}) or {}) if state is not None else {}
                if not contract or self.memory is None or not hasattr(self.memory, "chromatic_invariant_playbook_packet"):
                    return "[CHROMATIC INVARIANT-HISTORY SURFACE]\n- none"
                return await self.memory.chromatic_invariant_playbook_packet(
                    contract,
                    limit=int(args.get("limit", 5)),
                    max_chars=int(args.get("max_chars", 8000)),
                )
            if name == "forecast_chromatic_maintenance":
                query = str(args["query"])
                operation = "debug" if any(token in query.lower() for token in ("debug", "bug", "fix", "regression", "failure")) else "edit"
                predictor = ChromaticPredictiveMaintainer(self.root, getattr(self, "code_intelligence", None))
                predictor.index.build()
                impact = predictor.index.impact(query, max_depth=5, max_nodes=72)
                contract = self.chromatic_invariant_compiler.compile(
                    query,
                    operation=operation,
                    files=[str(value) for value in args.get("paths", [])] or impact.files[:16],
                    symbols=[str(value) for value in args.get("symbols", [])],
                    acceptance=[str(value) for value in args.get("acceptance", [])],
                )
                historical = []
                if self.memory is not None and hasattr(self.memory, "predictive_maintenance_history"):
                    historical = await self.memory.predictive_maintenance_history(
                        query, contract.to_dict(), limit=10
                    )
                validators = validator_candidates(self.root, impact)
                forecast = predictor.forecast(
                    query,
                    operation=operation,
                    contract=contract,
                    historical=historical,
                    validators=validators,
                )
                if self.memory is not None and hasattr(self.memory, "predictive_maintenance_history"):
                    chromatic_history = await self.memory.predictive_maintenance_history(
                        query, contract.to_dict(), risk_address_hex=forecast.risk_address_hex, limit=12
                    )
                    if chromatic_history:
                        forecast = predictor.forecast(
                            query, operation=operation, contract=contract,
                            historical=chromatic_history, validators=validators,
                        )
                self.last_predictive_forecast = forecast.to_dict()
                return predictor.render(forecast, int(args.get("max_chars", 14000)))
            if name == "inspect_chromatic_predictive_forecast":
                cycle = getattr(self, "edit_cycle", None)
                state = getattr(cycle, "state", None) if cycle is not None else None
                payload = dict(getattr(state, "predictive_forecast", {}) or {}) if state is not None else {}
                if not payload:
                    payload = dict(getattr(self, "last_predictive_forecast", {}) or {})
                if not payload:
                    return "[CHROMATIC PREDICTIVE MAINTENANCE]\nnone: no forecast compiled"
                rendered = ChromaticPredictiveMaintainer.render(
                    payload, int(args.get("max_chars", 14000))
                )
                if state is not None:
                    try:
                        outcome = evaluate_predictive_outcome(
                            payload,
                            changed_paths=list(state.changed_paths),
                            failures=int(state.failures),
                            validated=bool(state.validated),
                        )
                        state.predictive_outcome = outcome.to_dict()
                        rendered += "\n\n[PREDICTIVE FORECAST OBSERVATION]\n" + json.dumps(outcome.to_dict(), indent=2)
                    except Exception:
                        pass
                return rendered[: int(args.get("max_chars", 14000))]
            if name == "recall_chromatic_regression_history":
                if self.memory is None or not hasattr(self.memory, "predictive_maintenance_packet"):
                    return "[CHROMATIC REGRESSION-HISTORY SURFACE]\n- none"
                cycle = getattr(self, "edit_cycle", None)
                state = getattr(cycle, "state", None) if cycle is not None else None
                query = str(args.get("query") or (state.intent.request if state is not None else "")).strip()
                contract = dict(getattr(state, "invariant_contract", {}) or {}) if state is not None else {}
                if not query:
                    return "[CHROMATIC REGRESSION-HISTORY SURFACE]\n- none: query required without an active edit cycle"
                risk_address = ""
                if state is not None and isinstance(getattr(state, "predictive_forecast", None), dict):
                    risk_address = str(state.predictive_forecast.get("risk_address_hex") or "")
                return await self.memory.predictive_maintenance_packet(
                    query, contract or None, risk_address_hex=risk_address,
                    limit=int(args.get("limit", 8)),
                    max_chars=int(args.get("max_chars", 9000)),
                )
            if name == "recall_chromatic_change_history":
                if self.memory is None or not hasattr(self.memory, "chromatic_change_playbook_packet"):
                    return "[CHROMATIC CHANGE-HISTORY SURFACE]\n- none"
                fingerprints = dict(getattr(self, "maintenance_change_fingerprints", {}) or {})
                path = str(args.get("path") or "").replace("\\", "/").lstrip("./")
                if path:
                    selected = fingerprints.get(path)
                    fingerprints = {path: selected} if isinstance(selected, dict) else {}
                return await self.memory.chromatic_change_playbook_packet(
                    fingerprints,
                    limit=int(args.get("limit", 5)),
                    max_chars=int(args.get("max_chars", 8000)),
                )
            return await original_execute(self, name, args)

        ToolRegistry.__init__ = init
        ToolRegistry.attach_memory = attach_memory
        ToolRegistry.schemas = schemas
        ToolRegistry._commit_file = commit
        ToolRegistry.execute = execute
        ToolRegistry._humoid_maintenance_runtime = True

    try:
        from .edit_cycle import EditCycleOrchestrator
    except ImportError:
        return
    if getattr(EditCycleOrchestrator, "_humoid_maintenance_runtime", False):
        return

    original_begin = EditCycleOrchestrator.begin
    original_after_tool = EditCycleOrchestrator.after_tool

    async def begin(self: Any, request: str, paths: list[str] | None = None, acceptance: list[str] | None = None, force: bool = False) -> str:
        prompt = await original_begin(self, request, paths, acceptance, force)
        if not self.state:
            return prompt
        packets = [self.state.context]
        additions: list[str] = []
        contract = dict(getattr(self.state, "invariant_contract", {}) or {})
        historical: list[dict[str, Any]] = []
        if self.memory is not None:
            try:
                playbook = await self.memory.coding_playbook_packet(request, limit=4, max_chars=7000)
                if "\n- none" not in playbook:
                    packets.append(playbook)
                    additions.append("procedural")
            except Exception:
                pass
            try:
                if contract and hasattr(self.memory, "chromatic_invariant_playbook_packet"):
                    invariant_history = await self.memory.chromatic_invariant_playbook_packet(
                        contract, limit=4, max_chars=6000
                    )
                    if "\n- none" not in invariant_history:
                        packets.append(invariant_history)
                        additions.append("invariant-history")
            except Exception:
                pass
            try:
                if hasattr(self.memory, "predictive_maintenance_history"):
                    historical = list(await self.memory.predictive_maintenance_history(
                        request, contract or None, limit=10
                    ))
            except Exception:
                historical = []
        try:
            if contract:
                predictor = ChromaticPredictiveMaintainer(self.root)
                predictor.index.build()
                impact = predictor.index.impact(request, max_depth=5, max_nodes=72)
                invariant_contract = InvariantContract.from_dict(contract)
                validators = validator_candidates(self.root, impact)
                forecast = predictor.forecast(
                    request,
                    operation=self.state.intent.operation,
                    contract=invariant_contract,
                    historical=historical,
                    validators=validators,
                )
                if self.memory is not None and hasattr(self.memory, "predictive_maintenance_history"):
                    chromatic_history = list(await self.memory.predictive_maintenance_history(
                        request, contract, risk_address_hex=forecast.risk_address_hex, limit=12
                    ))
                    if chromatic_history:
                        forecast = predictor.forecast(
                            request, operation=self.state.intent.operation,
                            contract=invariant_contract, historical=chromatic_history, validators=validators,
                        )
                self.state.predictive_forecast = forecast.to_dict()
                packets.append(predictor.render(forecast, max_chars=7000))
                additions.append("predictive")
        except Exception:
            self.state.predictive_forecast = {}
        if len(packets) > 1:
            self._set_context(self._merge(packets))
            return self.prompt("initialized with " + " + ".join(additions) + " recall")
        return prompt

    async def after_tool(self: Any, name: str, args: dict[str, Any], output: str, ok: bool) -> str:
        if self.state and ok and name in {"write_file", "apply_patch", "append_file", "insert_text", "replace_text", "apply_file_edits"}:
            match = _FINGERPRINT_RE.search(output)
            if match:
                try:
                    payload = json.loads(match.group(1))
                except (TypeError, ValueError, json.JSONDecodeError):
                    payload = None
                if isinstance(payload, dict):
                    path = str(payload.get("path") or args.get("path") or "").replace("\\", "/").lstrip("./")
                    if path:
                        self.state.change_fingerprints[path] = payload
        result = await original_after_tool(self, name, args, output, ok)
        if self.state and self.state.predictive_forecast:
            try:
                self.state.predictive_outcome = evaluate_predictive_outcome(
                    self.state.predictive_forecast,
                    changed_paths=list(self.state.changed_paths),
                    failures=int(self.state.failures),
                    validated=bool(self.state.validated),
                ).to_dict()
            except Exception:
                pass
        return result

    EditCycleOrchestrator.begin = begin
    EditCycleOrchestrator.after_tool = after_tool
    EditCycleOrchestrator._humoid_maintenance_runtime = True
