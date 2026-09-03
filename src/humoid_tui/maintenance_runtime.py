from __future__ import annotations

"""Runtime integration for semantic diffs, maintenance planning, and playbooks."""

import json
import re
from typing import Any

from .chromatic_system import encode_change_delta
from .maintenance import (
    MaintenanceHealthAnalyzer,
    MaintenancePlanner,
    semantic_change_fingerprint,
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
        if not self.state or self.memory is None:
            return prompt
        try:
            playbook = await self.memory.coding_playbook_packet(request, limit=4, max_chars=7000)
        except Exception:
            return prompt
        if "\n- none" not in playbook:
            self._set_context(self._merge([self.state.context, playbook]))
            return self.prompt("initialized with procedural maintenance recall")
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
        return await original_after_tool(self, name, args, output, ok)

    EditCycleOrchestrator.begin = begin
    EditCycleOrchestrator.after_tool = after_tool
    EditCycleOrchestrator._humoid_maintenance_runtime = True
