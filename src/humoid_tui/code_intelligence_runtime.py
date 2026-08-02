from __future__ import annotations

"""Install incremental impact analysis into tools and edit-cycle prompts."""

import json
from typing import Any

from .code_intelligence import CodeIntelligenceIndex


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def _schemas() -> list[dict[str, Any]]:
    return [
        _schema(
            "build_code_intelligence",
            "Incrementally parse changed repository files into a persistent symbol/call/inheritance graph.",
            {"paths": {"type": "array", "items": {"type": "string"}}},
        ),
        _schema(
            "analyze_change_impact",
            "Return AST-backed anchors, callers, subclasses, imports, tests, blast radius, risk, and uncertainty for a requested change.",
            {
                "query": {"type": "string"},
                "max_depth": {"type": "integer", "minimum": 1, "maximum": 8},
                "max_nodes": {"type": "integer", "minimum": 1, "maximum": 200},
                "max_chars": {"type": "integer", "minimum": 1000, "maximum": 100000},
            },
            ["query"],
        ),
        _schema(
            "benchmark_code_localization",
            "Evaluate localization precision, recall, and reciprocal rank against expected files.",
            {
                "cases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "expected_files": {"type": "array", "items": {"type": "string"}},
                            "max_nodes": {"type": "integer"},
                        },
                        "required": ["query", "expected_files"],
                    },
                }
            },
            ["cases"],
        ),
    ]


def install_code_intelligence_runtime() -> None:
    from .tools import ToolRegistry

    if getattr(ToolRegistry, "_humoid_code_intelligence", False):
        return

    original_init = ToolRegistry.__init__
    original_schemas = ToolRegistry.schemas
    original_execute = ToolRegistry.execute

    def init(self: Any, settings: Any) -> None:
        original_init(self, settings)
        self.code_intelligence = CodeIntelligenceIndex(self.root)

    def schemas(self: Any) -> list[dict[str, Any]]:
        existing = original_schemas(self)
        names = {item["function"]["name"] for item in existing}
        return [*([item for item in _schemas() if item["function"]["name"] not in names]), *existing]

    async def execute(self: Any, name: str, args: dict[str, Any]) -> str:
        if name == "build_code_intelligence":
            stats = self.code_intelligence.build([str(value) for value in args.get("paths", [])] or None)
            return json.dumps(stats, indent=2)
        if name == "analyze_change_impact":
            self.code_intelligence.build()
            report = self.code_intelligence.impact(
                str(args["query"]),
                max_depth=int(args.get("max_depth", 3)),
                max_nodes=int(args.get("max_nodes", 30)),
            )
            return report.render(int(args.get("max_chars", 12000)))
        if name == "benchmark_code_localization":
            self.code_intelligence.build()
            return json.dumps(self.code_intelligence.benchmark(list(args.get("cases") or [])), indent=2)
        return await original_execute(self, name, args)

    ToolRegistry.__init__ = init
    ToolRegistry.schemas = schemas
    ToolRegistry.execute = execute
    ToolRegistry._humoid_code_intelligence = True

    try:
        from .edit_cycle import EditCycleOrchestrator
    except ImportError:
        return
    if getattr(EditCycleOrchestrator, "_humoid_code_intelligence", False):
        return

    original_begin = EditCycleOrchestrator.begin
    original_refresh = EditCycleOrchestrator.refresh

    async def begin(self: Any, request: str, paths: list[str] | None = None, acceptance: list[str] | None = None, force: bool = False) -> str:
        prompt = await original_begin(self, request, paths, acceptance, force)
        registry = getattr(self.perspective, "_tool_registry", None)
        intelligence = getattr(registry, "code_intelligence", None)
        if intelligence is None:
            intelligence = CodeIntelligenceIndex(self.root)
        intelligence.build(paths)
        report = intelligence.impact(request, max_depth=3, max_nodes=36)
        if self.state:
            self._set_context(self._merge([self.state.context, report.render(14000)]))
            return self.prompt("initialized with static impact frontier")
        return prompt

    async def refresh(self: Any, query: str | None = None, reason: str = "manual refresh") -> str:
        prompt = await original_refresh(self, query, reason)
        if not self.state:
            return prompt
        intelligence = CodeIntelligenceIndex(self.root)
        intelligence.build(self.state.changed_paths or None)
        report = intelligence.impact(query or self.state.intent.request, max_depth=4, max_nodes=40)
        self._set_context(self._merge([self.state.context, report.render(14000)]))
        return self.prompt(f"{reason}; refreshed static impact frontier")

    EditCycleOrchestrator.begin = begin
    EditCycleOrchestrator.refresh = refresh
    EditCycleOrchestrator._humoid_code_intelligence = True
