from __future__ import annotations

"""Install edit-cycle policy around the existing ToolRegistry and AgentHarness."""

import asyncio
import hashlib
import json
from typing import Any

from .edit_cycle import EditCycleOrchestrator


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def _cycle_schemas() -> list[dict[str, Any]]:
    return [
        _schema(
            "start_edit_cycle",
            "Start the mandatory context-building workflow for a file edit. Pass the original request verbatim.",
            {
                "request": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}},
                "acceptance": {"type": "array", "items": {"type": "string"}},
                "force": {"type": "boolean"},
            },
            ["request"],
        ),
        _schema(
            "refresh_edit_cycle",
            "Retrieve a narrower differential behavior subgraph after a changed hypothesis or evidence gap.",
            {"query": {"type": "string"}, "reason": {"type": "string"}},
        ),
        _schema("edit_cycle_status", "Return phase, evidence ledger, grounded files, changes, and validators.", {}),
        _schema("finish_edit_cycle", "Close the active cycle before the final completion message.", {"summary": {"type": "string"}}),
        _schema(
            "read_file_range",
            "Read a bounded line range with line numbers and the full-file SHA-256 digest.",
            {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            ["path", "start_line", "end_line"],
        ),
        _schema(
            "run_validation",
            "Run a repository validator and preserve its real exit status for the edit-cycle verifier.",
            {"command": {"type": "string"}},
            ["command"],
        ),
        _schema(
            "apply_patch",
            "Apply exact digest-checked replacements to one grounded existing file without regenerating it.",
            {
                "path": {"type": "string"},
                "expected_sha256": {"type": "string"},
                "replacements": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "old": {"type": "string"},
                            "new": {"type": "string"},
                            "count": {"type": "integer", "minimum": 1, "maximum": 100},
                        },
                        "required": ["old", "new"],
                    },
                },
            },
            ["path", "expected_sha256", "replacements"],
        ),
    ]


def _read_range(registry: Any, args: dict[str, Any]) -> str:
    path = registry._path(str(args["path"]))
    source = path.read_text(errors="replace")
    lines = source.splitlines()
    start = max(1, int(args["start_line"]))
    end = min(len(lines), int(args["end_line"]))
    if end < start:
        raise ValueError("end_line must be greater than or equal to start_line")
    digest = hashlib.sha256(source.encode(errors="replace")).hexdigest()
    body = "\n".join(f"{number:6d} | {lines[number - 1]}" for number in range(start, end + 1))
    return f"[FILE RANGE {path.relative_to(registry.root)}:{start}-{end} sha256={digest}]\n{body}"


async def _apply_patch(registry: Any, args: dict[str, Any]) -> str:
    relative = str(args["path"])
    path = registry._path(relative)
    source = path.read_text(errors="replace")
    digest = hashlib.sha256(source.encode(errors="replace")).hexdigest()
    expected = str(args.get("expected_sha256", "")).lower()
    if expected != digest:
        return f"ERROR: stale patch digest; expected={expected or 'missing'} current={digest}"
    updated = source
    replacements = args.get("replacements") or []
    if not replacements:
        return "ERROR: apply_patch requires at least one replacement"
    for index, replacement in enumerate(replacements, start=1):
        old = str(replacement.get("old", ""))
        new = str(replacement.get("new", ""))
        count = int(replacement.get("count", 1))
        actual = updated.count(old)
        if not old or actual != count:
            return f"ERROR: replacement {index} expected {count} exact occurrence(s), found {actual}"
        updated = updated.replace(old, new, count)
    if updated == source:
        return "ERROR: apply_patch produced no textual change"
    result = await registry._write_file(relative, updated)
    new_digest = hashlib.sha256(updated.encode(errors="replace")).hexdigest()
    return f"{result}; sha256={new_digest}; replacements={len(replacements)}"


async def _run_validation(registry: Any, args: dict[str, Any]) -> tuple[str, bool]:
    command = str(args.get("command", "")).strip()
    if not command:
        return "ERROR: run_validation requires a command", False
    if not registry.s.humoid_allow_shell:
        return "ERROR: shell is disabled; enable HUMOID_ALLOW_SHELL in an isolated workspace", False
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=registry.root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await asyncio.wait_for(proc.communicate(), registry.s.humoid_shell_timeout_seconds)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return f"ERROR: validation timed out: {command}", False
    except asyncio.CancelledError:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 3)
        except TimeoutError:
            proc.kill()
        raise
    text = output.decode(errors="replace")[:50000]
    if proc.returncode != 0:
        return f"ERROR: validator exited {proc.returncode}: {command}\n{text}", False
    return f"validator passed: {command}\n{text}", True


def _safe_status(cycle: Any) -> str:
    state = cycle.state
    if state is None:
        return json.dumps({"active": False}, indent=2)
    ledger = [
        {
            "sequence": event.sequence,
            "kind": event.kind,
            "summary": event.summary,
            "ok": event.ok,
            "digest": event.digest,
        }
        for event in state.ledger[-20:]
    ]
    payload = {
        "active": state.active,
        "phase": state.phase,
        "cycle": getattr(state, "cycle", getattr(state, "cycle_no", 0)),
        "context_revision": getattr(state, "revision", getattr(state, "context_revision", 0)),
        "context_digest": state.context_digest,
        "task_id": state.intent.task_id,
        "operation": state.intent.operation,
        "grounded_paths": sorted(state.grounded_paths),
        "exact_reads": sorted(state.exact_reads),
        "changed_paths": state.changed_paths,
        "validators": getattr(state, "validators", getattr(state, "validation_commands", [])),
        "validated": state.validated,
        "failures": getattr(state, "failures", getattr(state, "failure_count", 0)),
        "latest_error": state.latest_error,
        "ledger": ledger,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def install_tool_edit_cycle_runtime() -> None:
    from .tools import ToolRegistry

    if getattr(ToolRegistry, "_humoid_edit_cycle_runtime", False):
        return
    original_init = ToolRegistry.__init__
    original_schemas = ToolRegistry.schemas
    original_execute = ToolRegistry.execute
    original_close = ToolRegistry.close

    def init(self: Any, settings: Any) -> None:
        original_init(self, settings)
        self.edit_cycle = EditCycleOrchestrator(settings, self.root, self.perspective)

    def schemas(self: Any) -> list[dict[str, Any]]:
        existing = original_schemas(self)
        names = {item["function"]["name"] for item in existing}
        additions = [item for item in _cycle_schemas() if item["function"]["name"] not in names]
        for item in existing:
            if item["function"]["name"] in {"write_file", "read_file"}:
                item["function"]["description"] += " Existing edit targets are evidence-gated by the active edit cycle."
        return [*additions, *existing]

    async def execute(self: Any, name: str, args: dict[str, Any]) -> str:
        cycle = self.edit_cycle
        if name == "start_edit_cycle":
            return await cycle.begin(
                str(args["request"]),
                [str(value) for value in args.get("paths", [])] or None,
                [str(value) for value in args.get("acceptance", [])] or None,
                bool(args.get("force", False)),
            )
        if name == "refresh_edit_cycle":
            return await cycle.refresh(str(args.get("query") or "") or None, str(args.get("reason") or "manual refresh"))
        if name == "edit_cycle_status":
            return _safe_status(cycle)
        if name == "finish_edit_cycle":
            return await cycle.finish(str(args.get("summary") or ""))
        if name == "read_file_range":
            gate = cycle.before_tool(name, args)
            if gate:
                return gate
            output = _read_range(self, args)
            return await cycle.after_tool(name, args, output, True)
        if name == "apply_patch":
            gate = cycle.before_tool(name, args)
            if gate:
                return gate
            output = await _apply_patch(self, args)
            return await cycle.after_tool(name, args, output, not output.startswith("ERROR:"))
        if name == "run_validation":
            gate = cycle.before_tool(name, args)
            if gate:
                return gate
            output, ok = await _run_validation(self, args)
            return await cycle.after_tool("run_command", {"command": args.get("command", "")}, output, ok)
        if name == "clear_project_perspective" and cycle.active:
            return "ERROR: finish_edit_cycle before clearing project perspective"
        gate = cycle.before_tool(name, args)
        if gate:
            return gate
        try:
            output = str(await original_execute(self, name, args))
        except Exception as exc:
            if cycle.active and cycle.state:
                cycle.state.phase = "repair"
                cycle.state.failures += 1
                cycle.state.latest_error = f"{type(exc).__name__}: {exc}"
            raise
        return await cycle.after_tool(name, args, output, not output.startswith("ERROR:"))

    async def close(self: Any) -> None:
        if self.edit_cycle.active:
            await self.edit_cycle.finish("registry closed")
        await original_close(self)

    ToolRegistry.__init__ = init
    ToolRegistry.schemas = schemas
    ToolRegistry.execute = execute
    ToolRegistry.close = close
    ToolRegistry._humoid_edit_cycle_runtime = True


def install_agent_edit_cycle_runtime() -> None:
    from .agent import AgentHarness

    if getattr(AgentHarness, "_humoid_edit_cycle_runtime", False):
        return
    original_run = AgentHarness.run
    original_init = AgentHarness.__init__

    def init(self: Any, settings: Any, emit: Any) -> None:
        original_init(self, settings, emit)
        self.tools.edit_cycle.memory = self.memory

    async def run(self: Any, prompt: str) -> str:
        cleaned = prompt.strip()
        cycle = self.tools.edit_cycle
        if cleaned and cycle.compiler.is_edit_request(cleaned):
            cycle_prompt = await cycle.begin(cleaned)
            cleaned = (
                f"{cleaned}\n\nThe runtime already initialized the edit cycle; do not call start_edit_cycle again.\n\n"
                f"{cycle_prompt}"
            )
        return await original_run(self, cleaned)

    AgentHarness.__init__ = init
    AgentHarness.run = run
    AgentHarness._humoid_edit_cycle_runtime = True


def install_edit_cycle_runtime() -> None:
    install_tool_edit_cycle_runtime()
    install_agent_edit_cycle_runtime()
