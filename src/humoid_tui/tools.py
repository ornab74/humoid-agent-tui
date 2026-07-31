from __future__ import annotations
import asyncio
import difflib
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from .config import Settings

class ToolError(RuntimeError): pass

class ToolRegistry:
    def __init__(self, settings: Settings):
        self.s = settings
        self.root = settings.humoid_workspace.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.approve_write: Callable[[str, str, str], Awaitable[tuple[bool, str]]] | None = None
        self.undo_stack: list[tuple[Path, str | None]] = []
        self.invented: dict[str, dict[str, Any]] = {}

    def _path(self, relative: str) -> Path:
        p = (self.root / relative).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError("Path escapes workspace")
        return p

    def schemas(self):
        schemas = [
          {"type":"function","function":{"name":"list_files","description":"List workspace files.",
            "parameters":{"type":"object","properties":{"path":{"type":"string"}}}}},
          {"type":"function","function":{"name":"read_file","description":"Read a UTF-8 workspace file.",
            "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
          {"type":"function","function":{"name":"write_file","description":"Write a UTF-8 workspace file.",
            "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},
                          "required":["path","content"]}}},
          {"type":"function","function":{"name":"run_command","description":"Run a command in the workspace when enabled.",
            "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
          {"type":"function","function":{"name":"invent_tool","description":"Create and validate a temporary read-only tool composed from existing tools.",
            "parameters":{"type":"object","properties":{"name":{"type":"string"},"description":{"type":"string"},
            "steps":{"type":"array","items":{"type":"object"}},"validation_cases":{"type":"array","items":{"type":"object"}},
            "promote":{"type":"boolean"}},"required":["name","description","steps","validation_cases"]}}},
          {"type":"function","function":{"name":"undo_file_change","description":"Undo the most recent file write in this session.",
            "parameters":{"type":"object","properties":{}}}},
        ]
        for name, specification in self.invented.items():
            schemas.append({"type": "function", "function": {
                "name": name,
                "description": specification["description"],
                "parameters": {"type": "object", "properties": {}},
            }})
        return schemas

    async def _write_file(self, relative: str, content: str) -> str:
        path = self._path(relative)
        previous = path.read_text(errors="replace") if path.exists() else None
        before = previous or ""
        diff = "".join(difflib.unified_diff(
            before.splitlines(keepends=True), content.splitlines(keepends=True),
            fromfile=f"a/{relative}", tofile=f"b/{relative}",
        ))
        if self.approve_write:
            approved, content = await self.approve_write(relative, diff or "(no textual changes)", content)
            if not approved:
                raise ToolError("File change rejected by user")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.undo_stack.append((path, previous))
        path.write_text(content)
        return f"wrote {path.relative_to(self.root)} ({path.stat().st_size} bytes)"

    async def _undo(self) -> str:
        if not self.undo_stack:
            return "nothing to undo"
        path, previous = self.undo_stack.pop()
        if previous is None:
            path.unlink(missing_ok=True)
            return f"removed newly-created {path.relative_to(self.root)}"
        path.write_text(previous)
        return f"restored {path.relative_to(self.root)}"

    async def _invent(self, args: dict[str, Any]) -> str:
        name = str(args["name"]).strip()
        if not name.replace("_", "a").isalnum() or name in {item["function"]["name"] for item in self.schemas()}:
            raise ToolError("Invented tool name is invalid or already exists")
        steps = args.get("steps") or []
        cases = args.get("validation_cases") or []
        if not steps or not cases:
            raise ToolError("An invented tool requires steps and validation cases")
        if any(step.get("tool") not in {"list_files", "read_file"} for step in steps):
            raise ToolError("Temporary tools may compose only read-only tools")
        specification = {"description": str(args["description"]), "steps": steps}
        self.invented[name] = specification
        try:
            for case in cases[:5]:
                output = await self._execute_invented(name)
                expected = str(case.get("contains", ""))
                if expected and expected not in output:
                    raise ToolError(f"Validation failed: output did not contain {expected!r}")
        except Exception:
            self.invented.pop(name, None)
            raise
        if args.get("promote"):
            path = Path(".humoid/invented_tools.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.invented, indent=2))
        return f"invented and validated temporary tool {name} ({len(cases[:5])} cases)"

    async def _execute_invented(self, name: str) -> str:
        outputs = []
        for step in self.invented[name]["steps"]:
            outputs.append(await self.execute(step["tool"], step.get("arguments", {})))
        return "\n\n".join(outputs)

    async def execute(self, name: str, args: dict[str,Any]) -> str:
        if name == "list_files":
            p=self._path(args.get("path","."))
            return "\n".join(str(x.relative_to(self.root)) for x in sorted(p.rglob("*")) if x.is_file())[:20000]
        if name == "read_file":
            return self._path(args["path"]).read_text(errors="replace")[:100000]
        if name == "write_file":
            return await self._write_file(args["path"], args["content"])
        if name == "undo_file_change":
            return await self._undo()
        if name == "invent_tool":
            return await self._invent(args)
        if name in self.invented:
            return await self._execute_invented(name)
        if name == "run_command":
            if not self.s.humoid_allow_shell: raise ToolError("Shell is disabled")
            proc=await asyncio.create_subprocess_shell(args["command"], cwd=self.root,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try: out,_=await asyncio.wait_for(proc.communicate(),self.s.humoid_shell_timeout_seconds)
            except TimeoutError:
                proc.kill(); raise ToolError("Command timed out")
            except asyncio.CancelledError:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), 3)
                except TimeoutError:
                    proc.kill()
                raise
            return out.decode(errors="replace")[:50000]
        raise ToolError(f"Unknown tool: {name}")
