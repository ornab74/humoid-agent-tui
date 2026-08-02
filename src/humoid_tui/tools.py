from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from .config import Settings
from .perspective import ProjectPerspectiveIndex


class ToolError(RuntimeError):
    pass


class ToolRegistry:
    def __init__(self, settings: Settings):
        self.s = settings
        self.root = settings.humoid_workspace.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.approve_write: Callable[[str, str, str], Awaitable[tuple[bool, str]]] | None = None
        self.undo_stack: list[tuple[Path, str | None]] = []
        self.invented: dict[str, dict[str, Any]] = {}
        self.perspective = ProjectPerspectiveIndex(settings, self.root)

    def _path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if path != self.root and self.root not in path.parents:
            raise ToolError("Path escapes workspace")
        return path

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _occurrence_index(text: str, needle: str, occurrence: int) -> int:
        if occurrence == 0:
            raise ToolError("occurrence is 1-based and cannot be zero")
        if occurrence < 0:
            position = len(text)
            for _ in range(abs(occurrence)):
                position = text.rfind(needle, 0, position)
                if position < 0:
                    return -1
            return position
        position = -1
        start = 0
        for _ in range(occurrence):
            position = text.find(needle, start)
            if position < 0:
                return -1
            start = position + len(needle)
        return position

    def schemas(self):
        schemas = [
            {"type": "function", "function": {
                "name": "list_files",
                "description": "List workspace files. Prefer build_project_perspective once for broad repository reviews instead of repeatedly listing and reading files.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }},
            {"type": "function", "function": {
                "name": "read_file",
                "description": "Read an exact UTF-8 workspace file. During project review, search the project perspective first and use this only for exact syntax before editing.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            }},
            {"type": "function", "function": {
                "name": "read_file_range",
                "description": "Read only a bounded line range from a UTF-8 workspace file. Prefer this over reading an entire large file when verifying an edit anchor.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                }, "required": ["path", "start_line", "end_line"]},
            }},
            {"type": "function", "function": {
                "name": "write_file",
                "description": "Replace a complete UTF-8 workspace file. Use only for new files or intentional whole-file regeneration; prefer append_file, insert_text, replace_text, or apply_file_edits for existing files.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"}, "content": {"type": "string"},
                    "expected_sha256": {"type": "string", "description": "Optional hash of the current file to reject stale edits."},
                }, "required": ["path", "content"]},
            }},
            {"type": "function", "function": {
                "name": "append_file",
                "description": "Append text to an existing file without resending or rewriting its full contents. Can create the file when requested and can enforce a current-content SHA-256 precondition.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"}, "content": {"type": "string"},
                    "ensure_newline": {"type": "boolean", "description": "Insert one newline before appended content when the file does not already end with one."},
                    "create": {"type": "boolean", "description": "Create the file when it does not exist."},
                    "expected_sha256": {"type": "string"},
                }, "required": ["path", "content"]},
            }},
            {"type": "function", "function": {
                "name": "insert_text",
                "description": "Surgically insert text immediately before or after an exact anchor. Fails safely when the requested occurrence is absent or ambiguous.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"}, "anchor": {"type": "string"}, "content": {"type": "string"},
                    "position": {"type": "string", "enum": ["before", "after"]},
                    "occurrence": {"type": "integer", "description": "1-based occurrence; -1 means last occurrence."},
                    "expected_matches": {"type": "integer", "minimum": 1},
                    "expected_sha256": {"type": "string"},
                }, "required": ["path", "anchor", "content", "position"]},
            }},
            {"type": "function", "function": {
                "name": "replace_text",
                "description": "Surgically replace or delete exact text in a file. By default requires exactly one match; set expected_matches explicitly for intentional multi-replacement.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"},
                    "expected_matches": {"type": "integer", "minimum": 1},
                    "replace_all": {"type": "boolean"},
                    "expected_sha256": {"type": "string"},
                }, "required": ["path", "old_text", "new_text"]},
            }},
            {"type": "function", "function": {
                "name": "apply_file_edits",
                "description": "Apply a sequence of append, prepend, insert-before, insert-after, replace, or delete operations atomically to one file. The file is written once, reviewed as one diff, and rolled back as one undo step.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"},
                    "expected_sha256": {"type": "string"},
                    "edits": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
                        "operation": {"type": "string", "enum": ["append", "prepend", "insert_before", "insert_after", "replace", "delete"]},
                        "anchor": {"type": "string"}, "old_text": {"type": "string"}, "content": {"type": "string"},
                        "occurrence": {"type": "integer"}, "expected_matches": {"type": "integer", "minimum": 1},
                        "replace_all": {"type": "boolean"}, "ensure_newline": {"type": "boolean"},
                    }, "required": ["operation"]}},
                }, "required": ["path", "edits"]},
            }},
            {"type": "function", "function": {
                "name": "run_command",
                "description": "Run a command in the workspace when enabled.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
            }},
            {"type": "function", "function": {
                "name": "build_project_perspective",
                "description": "Index the project once for a review/build task. Chunks text files, creates a temporary task-scoped Weaviate collection when available, and falls back to private local vectors. Call this before broad project review instead of repeatedly opening files.",
                "parameters": {"type": "object", "properties": {
                    "objective": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}, "force": {"type": "boolean"},
                }, "required": ["objective"]},
            }},
            {"type": "function", "function": {
                "name": "search_project_perspective",
                "description": "Retrieve a bounded, diverse repository evidence packet from the current task perspective. Lower-ranked evidence is folded into a context accordion.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    "max_chars": {"type": "integer", "minimum": 1000, "maximum": 100000},
                }, "required": ["query"]},
            }},
            {"type": "function", "function": {
                "name": "expand_project_perspective",
                "description": "Expand specific perspective chunk IDs and nearby chunks only after search identifies a concrete evidence gap.",
                "parameters": {"type": "object", "properties": {
                    "chunk_ids": {"type": "array", "items": {"type": "string"}}, "radius": {"type": "integer", "minimum": 0, "maximum": 4},
                    "max_chars": {"type": "integer", "minimum": 1000, "maximum": 100000},
                }, "required": ["chunk_ids"]},
            }},
            {"type": "function", "function": {
                "name": "project_perspective_status", "description": "Show the active perspective backend, task objective, manifest, chunk count, and local deltas.",
                "parameters": {"type": "object", "properties": {}},
            }},
            {"type": "function", "function": {
                "name": "clear_project_perspective", "description": "Delete the temporary project perspective collection and local task cache when the task is complete.",
                "parameters": {"type": "object", "properties": {}},
            }},
            {"type": "function", "function": {
                "name": "invent_tool", "description": "Create and validate a temporary read-only tool composed from existing tools.",
                "parameters": {"type": "object", "properties": {
                    "name": {"type": "string"}, "description": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "object"}},
                    "validation_cases": {"type": "array", "items": {"type": "object"}}, "promote": {"type": "boolean"},
                }, "required": ["name", "description", "steps", "validation_cases"]},
            }},
            {"type": "function", "function": {
                "name": "undo_file_change", "description": "Undo the most recent file mutation in this session and synchronize the active perspective.",
                "parameters": {"type": "object", "properties": {}},
            }},
        ]
        for name, specification in self.invented.items():
            schemas.append({"type": "function", "function": {
                "name": name, "description": specification["description"],
                "parameters": {"type": "object", "properties": {}},
            }})
        return schemas

    def _read_existing(self, relative: str, *, create: bool = False) -> tuple[Path, str | None, str]:
        path = self._path(relative)
        if not path.exists():
            if not create:
                raise ToolError(f"File does not exist: {relative}")
            return path, None, ""
        if not path.is_file():
            raise ToolError(f"Path is not a file: {relative}")
        previous = path.read_text(encoding="utf-8", errors="replace")
        return path, previous, previous

    def _check_precondition(self, current: str, expected_sha256: str | None) -> None:
        if expected_sha256 and self._sha256(current) != expected_sha256.strip().lower():
            raise ToolError(
                "File changed since it was inspected; SHA-256 precondition failed. "
                f"Current SHA-256 is {self._sha256(current)}"
            )

    async def _commit_file(self, relative: str, path: Path, previous: str | None, content: str) -> str:
        before = previous or ""
        if content == before and previous is not None:
            return f"no changes to {relative}; sha256={self._sha256(content)}"
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
        path.write_text(content, encoding="utf-8")
        await self.perspective.notify_file_changed(relative, content)
        return (
            f"updated {path.relative_to(self.root)} ({path.stat().st_size} bytes); "
            f"sha256={self._sha256(content)}"
        )

    async def _write_file(self, relative: str, content: str, expected_sha256: str | None = None) -> str:
        path, previous, current = self._read_existing(relative, create=True)
        self._check_precondition(current, expected_sha256)
        return await self._commit_file(relative, path, previous, content)

    async def _append_file(self, args: dict[str, Any]) -> str:
        relative = str(args["path"])
        path, previous, current = self._read_existing(relative, create=bool(args.get("create", False)))
        self._check_precondition(current, args.get("expected_sha256"))
        separator = "\n" if bool(args.get("ensure_newline", True)) and current and not current.endswith("\n") else ""
        return await self._commit_file(relative, path, previous, current + separator + str(args["content"]))

    async def _insert_text(self, args: dict[str, Any]) -> str:
        relative = str(args["path"])
        path, previous, current = self._read_existing(relative)
        self._check_precondition(current, args.get("expected_sha256"))
        anchor = str(args["anchor"])
        if not anchor:
            raise ToolError("insert_text anchor cannot be empty")
        matches = current.count(anchor)
        expected = int(args.get("expected_matches", 1))
        if matches != expected:
            raise ToolError(f"Anchor match count was {matches}, expected {expected}; refusing ambiguous edit")
        occurrence = int(args.get("occurrence", 1))
        index = self._occurrence_index(current, anchor, occurrence)
        if index < 0:
            raise ToolError(f"Requested anchor occurrence {occurrence} was not found")
        insertion = index if args["position"] == "before" else index + len(anchor)
        updated = current[:insertion] + str(args["content"]) + current[insertion:]
        return await self._commit_file(relative, path, previous, updated)

    async def _replace_text(self, args: dict[str, Any]) -> str:
        relative = str(args["path"])
        path, previous, current = self._read_existing(relative)
        self._check_precondition(current, args.get("expected_sha256"))
        old = str(args["old_text"])
        if not old:
            raise ToolError("replace_text old_text cannot be empty")
        matches = current.count(old)
        expected = int(args.get("expected_matches", 1))
        if matches != expected:
            raise ToolError(f"Replacement match count was {matches}, expected {expected}; refusing ambiguous edit")
        count = -1 if bool(args.get("replace_all", False)) else 1
        updated = current.replace(old, str(args["new_text"]), count)
        return await self._commit_file(relative, path, previous, updated)

    def _apply_one_edit(self, text: str, edit: dict[str, Any]) -> str:
        operation = str(edit.get("operation", ""))
        content = str(edit.get("content", ""))
        if operation == "append":
            separator = "\n" if bool(edit.get("ensure_newline", True)) and text and not text.endswith("\n") else ""
            return text + separator + content
        if operation == "prepend":
            separator = "\n" if bool(edit.get("ensure_newline", False)) and content and not content.endswith("\n") else ""
            return content + separator + text
        if operation in {"insert_before", "insert_after"}:
            anchor = str(edit.get("anchor", ""))
            if not anchor:
                raise ToolError(f"{operation} requires a non-empty anchor")
            matches = text.count(anchor)
            expected = int(edit.get("expected_matches", 1))
            if matches != expected:
                raise ToolError(f"{operation} anchor matched {matches} times, expected {expected}")
            occurrence = int(edit.get("occurrence", 1))
            index = self._occurrence_index(text, anchor, occurrence)
            if index < 0:
                raise ToolError(f"{operation} occurrence {occurrence} was not found")
            insertion = index if operation == "insert_before" else index + len(anchor)
            return text[:insertion] + content + text[insertion:]
        if operation in {"replace", "delete"}:
            old = str(edit.get("old_text", ""))
            if not old:
                raise ToolError(f"{operation} requires non-empty old_text")
            matches = text.count(old)
            expected = int(edit.get("expected_matches", 1))
            if matches != expected:
                raise ToolError(f"{operation} matched {matches} times, expected {expected}")
            replacement = "" if operation == "delete" else content
            count = -1 if bool(edit.get("replace_all", False)) else 1
            return text.replace(old, replacement, count)
        raise ToolError(f"Unsupported file edit operation: {operation!r}")

    async def _apply_file_edits(self, args: dict[str, Any]) -> str:
        relative = str(args["path"])
        path, previous, current = self._read_existing(relative)
        self._check_precondition(current, args.get("expected_sha256"))
        updated = current
        for index, edit in enumerate(args.get("edits") or [], start=1):
            try:
                updated = self._apply_one_edit(updated, dict(edit))
            except ToolError as exc:
                raise ToolError(f"Atomic edit {index} failed; no file changes were written: {exc}") from exc
        return await self._commit_file(relative, path, previous, updated)

    async def _undo(self) -> str:
        if not self.undo_stack:
            return "nothing to undo"
        path, previous = self.undo_stack.pop()
        relative = path.relative_to(self.root).as_posix()
        if previous is None:
            path.unlink(missing_ok=True)
            await self.perspective.notify_file_deleted(relative)
            return f"removed newly-created {relative}"
        path.write_text(previous, encoding="utf-8")
        await self.perspective.notify_file_changed(relative, previous)
        return f"restored {relative}; sha256={self._sha256(previous)}"

    async def _invent(self, args: dict[str, Any]) -> str:
        name = str(args["name"]).strip()
        if not name.replace("_", "a").isalnum() or name in {item["function"]["name"] for item in self.schemas()}:
            raise ToolError("Invented tool name is invalid or already exists")
        steps = args.get("steps") or []
        cases = args.get("validation_cases") or []
        if not steps or not cases:
            raise ToolError("An invented tool requires steps and validation cases")
        readonly = {"list_files", "read_file", "read_file_range", "search_project_perspective", "expand_project_perspective", "project_perspective_status"}
        if any(step.get("tool") not in readonly for step in steps):
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
            path.write_text(json.dumps(self.invented, indent=2), encoding="utf-8")
        return f"invented and validated temporary tool {name} ({len(cases[:5])} cases)"

    async def _execute_invented(self, name: str) -> str:
        outputs = []
        for step in self.invented[name]["steps"]:
            outputs.append(await self.execute(step["tool"], step.get("arguments", {})))
        return "\n\n".join(outputs)

    async def execute(self, name: str, args: dict[str, Any]) -> str:
        if name == "list_files":
            path = self._path(args.get("path", "."))
            return "\n".join(str(item.relative_to(self.root)) for item in sorted(path.rglob("*")) if item.is_file())[:20000]
        if name == "read_file":
            text = self._path(args["path"]).read_text(encoding="utf-8", errors="replace")
            return f"sha256={self._sha256(text)}\n{text[:100000]}"
        if name == "read_file_range":
            start, end = int(args["start_line"]), int(args["end_line"])
            if end < start:
                raise ToolError("end_line must be greater than or equal to start_line")
            text = self._path(args["path"]).read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            numbered = "\n".join(f"{number}: {lines[number - 1]}" for number in range(start, min(end, len(lines)) + 1))
            return f"sha256={self._sha256(text)}\n{numbered}"
        if name == "write_file":
            return await self._write_file(str(args["path"]), str(args["content"]), args.get("expected_sha256"))
        if name == "append_file":
            return await self._append_file(args)
        if name == "insert_text":
            return await self._insert_text(args)
        if name == "replace_text":
            return await self._replace_text(args)
        if name == "apply_file_edits":
            return await self._apply_file_edits(args)
        if name == "build_project_perspective":
            return await self.perspective.build(
                str(args["objective"]), [str(value) for value in (args.get("paths") or [])] or None,
                bool(args.get("force", False)),
            )
        if name == "search_project_perspective":
            return await self.perspective.search(
                str(args["query"]), int(args["limit"]) if args.get("limit") is not None else None,
                int(args["max_chars"]) if args.get("max_chars") is not None else None,
            )
        if name == "expand_project_perspective":
            return await self.perspective.expand(
                [str(value) for value in args.get("chunk_ids", [])], int(args.get("radius", 1)),
                int(args["max_chars"]) if args.get("max_chars") is not None else None,
            )
        if name == "project_perspective_status":
            return self.perspective.status_json()
        if name == "clear_project_perspective":
            return await self.perspective.clear()
        if name == "undo_file_change":
            return await self._undo()
        if name == "invent_tool":
            return await self._invent(args)
        if name in self.invented:
            return await self._execute_invented(name)
        if name == "run_command":
            if not self.s.humoid_allow_shell:
                raise ToolError("Shell is disabled")
            process = await asyncio.create_subprocess_shell(
                args["command"], cwd=self.root,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            try:
                output, _ = await asyncio.wait_for(process.communicate(), self.s.humoid_shell_timeout_seconds)
            except TimeoutError:
                process.kill()
                raise ToolError("Command timed out")
            except asyncio.CancelledError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 3)
                except TimeoutError:
                    process.kill()
                raise
            return output.decode(errors="replace")[:50000]
        raise ToolError(f"Unknown tool: {name}")

    async def close(self) -> None:
        await self.perspective.clear()
