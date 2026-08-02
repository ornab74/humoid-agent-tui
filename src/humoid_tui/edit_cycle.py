from __future__ import annotations

"""Evidence-gated context compiler for surgical repository edits."""

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

FILE_RE = re.compile(r"(?<![\w.-])(?:[\w.-]+/)*[\w.-]+\.(?:py|pyi|js|mjs|cjs|ts|tsx|jsx|dart|go|rs|java|kt|kts|c|cc|cpp|h|hpp|cs|rb|php|swift|scala|sh|sql|html|css|scss|json|ya?ml|toml|md)\b", re.I)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")
CITATION_RE = re.compile(r"(?P<path>(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]+):\d+-\d+")
NODE_RE = re.compile(r"\[PERSPECTIVE NODE\s+([^\s|]+)")
EDIT_WORDS = {"add", "change", "create", "delete", "edit", "fix", "implement", "modify", "move", "patch", "refactor", "remove", "rename", "replace", "rewrite", "update"}
DEBUG_WORDS = {"bug", "crash", "debug", "error", "exception", "fail", "failed", "failure", "fix", "regression", "traceback"}
RISK_HINTS = {
    "auth": {"auth", "credential", "login", "permission", "session", "token"},
    "concurrency": {"async", "concurrent", "deadlock", "lock", "parallel", "race", "thread"},
    "storage": {"database", "migration", "schema", "sqlite", "storage", "weaviate"},
    "network": {"api", "client", "http", "request", "response", "server", "socket"},
    "ui": {"component", "screen", "tui", "ui", "view", "widget"},
}
VALIDATORS = {"pytest", "unittest", "ruff", "tox", "npm test", "pnpm test", "yarn test", "tsc", "cargo test", "go test", "mvn test", "gradle test", "dart test"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _digest(value: str, size: int = 16) -> str:
    return hashlib.blake2b(value.encode(errors="replace"), digest_size=16).hexdigest()[:size]


def _clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 24] + "\n[...cycle context clipped...]"


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _path(value: str) -> str:
    return value.strip().replace("\\", "/").lstrip("./")


@dataclass(frozen=True, slots=True)
class EditIntent:
    task_id: str
    request: str
    operation: str
    files: tuple[str, ...]
    symbols: tuple[str, ...]
    constraints: tuple[str, ...]
    acceptance: tuple[str, ...]
    risks: tuple[str, ...]
    queries: tuple[str, ...]


class EditIntentCompiler:
    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def is_edit_request(request: str) -> bool:
        return bool(set(TOKEN_RE.findall(request.lower())) & EDIT_WORDS)

    def compile(self, request: str, paths: list[str] | None = None, acceptance: list[str] | None = None) -> EditIntent:
        lowered = request.lower()
        words = set(TOKEN_RE.findall(lowered))
        operation = "debug" if words & DEBUG_WORDS else "edit"
        files = [_path(match.group(0)) for match in FILE_RE.finditer(request)]
        files.extend(_path(value) for value in paths or [])
        symbols = [token for token in TOKEN_RE.findall(request) if "_" in token or any(c.isupper() for c in token[1:])]
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", request) if part.strip()]
        constraints = [s for s in sentences if any(mark in s.lower() for mark in ("must", "preserve", "without", "only", "do not", "don't", "keep"))]
        accepted = [s for s in sentences if any(mark in s.lower() for mark in ("test", "verify", "ensure", "expect", "should"))]
        accepted.extend(acceptance or [])
        risks = [name for name, hints in RISK_HINTS.items() if any(hint in lowered for hint in hints)]
        symbol_text = " ".join(_unique(symbols)[:12]) or request
        file_text = " ".join(_unique(files)[:8]) or "affected files"
        queries = [
            request,
            f"{symbol_text} definitions references callers imports control flow data flow",
            f"{file_text} tests assertions fixtures acceptance validation",
            f"{symbol_text} {' '.join(risks)} invariants compatibility failure modes",
        ]
        if operation == "debug":
            queries.append(f"{request} regression traceback historical failure bug zone")
        return EditIntent(
            _digest(request, 12), request, operation, _unique(files), _unique(symbols)[:20],
            _unique(constraints)[:12], _unique(accepted)[:12], _unique(risks), _unique(queries),
        )


@dataclass(slots=True)
class CycleEvent:
    sequence: int
    kind: str
    summary: str
    ok: bool
    digest: str


@dataclass(slots=True)
class CycleState:
    intent: EditIntent
    phase: str = "localize"
    cycle: int = 0
    revision: int = 0
    context: str = ""
    context_digest: str = ""
    grounded_paths: set[str] = field(default_factory=set)
    exact_reads: set[str] = field(default_factory=set)
    changed_paths: list[str] = field(default_factory=list)
    validators: list[str] = field(default_factory=list)
    latest_error: str = ""
    failures: int = 0
    validated: bool = False
    active: bool = True
    repeats: Counter[str] = field(default_factory=Counter)
    ledger: list[CycleEvent] = field(default_factory=list)


class EditCycleOrchestrator:
    """Compile, retrieve, gate, verify, and re-prompt an edit task."""

    def __init__(self, settings: Any, root: Path, perspective: Any, memory: Any | None = None) -> None:
        self.settings, self.root, self.perspective = settings, root.resolve(), perspective
        self.compiler = EditIntentCompiler(self.root)
        self.memory = memory
        self.state: CycleState | None = None
        self.max_queries = _env_int("HUMOID_EDIT_CYCLE_MAX_QUERIES", 5)
        self.search_limit = _env_int("HUMOID_EDIT_CYCLE_SEARCH_LIMIT", 10)
        self.context_limit = _env_int("HUMOID_EDIT_CYCLE_CONTEXT_MAX_CHARS", 18000, 2000)
        self.prompt_limit = _env_int("HUMOID_EDIT_CYCLE_PROMPT_MAX_CHARS", 24000, 3000)
        self.repeat_limit = _env_int("HUMOID_EDIT_CYCLE_REPEAT_LIMIT", 3)

    @property
    def active(self) -> bool:
        return bool(self.state and self.state.active)

    async def begin(self, request: str, paths: list[str] | None = None, acceptance: list[str] | None = None, force: bool = False) -> str:
        if self.active and self.state and self.state.intent.request == request.strip() and not force:
            return self.prompt("cycle already active")
        intent = self.compiler.compile(request.strip(), paths, acceptance)
        await self.perspective.build(intent.request, paths or None, force=force or bool(getattr(self.perspective, "chunks", None)))
        packets = []
        budget = max(1800, self.context_limit // min(self.max_queries, len(intent.queries)))
        for query in intent.queries[: self.max_queries]:
            packets.append(await self.perspective.search(query, limit=self.search_limit, max_chars=budget))
        self.state = CycleState(intent=intent)
        self._set_context(self._merge(packets))
        self._event("start", f"compiled {intent.operation}; queries={len(packets)}", True)
        return self.prompt("initialized")

    async def refresh(self, query: str | None = None, reason: str = "manual refresh") -> str:
        if not self.state:
            return "ERROR: no active edit cycle"
        packet = await self.perspective.search(query or self._repair_query(), limit=self.search_limit, max_chars=self.context_limit)
        self._set_context(self._merge([self.state.context, packet]))
        self._event("refresh", reason, True)
        return self.prompt(reason)

    def before_tool(self, name: str, args: dict[str, Any]) -> str | None:
        if name in {"start_edit_cycle", "refresh_edit_cycle", "edit_cycle_status", "finish_edit_cycle"}:
            return None
        if not self.active:
            if name in {"write_file", "apply_patch"}:
                return "ERROR: file edits require start_edit_cycle with the original user request"
            return None
        state = self.state
        assert state is not None
        signature = _digest(f"{name}:{json.dumps(args, sort_keys=True, default=str)}", 20)
        state.repeats[signature] += 1
        if state.repeats[signature] > self.repeat_limit:
            state.phase = "blocked"
            return "ERROR: repeated identical action blocked; change the hypothesis or refresh context"
        if name in {"write_file", "apply_patch"}:
            path = _path(str(args.get("path", "")))
            target = self.root / path
            grounded = path in state.grounded_paths or (not target.exists() and any(str(Path(path).parent).replace("\\", "/") in item for item in state.grounded_paths))
            if not grounded:
                return f"ERROR: {path} is not grounded by repository evidence"
            if target.exists() and path not in state.exact_reads:
                return f"ERROR: read_file or read_file_range must inspect {path} before editing"
            if state.phase not in {"inspect", "edit", "repair"}:
                return f"ERROR: edits are not allowed during phase {state.phase}"
        return None

    async def after_tool(self, name: str, args: dict[str, Any], output: str, ok: bool) -> str:
        if not self.active:
            return output
        state = self.state
        assert state is not None
        state.cycle += 1
        state.grounded_paths.update(self._paths(output))
        self._event(name, _clip(output.replace("\n", " "), 320), ok)
        if name in {"search_project_perspective", "expand_project_perspective"} and ok:
            state.phase = "inspect"
        elif name in {"read_file", "read_file_range"} and ok:
            path = _path(str(args.get("path", "")))
            state.exact_reads.add(path); state.grounded_paths.add(path); state.phase = "edit"
        elif name in {"write_file", "apply_patch"} and ok:
            path = _path(str(args.get("path", "")))
            if path not in state.changed_paths: state.changed_paths.append(path)
            state.phase = "verify"; state.validated = False
        elif name == "undo_file_change" and ok:
            state.phase = "inspect"; state.validated = False
        elif name == "run_command" and self._validator(str(args.get("command", ""))):
            command = str(args.get("command", "")); state.validators.append(command)
            if ok:
                state.validated = True; state.phase = "conclude"; await self._remember(True, command, output)
            else:
                state.failures += 1; state.latest_error = _clip(output, 6000); state.phase = "repair"
                await self._remember(False, command, output)
                await self.refresh(self._repair_query(), "validator failure differential retrieval")
        elif not ok:
            state.failures += 1; state.latest_error = _clip(output, 6000); state.phase = "repair"
        return f"{output}\n\n{self.prompt(f'processed {name}')}"

    async def finish(self, summary: str = "") -> str:
        if not self.state:
            return "No active edit cycle."
        self.state.active = False; self.state.phase = "conclude"
        self._event("finish", summary or "closed", True)
        return f"Edit cycle {self.state.intent.task_id} closed; validated={self.state.validated}; changed={','.join(self.state.changed_paths) or 'none'}"

    def status_json(self) -> str:
        if not self.state:
            return json.dumps({"active": False}, indent=2)
        state = self.state
        return json.dumps({
            "active": state.active, "phase": state.phase, "cycle": state.cycle,
            "context_revision": state.revision, "context_digest": state.context_digest,
            "task_id": state.intent.task_id, "operation": state.intent.operation,
            "grounded_paths": sorted(state.grounded_paths), "exact_reads": sorted(state.exact_reads),
            "changed_paths": state.changed_paths, "validators": state.validators,
            "validated": state.validated, "failures": state.failures, "latest_error": state.latest_error,
            "ledger": [event.__dict__ for event in state.ledger[-20:]],
        }, indent=2)

    def prompt(self, reason: str) -> str:
        if not self.state:
            return "[EDIT CYCLE PROMPT]\nNo active cycle."
        state = self.state
        cycle_id = _digest(f"{state.intent.task_id}:{state.cycle}:{state.phase}:{state.revision}:{state.ledger[-1].digest if state.ledger else ''}", 18)
        contracts = {
            "localize": "Do not edit. Select the minimum connected files and symbols; expand only a concrete missing edge, then read the exact target.",
            "inspect": "Do not edit yet. Read the exact implementation and closest test/config dependency; identify invariants and the smallest patch.",
            "edit": "Make one coherent minimal patch in grounded, exact-read files. Prefer digest-checked apply_patch for large existing files.",
            "verify": "Do not speculate. Run the narrowest meaningful validator first; broaden only when affordable.",
            "repair": "Revise the root-cause hypothesis from the validator error and differential context; make one focused repair before re-testing.",
            "conclude": "Stop editing. Close the cycle and report changed files, observed validation, and unresolved risks.",
            "blocked": "Stop repeating the action. Inspect status or refresh with a narrower query.",
        }
        context = state.context if state.phase not in {"verify", "conclude"} else f"paths={','.join(sorted(state.grounded_paths))}; nodes={','.join(NODE_RE.findall(state.context))}"
        ledger = "\n".join(f"- {e.sequence}:{e.kind}:{'ok' if e.ok else 'error'}:{e.digest} {e.summary}" for e in state.ledger[-8:]) or "- none"
        return _clip(f"""[EDIT CYCLE PROMPT v2]
cycle_id={cycle_id}
task_id={state.intent.task_id}
phase={state.phase}
reason={reason}
context_revision={state.revision}
context_digest={state.context_digest}

ORIGINAL REQUEST
{state.intent.request}

COMPILED INTENT
operation={state.intent.operation}
files={','.join(state.intent.files) or 'none'}
symbols={','.join(state.intent.symbols) or 'none'}
constraints={json.dumps(state.intent.constraints)}
acceptance={json.dumps(state.intent.acceptance)}
risks={','.join(state.intent.risks) or 'none'}

CURRENT STATE
grounded={','.join(sorted(state.grounded_paths)) or 'none'}
exact_reads={','.join(sorted(state.exact_reads)) or 'none'}
changed={','.join(state.changed_paths) or 'none'}
validators={','.join(state.validators) or 'none'}
validated={state.validated}
latest_error={state.latest_error or 'none'}

EVIDENCE LEDGER
{ledger}

PHASE CONTRACT
{contracts[state.phase]}

CONTEXT SNAPSHOT
{context}

RULES
- This prompt governs only the next model round.
- Do not invent files, symbols, tests, or validation.
- Existing files require grounded evidence and an exact read before editing.
- A failed validator starts a new repair context; do not repeat the same patch blindly.
- Keep calls literal and small for local models.
[/EDIT CYCLE PROMPT]""", self.prompt_limit)

    def _set_context(self, packet: str) -> None:
        assert self.state is not None
        self.state.context = _clip(packet, self.context_limit)
        self.state.revision += 1
        self.state.context_digest = _digest(self.state.context, 20)
        self.state.grounded_paths.update(self._paths(packet))

    def _merge(self, packets: list[str]) -> str:
        nodes: dict[str, str] = {}; edges: list[str] = []; fallback: list[str] = []
        for packet in packets:
            matches = list(NODE_RE.finditer(packet))
            edges.extend(line for line in packet.splitlines() if line.startswith("- ") and "-->" in line)
            if not matches: fallback.append(packet); continue
            for index, match in enumerate(matches):
                block = packet[match.start(): matches[index + 1].start() if index + 1 < len(matches) else len(packet)]
                block = block.split("\n\n[CONTEXT ACCORDION", 1)[0]
                nodes.setdefault(match.group(1), block.strip())
        parts = [f"[EDIT CYCLE RETRIEVAL] queries={len(packets)} nodes={len(nodes)}"]
        if edges: parts.append("\n[BEHAVIOR EDGES]\n" + "\n".join(dict.fromkeys(edges)))
        for block in nodes.values():
            if len("".join(parts)) + len(block) > self.context_limit: break
            parts.append("\n\n" + block)
        if not nodes: parts.append("\n\n" + _clip("\n\n".join(fallback), self.context_limit - len("".join(parts))))
        return _clip("".join(parts), self.context_limit)

    @staticmethod
    def _paths(value: str) -> tuple[str, ...]:
        values = [_path(match.group("path")) for match in CITATION_RE.finditer(value)]
        values.extend(_path(match.group(0)) for match in FILE_RE.finditer(value))
        return _unique(values)

    def _event(self, kind: str, summary: str, ok: bool) -> None:
        assert self.state is not None
        number = len(self.state.ledger) + 1
        self.state.ledger.append(CycleEvent(number, kind, summary, ok, _digest(f"{number}:{kind}:{summary}:{ok}", 14)))

    def _repair_query(self) -> str:
        assert self.state is not None
        return f"{self.state.intent.request}\nchanged={' '.join(self.state.changed_paths)}\nerror={self.state.latest_error}\ndefinitions callers tests configuration regression root cause"

    @staticmethod
    def _validator(command: str) -> bool:
        lowered = " ".join(command.lower().split())
        return any(value in lowered for value in VALIDATORS) or any(value in lowered for value in (" lint", " typecheck", " test", " check"))

    async def _remember(self, success: bool, command: str, output: str) -> None:
        if self.memory is None or not self.state:
            return
        state = self.state
        try:
            await self.memory.add(
                f"CODE {'FIX VERIFIED' if success else 'FAILURE MEMORY'}\nrequest={state.intent.request}\ncommand={command}\nfiles={','.join(state.changed_paths)}\nresult={_clip(output, 4000)}",
                memory_tier="semantic" if success else "episodic",
                channel="code_validation" if success else "code_failure",
                task_id=state.intent.task_id,
                validation_status="verified" if success else "observed",
                metadata={"files": state.changed_paths, "command": command, "context_digest": state.context_digest},
            )
        except Exception:
            pass
