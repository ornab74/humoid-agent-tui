from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from humoid_tui.edit_cycle import EditCycleOrchestrator, EditIntentCompiler
from humoid_tui.edit_cycle_runtime import _apply_patch, _read_range, _run_validation


class FakePerspective:
    def __init__(self) -> None:
        self.chunks = {}
        self.queries: list[str] = []

    async def build(self, objective, paths=None, force=False):
        self.objective = objective
        return "built"

    async def search(self, query, limit=10, max_chars=18000):
        self.queries.append(query)
        return (
            "[PROJECT PERSPECTIVE: minimal connected behavior subgraph]\n"
            "[PERSPECTIVE NODE auth-node | src/auth/session.py:1-20 | anchor=true]\n"
            "definitions=SessionManager, refresh\n"
            "class SessionManager:\n    def refresh(self, token): return token\n"
        )


class FakeMemory:
    def __init__(self) -> None:
        self.records = []

    async def add(self, text, **metadata):
        self.records.append((text, metadata))
        return "memory-1"


@dataclass
class FakeSettings:
    humoid_allow_shell: bool = True
    humoid_shell_timeout_seconds: int = 10


class FakeRegistry:
    def __init__(self, root: Path, shell: bool = True) -> None:
        self.root = root
        self.s = FakeSettings(humoid_allow_shell=shell)

    def _path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        assert path == self.root or self.root in path.parents
        return path

    async def _write_file(self, relative: str, content: str) -> str:
        path = self._path(relative)
        path.write_text(content)
        return f"wrote {relative}"


def test_intent_compiler_builds_code_specific_query_lattice(tmp_path):
    intent = EditIntentCompiler(tmp_path).compile(
        "Fix SessionManager.refresh in src/auth/session.py. Preserve concurrent retry behavior and verify tests."
    )

    assert intent.operation == "debug"
    assert "src/auth/session.py" in intent.files
    assert "SessionManager" in intent.symbols
    assert "auth" in intent.risks
    assert len(intent.queries) == 5
    assert any("callers imports control flow" in query for query in intent.queries)
    assert any("historical failure bug zone" in query for query in intent.queries)


async def test_cycle_gates_write_until_graph_grounding_and_exact_read(tmp_path):
    target = tmp_path / "src" / "auth" / "session.py"
    target.parent.mkdir(parents=True)
    target.write_text("class SessionManager:\n    pass\n")
    cycle = EditCycleOrchestrator(FakeSettings(), tmp_path, FakePerspective())

    prompt = await cycle.begin("Fix SessionManager in src/auth/session.py")
    assert "phase=localize" in prompt
    assert "src/auth/session.py" in cycle.state.grounded_paths

    blocked = cycle.before_tool("write_file", {"path": "src/auth/session.py"})
    assert "must inspect" in blocked

    read_output = "[FILE RANGE src/auth/session.py:1-2 sha256=abc]\n1 | class SessionManager:"
    await cycle.after_tool(
        "read_file_range",
        {"path": "src/auth/session.py", "start_line": 1, "end_line": 2},
        read_output,
        True,
    )
    assert cycle.state.phase == "edit"
    assert cycle.before_tool("write_file", {"path": "src/auth/session.py"}) is None


async def test_failed_validator_creates_memory_and_new_repair_context(tmp_path):
    memory = FakeMemory()
    perspective = FakePerspective()
    cycle = EditCycleOrchestrator(FakeSettings(), tmp_path, perspective, memory=memory)
    await cycle.begin("Fix token refresh regression in src/auth/session.py")
    await cycle.after_tool(
        "read_file",
        {"path": "src/auth/session.py"},
        "src/auth/session.py:1-20",
        True,
    )
    await cycle.after_tool(
        "write_file",
        {"path": "src/auth/session.py"},
        "wrote src/auth/session.py",
        True,
    )

    result = await cycle.after_tool(
        "run_command",
        {"command": "pytest -q tests/test_session.py"},
        "ERROR: assertion failed in SessionManager.refresh",
        False,
    )

    assert cycle.state.phase == "repair"
    assert cycle.state.failures == 1
    assert cycle.state.revision >= 2
    assert memory.records
    assert "CODE FAILURE MEMORY" in memory.records[0][0]
    assert "phase=repair" in result
    assert any("root cause" in query for query in perspective.queries)


async def test_range_read_and_digest_checked_patch_preserve_untouched_text(tmp_path):
    path = tmp_path / "large.py"
    source = "header\nold behavior\nfooter\n"
    path.write_text(source)
    registry = FakeRegistry(tmp_path)

    packet = _read_range(registry, {"path": "large.py", "start_line": 2, "end_line": 2})
    digest = hashlib.sha256(source.encode()).hexdigest()
    assert digest in packet
    assert "old behavior" in packet

    stale = await _apply_patch(registry, {
        "path": "large.py",
        "expected_sha256": "deadbeef",
        "replacements": [{"old": "old behavior", "new": "new behavior"}],
    })
    assert stale.startswith("ERROR: stale patch digest")

    applied = await _apply_patch(registry, {
        "path": "large.py",
        "expected_sha256": digest,
        "replacements": [{"old": "old behavior", "new": "new behavior"}],
    })
    assert "replacements=1" in applied
    assert path.read_text() == "header\nnew behavior\nfooter\n"


async def test_run_validation_preserves_real_exit_status(tmp_path):
    registry = FakeRegistry(tmp_path)

    passed, ok = await _run_validation(registry, {"command": "python -c \"print('ok')\""})
    assert ok is True
    assert "validator passed" in passed

    failed, ok = await _run_validation(registry, {"command": "python -c \"raise SystemExit(7)\""})
    assert ok is False
    assert "exited 7" in failed
