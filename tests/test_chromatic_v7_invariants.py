from __future__ import annotations

import pytest

from humoid_tui.chromatic_invariants import ChromaticInvariantCompiler
from humoid_tui.chromatic_system import encode_invariant_address
from humoid_tui.edit_cycle import EditCycleOrchestrator
from humoid_tui.maintenance import MaintenancePlanner, semantic_change_fingerprint


class _Perspective:
    def __init__(self) -> None:
        self.chunks = {}

    async def build(self, objective, paths=None, force=False):
        return "built"

    async def search(self, query, limit=10, max_chars=18000):
        return (
            "[PROJECT PERSPECTIVE: minimal connected behavior subgraph]\n"
            "[PERSPECTIVE NODE core | src/core.py:1-20 | anchor=true]\n"
            "definitions=compute_total\n"
            "def compute_total(items): return sum(items)\n"
        )


class _Settings:
    pass


def _repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "core.py").write_text(
        "import math\n\ndef compute_total(items):\n    if not items:\n        raise ValueError('items')\n    return sum(items)\n",
        encoding="utf-8",
    )
    (root / "src" / "service.py").write_text(
        "from src.core import compute_total\n\ndef checkout(items):\n    return compute_total(items)\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_core.py").write_text(
        "from src.core import compute_total\n\ndef test_total():\n    assert compute_total([1, 2]) == 3\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    return root


def test_invariant_address_is_deterministic_and_shape_sensitive():
    left = encode_invariant_address(
        kind="public-signature",
        severity="critical",
        source="baseline",
        path="src/core.py",
        symbol="compute_total",
        statement="preserve public signature",
        operation="debug",
    )
    same = encode_invariant_address(
        kind="public-signature",
        severity="critical",
        source="baseline",
        path="src/core.py",
        symbol="compute_total",
        statement="preserve public signature",
        operation="debug",
    )
    parse = encode_invariant_address(
        kind="parseability",
        severity="critical",
        source="baseline",
        path="src/core.py",
        statement="remain parseable",
        operation="debug",
    )
    assert left.address == same.address
    assert len(left.address) == 16
    assert left.distance(same) == 0
    assert left.distance(parse) > 0


def test_compiler_builds_static_api_dependency_and_validation_obligations(tmp_path):
    root = _repo(tmp_path)
    compiler = ChromaticInvariantCompiler(root)
    contract = compiler.compile(
        "Fix compute_total in src/core.py without breaking callers and verify tests.",
        operation="debug",
        files=["src/core.py"],
        symbols=["compute_total"],
        constraints=["without breaking callers"],
        acceptance=["tests should pass"],
    )
    kinds = {item.kind for item in contract.invariants}
    assert "parseability" in kinds
    assert "public-signature" in kinds
    assert "symbol-presence" in kinds
    assert "dependency-continuity" in kinds
    assert "exception-continuity" in kinds
    assert "validation-obligation" in kinds
    assert "explicit-constraint" in kinds
    assert contract.surface_key
    assert contract.linked_tests
    assert all(len(item.address_hex) == 32 for item in contract.invariants)


def test_proof_passes_static_invariants_but_keeps_explicit_validation_open(tmp_path):
    root = _repo(tmp_path)
    compiler = ChromaticInvariantCompiler(root)
    contract = compiler.compile(
        "Fix compute_total in src/core.py and verify tests.",
        operation="debug",
        files=["src/core.py"],
        symbols=["compute_total"],
        acceptance=["verify tests"],
    )
    open_report = compiler.evaluate(contract, changed_paths=["src/core.py"])
    closed_report = compiler.evaluate(
        contract,
        changed_paths=["src/core.py"],
        validators=["python -m pytest tests/test_core.py"],
        validated=True,
    )
    assert open_report.gate is False
    assert open_report.status == "unverified"
    assert closed_report.gate is True
    assert closed_report.failed == 0
    assert closed_report.proof_score > open_report.proof_score


def test_signature_break_is_a_blocking_invariant_failure(tmp_path):
    root = _repo(tmp_path)
    compiler = ChromaticInvariantCompiler(root)
    contract = compiler.compile(
        "Fix compute_total in src/core.py without breaking callers.",
        operation="debug",
        files=["src/core.py"],
        symbols=["compute_total"],
    )
    before = (root / "src" / "core.py").read_text(encoding="utf-8")
    after = before.replace("def compute_total(items):", "def compute_total(items, tax=0):")
    (root / "src" / "core.py").write_text(after, encoding="utf-8")
    fp = semantic_change_fingerprint("src/core.py", before, after).to_dict()
    report = compiler.evaluate(
        contract,
        change_fingerprints={"src/core.py": fp},
        changed_paths=["src/core.py"],
        validators=["python -m pytest tests/test_core.py"],
        validated=True,
    )
    assert report.gate is False
    assert report.failed >= 1
    assert any(check.kind == "public-signature" and check.status == "fail" for check in report.checks)


@pytest.mark.asyncio
async def test_edit_cycle_compiles_contract_and_blocks_conclusion_on_preservation_break(tmp_path):
    root = _repo(tmp_path)
    cycle = EditCycleOrchestrator(_Settings(), root, _Perspective())
    await cycle.begin(
        "Fix compute_total in src/core.py without breaking callers and verify tests.",
        paths=["src/core.py"],
        acceptance=["tests should pass"],
    )
    assert cycle.state is not None
    assert cycle.state.invariant_contract
    assert "invariant_contract=" in cycle.prompt("test")

    before = (root / "src" / "core.py").read_text(encoding="utf-8")
    after = before.replace("def compute_total(items):", "def compute_total(items, tax=0):")
    (root / "src" / "core.py").write_text(after, encoding="utf-8")
    cycle.state.changed_paths.append("src/core.py")
    cycle.state.change_fingerprints["src/core.py"] = semantic_change_fingerprint(
        "src/core.py", before, after
    ).to_dict()
    cycle.state.validators.append("python -m pytest tests/test_core.py")
    cycle.state.validated = True
    cycle._refresh_invariant_proof()

    result = await cycle.finish("done")
    assert result.startswith("ERROR: edit cycle cannot conclude")
    assert cycle.state.active is True
    assert cycle.state.phase == "repair"


@pytest.mark.asyncio
async def test_maintenance_plan_includes_chromatic_invariant_preflight(tmp_path):
    root = _repo(tmp_path)
    plan = await MaintenancePlanner(root).plan(
        "Fix compute_total in src/core.py without breaking callers and verify tests.",
        memory=None,
        max_chars=16000,
    )
    assert "CHROMATIC INVARIANT PREFLIGHT" in plan
    assert "CHROMATIC INVARIANT CONTRACT" in plan
    assert "public-signature" in plan or "parseability" in plan


def test_runtime_exposes_v7_invariant_tools():
    from humoid_tui.maintenance_runtime import _schemas

    names = {item["function"]["name"] for item in _schemas()}
    assert "compile_chromatic_invariants" in names
    assert "inspect_chromatic_invariant_proof" in names

@pytest.mark.asyncio
async def test_rhomgaic_recalls_procedures_by_invariant_contract_shape(tmp_path):
    from humoid_tui.rhomgaic_memory import RhomgaicMemory

    root = _repo(tmp_path)
    compiler = ChromaticInvariantCompiler(root)
    contract = compiler.compile(
        "Fix compute_total in src/core.py without breaking callers and verify tests.",
        operation="debug",
        files=["src/core.py"],
        symbols=["compute_total"],
        acceptance=["tests should pass"],
    )
    memory = RhomgaicMemory(tmp_path / "rhomgaic.sqlite3")
    await memory.initialize()
    await memory.add(
        "procedure-with-invariant-contract",
        "CODING PROCEDURE MEMORY API-safe repair",
        memory_tier="procedural",
        validation_status="verified",
        metadata={"coding_trajectory": {
            "request": "Fix compute_total without breaking callers",
            "operation": "debug",
            "changed_paths": ["src/core.py", "tests/test_core.py"],
            "symbols": ["compute_total"],
            "validators": ["python -m pytest tests/test_core.py"],
            "validated": True,
            "invariant_contract": contract.to_dict(),
            "invariant_proof": {"status": "pass", "gate": True, "proof_score": 1.0},
            "steps": [{"kind": "apply_patch"}, {"kind": "run_command"}],
        }},
    )

    hits = await memory.search_invariant_procedures(contract.to_dict(), limit=4)
    packet = await memory.invariant_procedure_packet(contract.to_dict(), limit=4)

    assert hits
    assert hits[0]["invariant_similarity"] > 0.95
    assert "public-signature" in hits[0]["matched_kinds"]
    assert "python -m pytest tests/test_core.py" in hits[0]["validators"]
    assert "CHROMATIC INVARIANT-HISTORY SURFACE" in packet


def test_runtime_exposes_invariant_history_tool():
    from humoid_tui.maintenance_runtime import _schemas

    names = {item["function"]["name"] for item in _schemas()}
    assert "recall_chromatic_invariant_history" in names


def test_intentional_signature_change_does_not_emit_preservation_gate(tmp_path):
    root = _repo(tmp_path)
    compiler = ChromaticInvariantCompiler(root)
    contract = compiler.compile(
        "Change signature of compute_total in src/core.py to add an explicit tax argument.",
        operation="edit",
        files=["src/core.py"],
        symbols=["compute_total"],
    )
    assert not any(item.kind == "public-signature" for item in contract.invariants)
    assert any(item.kind == "symbol-presence" for item in contract.invariants)
