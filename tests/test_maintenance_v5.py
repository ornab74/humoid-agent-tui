import json

import pytest

from humoid_tui.config import Settings
from humoid_tui.maintenance import (
    MaintenanceHealthAnalyzer,
    MaintenancePlanner,
    semantic_change_fingerprint,
)
from humoid_tui.memory import MemoryRouter
from humoid_tui.rhomgaic_memory import RhomgaicMemory


def test_semantic_change_fingerprint_detects_api_dependency_and_exception_changes():
    before = '''\nimport math\n\ndef compute_total(items):\n    return sum(items)\n'''
    after = '''\nimport decimal\n\ndef compute_total(items, *, tax=0):\n    if not items:\n        raise ValueError("items required")\n    return sum(items) + tax\n'''

    fingerprint = semantic_change_fingerprint("src/core.py", before, after)

    assert fingerprint.parse_ok_after is True
    assert fingerprint.public_api_change is True
    assert "compute_total" in fingerprint.signature_changes
    assert "decimal" in fingerprint.added_imports
    assert "math" in fingerprint.removed_imports
    assert any(value.endswith("ValueError") for value in fingerprint.added_raises)
    assert fingerprint.risk_score > 1.0
    assert fingerprint.fingerprint


@pytest.mark.asyncio
async def test_rhomgaic_procedural_memory_aggregates_reusable_success_shape(tmp_path):
    memory = RhomgaicMemory(tmp_path / "procedures.sqlite3")
    await memory.initialize()
    trajectory = {
        "request": "debug src/cache.py CacheIndex invalidation regression",
        "operation": "debug",
        "changed_paths": ["src/cache.py", "tests/test_cache.py"],
        "symbols": ["CacheIndex"],
        "risks": ["storage"],
        "validators": ["python -m pytest tests/test_cache.py"],
        "validated": True,
        "steps": [
            {"kind": "search_project_perspective"},
            {"kind": "read_file_range"},
            {"kind": "apply_patch"},
            {"kind": "run_command"},
        ],
    }
    await memory.add(
        "procedure-memory-1",
        "CODING PROCEDURE MEMORY cache repair",
        memory_tier="procedural",
        validation_status="verified",
        metadata={"coding_trajectory": trajectory},
    )
    second = dict(trajectory)
    second["request"] = "debug src/session_cache.py SessionCache stale invalidation"
    second["changed_paths"] = ["src/session_cache.py", "tests/test_session_cache.py"]
    second["symbols"] = ["SessionCache"]
    await memory.add(
        "procedure-memory-2",
        "CODING PROCEDURE MEMORY session cache repair",
        memory_tier="procedural",
        validation_status="verified",
        metadata={"coding_trajectory": second},
    )

    procedures = await memory.search_procedures(
        "debug src/cache.py CacheIndex invalidation and run pytest", limit=4
    )
    stats = await memory.stats()

    assert procedures
    assert procedures[0]["support_count"] == 2
    assert procedures[0]["success_rate"] == 1.0
    assert "apply_patch" in procedures[0]["step_kinds"]
    assert stats["coding_procedures"] == 1


@pytest.mark.asyncio
async def test_router_persists_procedure_and_exposes_it_in_coding_context(tmp_path):
    router = MemoryRouter(Settings(
        humoid_memory_backend="sqlite",
        humoid_memory_db=tmp_path / "provider.sqlite3",
        humoid_rhomgaic_db=tmp_path / "rhomgaic.sqlite3",
        humoid_memory_packet_max_chars=6000,
    ))
    await router.initialize()
    await router.add(
        "Current history: CacheIndex lives in src/cache.py",
        validation_status="observed",
    )
    procedure_id = await router.record_coding_trajectory({
        "task_id": "cache-regression",
        "request": "fix src/cache.py CacheIndex invalidation regression",
        "operation": "debug",
        "changed_paths": ["src/cache.py", "tests/test_cache.py"],
        "symbols": ["CacheIndex"],
        "risks": ["storage"],
        "validators": ["python -m pytest tests/test_cache.py"],
        "validated": True,
        "failures": 0,
        "summary": "localized caller, patched invalidation, ran focused regression test",
        "steps": [
            {"kind": "search_project_perspective"},
            {"kind": "read_file_range"},
            {"kind": "apply_patch"},
            {"kind": "run_command"},
        ],
    })

    packet = await router.context_packet(
        "debug src/cache.py CacheIndex invalidation", depth="working"
    )
    durable = await router.list_memories(limit=50)

    assert procedure_id in {item.memory_id for item in durable}
    assert any(item.memory_tier == "procedural" for item in durable)
    assert "PROCEDURAL MAINTENANCE SURFACE" in packet
    assert "apply_patch" in packet
    assert router.last_trace["procedural_playbooks"] >= 1
    await router.close()


@pytest.mark.asyncio
async def test_maintenance_planner_fuses_current_impact_with_historical_playbook(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "core.py").write_text(
        "def compute_total(items):\n    return sum(items)\n", encoding="utf-8"
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

    router = MemoryRouter(Settings(
        humoid_workspace=root,
        humoid_memory_backend="sqlite",
        humoid_memory_db=tmp_path / "provider.sqlite3",
        humoid_rhomgaic_db=tmp_path / "rhomgaic.sqlite3",
    ))
    await router.initialize()
    await router.record_coding_trajectory({
        "request": "debug src/core.py compute_total regression",
        "operation": "debug",
        "changed_paths": ["src/core.py", "tests/test_core.py"],
        "symbols": ["compute_total"],
        "validators": ["python -m pytest tests/test_core.py"],
        "validated": True,
        "steps": [
            {"kind": "analyze_change_impact"},
            {"kind": "read_file_range"},
            {"kind": "apply_patch"},
            {"kind": "run_command"},
        ],
    })

    plan = await MaintenancePlanner(root).plan(
        "debug src/core.py compute_total regression", router, max_chars=12000
    )

    assert "MAINTENANCE EXECUTION CONTRACT" in plan
    assert "src/core.py" in plan
    assert "src/service.py" in plan
    assert "HISTORICAL PROCEDURAL HINTS" in plan
    assert "apply_patch" in plan
    assert "python -m pytest" in plan
    await router.close()


def test_maintenance_health_report_marks_pressure_without_calling_it_quality(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = ["def overloaded(value):"]
    source.extend(f"    if value == {index}: return {index}" for index in range(90))
    source.append("    # TODO: split responsibilities")
    source.append("    try: return value")
    source.append("    except Exception: return None")
    (root / "module.py").write_text("\n".join(source) + "\n", encoding="utf-8")

    analyzer = MaintenanceHealthAnalyzer(root)
    report = analyzer.analyze(max_findings=20)
    rendered = analyzer.render(report)

    kinds = {item["kind"] for item in report["findings"]}
    assert "large-symbol" in kinds
    assert "deferred-work" in kinds
    assert "broad-exception-surface" in kinds
    assert "not a quality grade" in rendered.lower()
