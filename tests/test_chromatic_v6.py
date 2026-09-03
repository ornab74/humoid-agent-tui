import json
import sqlite3

import pytest

from humoid_tui.chromatic_system import build_hierarchy_surface, encode_change_delta
from humoid_tui.config import Settings
from humoid_tui.maintenance import semantic_change_fingerprint
from humoid_tui.memory import MemoryRouter
from humoid_tui.rhomgaic_memory import ChromaticCodeEmbedding, RhomgaicMemory


def test_chromatic_hierarchy_builds_project_to_behavior_resolution_pyramid():
    text = (
        "debug src/humoid_tui/memory/cache.py CacheIndex regression callers tests "
        "```python\nfrom humoid_tui.memory import cache\n"
        "def refresh_cache():\n    cache.invalidate()\n```"
    )
    encoded = ChromaticCodeEmbedding.encode(text)
    hierarchy = build_hierarchy_surface(
        text=text,
        action=encoded.action,
        language=encoded.language,
        files=encoded.files,
        symbols=encoded.symbols,
        ast_terms=encoded.ast_terms,
    )
    surface = encoded.surface(text)

    assert {level for level, _ in hierarchy.resolution_keys} == {"l0", "l1", "l2", "l3"}
    assert len({key for _, key in hierarchy.resolution_keys}) >= 3
    assert any(scope.startswith("src/humoid_tui") for scope in hierarchy.path_scopes)
    assert "calls" in hierarchy.behaviors
    assert "regression" in hierarchy.behaviors
    assert surface["hierarchy"]["resolution_keys"]["l3"] == dict(hierarchy.resolution_keys)["l3"]


def test_chromatic_change_delta_encodes_structural_risk_without_source():
    before = "def public_api(value):\n    return value\n"
    after = (
        "import decimal\n\n"
        "def public_api(value, *, scale=1):\n"
        "    if value is None:\n"
        "        raise ValueError('value')\n"
        "    return decimal.Decimal(value) * scale\n"
    )
    fingerprint = semantic_change_fingerprint("src/api.py", before, after).to_dict()
    delta = encode_change_delta(fingerprint)
    payload = delta.to_dict()

    assert len(delta.address) == 16
    assert payload["public_api_change"] is True
    assert payload["parse_ok_after"] is True
    assert payload["risk_bucket"] > 0
    assert payload["channels"]["signature_changes"] > 0
    assert "source" not in payload


@pytest.mark.asyncio
async def test_rhomgaic_multiresolution_candidate_collapse_exposes_trace(tmp_path):
    memory = RhomgaicMemory(tmp_path / "rhomgaic.sqlite3", candidate_limit=128)
    await memory.initialize()
    await memory.add(
        "target",
        "debug src/payments/pricing.py calculate_total regression callers and tests",
        validation_status="verified",
    )
    for index in range(140):
        await memory.add(
            f"noise-{index}",
            f"updated docs/section_{index}.md formatting and prose",
            validation_status="observed",
        )

    hits = await memory.search(
        "debug src/payments/pricing.py calculate_total regression", limit=4
    )
    stats = await memory.stats()
    trace = stats["last_chromatic_trace"]

    assert hits and hits[0].memory_id == "target"
    assert trace["universe"] == 141
    assert trace["selected_candidates"] < trace["universe"]
    assert trace["candidates_avoided"] > 0
    assert trace["stage_matches"]["resolution_l2"] >= 1
    assert stats["chromatic_hierarchy_postings"] > 0
    assert stats["schema_version"] == 6


@pytest.mark.asyncio
async def test_procedure_memory_persists_chromatic_change_deltas(tmp_path):
    memory = RhomgaicMemory(tmp_path / "procedures.sqlite3")
    await memory.initialize()
    fingerprint = semantic_change_fingerprint(
        "src/core.py",
        "def compute(x):\n    return x\n",
        "def compute(x, *, scale=1):\n    return x * scale\n",
    ).to_dict()
    await memory.add(
        "procedure-source",
        "CODING PROCEDURE MEMORY compute API repair",
        memory_tier="procedural",
        validation_status="verified",
        metadata={
            "coding_trajectory": {
                "request": "refactor src/core.py compute without breaking callers",
                "operation": "edit",
                "changed_paths": ["src/core.py"],
                "validators": ["python -m pytest tests/test_core.py"],
                "validated": True,
                "change_fingerprints": {"src/core.py": fingerprint},
                "steps": [{"kind": "apply_patch"}, {"kind": "run_command"}],
            }
        },
    )

    with sqlite3.connect(tmp_path / "procedures.sqlite3") as db:
        row = db.execute("SELECT trajectory,text FROM rhomgaic_procedures").fetchone()
    trajectory = json.loads(row[0])

    assert trajectory["chromatic_change_deltas"]
    assert len(trajectory["chromatic_change_deltas"][0]["address"]) == 32
    assert "chromatic_changes=" in row[1]


@pytest.mark.asyncio
async def test_router_exposes_chromatic_surface_packet(tmp_path):
    router = MemoryRouter(Settings(
        humoid_memory_backend="sqlite",
        humoid_memory_db=tmp_path / "provider.sqlite3",
        humoid_rhomgaic_db=tmp_path / "rhomgaic.sqlite3",
    ))
    await router.initialize()
    await router.add(
        "Fixed src/cache.py CacheIndex invalidation and linked regression tests",
        validation_status="verified",
    )

    packet = await router.chromatic_surface_packet(
        "debug src/cache.py CacheIndex invalidation regression", limit=4
    )

    assert "CHROMATIC CODE SURFACE" in packet
    assert "resolution_keys=" in packet
    assert "candidate_collapse=" in packet
    assert "src/cache.py" in packet
    await router.close()

@pytest.mark.asyncio
async def test_chromatic_change_history_transfers_validation_strategy_by_change_shape(tmp_path):
    memory = RhomgaicMemory(tmp_path / "change-history.sqlite3")
    await memory.initialize()
    api_change = semantic_change_fingerprint(
        "src/payments/api.py",
        "def charge(amount):\n    return amount\n",
        "def charge(amount, *, currency='USD'):\n    return amount\n",
    ).to_dict()
    internal_change = semantic_change_fingerprint(
        "src/cache/internal.py",
        "def _normalize(value):\n    return value\n",
        "def _normalize(value):\n    return value.strip()\n",
    ).to_dict()
    await memory.add(
        "api-procedure",
        "procedure for public signature migration",
        memory_tier="procedural",
        validation_status="verified",
        metadata={"coding_trajectory": {
            "request": "extend payments public API",
            "operation": "edit",
            "changed_paths": ["src/payments/api.py"],
            "validators": ["python -m pytest tests/test_payments.py", "python -m pytest"],
            "validated": True,
            "change_fingerprints": {"src/payments/api.py": api_change},
            "steps": [{"kind": "apply_patch"}, {"kind": "run_command"}],
        }},
    )
    await memory.add(
        "internal-procedure",
        "procedure for tiny private cleanup",
        memory_tier="procedural",
        validation_status="verified",
        metadata={"coding_trajectory": {
            "request": "clean cache private helper",
            "operation": "edit",
            "changed_paths": ["src/cache/internal.py"],
            "validators": ["python -m pytest tests/test_cache.py"],
            "validated": True,
            "change_fingerprints": {"src/cache/internal.py": internal_change},
            "steps": [{"kind": "apply_patch"}, {"kind": "run_command"}],
        }},
    )
    current = semantic_change_fingerprint(
        "src/orders/api.py",
        "def submit(order):\n    return order\n",
        "def submit(order, *, idempotency_key=None):\n    return order\n",
    ).to_dict()

    procedures = await memory.search_change_procedures(
        {"src/orders/api.py": current}, limit=4
    )

    assert procedures
    assert procedures[0]["procedure_id"] != procedures[-1]["procedure_id"]
    assert procedures[0]["change_similarity"] >= procedures[-1]["change_similarity"]
    assert "python -m pytest" in " ".join(procedures[0]["validators"])


@pytest.mark.asyncio
async def test_progressive_trace_reports_chromatic_resolution_and_candidate_collapse(tmp_path):
    router = MemoryRouter(Settings(
        humoid_memory_backend="sqlite",
        humoid_memory_db=tmp_path / "provider.sqlite3",
        humoid_rhomgaic_db=tmp_path / "rhomgaic.sqlite3",
        humoid_rag_context_cache=False,
    ))
    await router.initialize()
    await router.add(
        "debug src/cache.py CacheIndex invalidation regression callers tests",
        validation_status="verified",
    )
    await router.context_packet(
        "debug src/cache.py CacheIndex invalidation regression", depth="working"
    )

    plan = router.last_trace["retrieval_plan"]
    collapse = router.last_trace["chromatic_candidate_trace"]
    assert plan["chromatic_resolution"] in {"l2-file", "l3-symbol-behavior"}
    assert collapse["selected_candidates"] <= collapse["candidate_limit"]
    assert "specific" in collapse["opened_stages"]
    await router.close()

def test_maintenance_runtime_exposes_chromatic_diagnostic_tools():
    from humoid_tui.maintenance_runtime import _schemas

    names = {item["function"]["name"] for item in _schemas()}
    assert "inspect_chromatic_code_surface" in names
    assert "recall_chromatic_change_history" in names
