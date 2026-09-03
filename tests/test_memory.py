import pytest

from humoid_tui.config import Settings
from humoid_tui.memory import MemoryRouter, SQLiteMemory

@pytest.mark.asyncio
async def test_memory_search(tmp_path):
    m=SQLiteMemory(tmp_path/"m.sqlite")
    await m.initialize()
    await m.add("alpha beta gamma", memory_tier="semantic")
    hits=await m.search("alpha gamma")
    assert hits and hits[0].memory_tier=="semantic"


@pytest.mark.asyncio
async def test_coding_retrieval_boosts_exact_file_and_suppresses_noise(tmp_path):
    memory = SQLiteMemory(tmp_path / "coding.sqlite3")
    await memory.initialize()
    await memory.add(
        "Provider stream failed while touching src/cache.py validation errors",
        validation_status="observed",
    )
    expected = await memory.add(
        "Fixed src/cache.py invalidation and tests pass",
        validation_status="verified",
    )

    hits = await memory.search("edit src/cache.py invalidation", limit=2)

    assert hits[0].memory_id == expected


@pytest.mark.asyncio
async def test_context_packet_is_compact_and_exposes_working_files(tmp_path):
    router = MemoryRouter(Settings(
        humoid_memory_backend="sqlite",
        humoid_memory_db=tmp_path / "packet.sqlite3",
        humoid_memory_packet_max_chars=500,
    ))
    await router.initialize()
    await router.add("Implemented src/agent.py parser " + ("detail " * 200))

    packet = await router.context_packet("src/agent.py parser")

    assert "RETRIEVED WORKING SET" in packet
    assert "src/agent.py" in packet
    assert len(packet) <= 500
    await router.close()

from humoid_tui.rhomgaic_memory import ChromaticCodeEmbedding, RhomgaicMemory


def test_chromatic_code_embedding_is_deterministic_and_action_aware():
    left = ChromaticCodeEmbedding.encode("edit src/cache.py and run pytest")
    right = ChromaticCodeEmbedding.encode("edit src/cache.py and run pytest")
    other = ChromaticCodeEmbedding.encode("read docs/architecture.md")
    assert left.address == right.address
    assert left.vector == right.vector
    assert left.action in {"edit", "test"}
    assert left.distance(right) == 0
    assert left.distance(other) > 0


@pytest.mark.asyncio
async def test_rhomgaic_surface_recalls_exact_code_context(tmp_path):
    memory = RhomgaicMemory(tmp_path / "rhomgaic.sqlite3")
    await memory.initialize()
    await memory.add("a", "implemented src/cache.py invalidation path and CacheIndex refresh", validation_status="verified")
    await memory.add("b", "updated docs/theme.md headings", validation_status="verified")
    hits = await memory.search("fix src/cache.py CacheIndex invalidation", limit=2)
    assert hits
    assert hits[0].memory_id == "a"


@pytest.mark.asyncio
async def test_router_progressive_rag_exposes_stages(tmp_path):
    router = MemoryRouter(Settings(
        humoid_memory_backend="sqlite",
        humoid_memory_db=tmp_path / "provider.sqlite3",
        humoid_rhomgaic_db=tmp_path / "rhomgaic.sqlite3",
        humoid_memory_packet_max_chars=1200,
    ))
    await router.initialize()
    await router.add("Fixed src/agent.py memory planner and tests pass", validation_status="verified")
    await router.add("Investigated src/agent.py provider loop", validation_status="observed")
    packet = await router.context_packet("edit src/agent.py memory planner")
    assert "RAG STAGES:" in packet
    assert "src/agent.py" in packet
    assert "reflex" in packet
    await router.close()

@pytest.mark.asyncio
async def test_rhomgaic_graph_expands_cross_file_symbol_history(tmp_path):
    memory = RhomgaicMemory(tmp_path / "graph.sqlite3")
    await memory.initialize()
    await memory.add(
        "impl",
        "Fixed src/cache.py CacheIndex invalidation ordering",
        validation_status="verified",
        task_id="cache-fix",
    )
    await memory.add(
        "test",
        "Added tests/test_cache.py coverage for CacheIndex invalidation regression",
        validation_status="verified",
        task_id="cache-fix",
    )
    await memory.add(
        "noise",
        "Updated docs/theme.md typography",
        validation_status="verified",
    )

    anchors = await memory.search("edit src/cache.py CacheIndex", limit=1)
    expanded = await memory.expand_graph(anchors, limit=4, hops=2)

    assert anchors and anchors[0].memory_id == "impl"
    assert any(hit.memory_id == "test" for hit in expanded)
    related = next(hit for hit in expanded if hit.memory_id == "test")
    assert related.metadata.get("rhomgaic_graph") is True
    assert related.metadata.get("graph_path")


@pytest.mark.asyncio
async def test_router_update_reindexes_rhomgaic_sidecar(tmp_path):
    router = MemoryRouter(Settings(
        humoid_memory_backend="sqlite",
        humoid_memory_db=tmp_path / "provider.sqlite3",
        humoid_rhomgaic_db=tmp_path / "rhomgaic.sqlite3",
    ))
    await router.initialize()
    memory_id = await router.add(
        "Old src/cache.py LegacyCache behavior",
        validation_status="observed",
    )
    await router.update_memory(
        memory_id,
        text="Fixed src/cache.py CacheIndex invalidation and tests pass",
        validation_status="verified",
    )

    hits = await router.search("CacheIndex invalidation", limit=3)
    assert hits and hits[0].memory_id == memory_id
    assert "CacheIndex" in hits[0].text
    await router.close()


@pytest.mark.asyncio
async def test_agentic_depth_opens_context_incrementally(tmp_path):
    router = MemoryRouter(Settings(
        humoid_memory_backend="sqlite",
        humoid_memory_db=tmp_path / "provider.sqlite3",
        humoid_rhomgaic_db=tmp_path / "rhomgaic.sqlite3",
        humoid_memory_packet_max_chars=6000,
        humoid_rag_reflex_max_chars=1200,
        humoid_rag_working_max_chars=3000,
    ))
    await router.initialize()
    await router.add("Fixed src/cache.py CacheIndex invalidation", task_id="cache", validation_status="verified")
    await router.add("Added tests/test_cache.py CacheIndex regression coverage", task_id="cache", validation_status="verified")
    await router.add("Updated src/cache_metrics.py cache instrumentation", task_id="cache", validation_status="observed")

    reflex = await router.context_packet("fix src/cache.py CacheIndex", depth="reflex")
    reflex_trace = dict(router.last_trace)
    deep = await router.context_packet("fix src/cache.py CacheIndex", depth="subsurface")
    deep_trace = dict(router.last_trace)

    assert "reflex" in reflex
    assert len(reflex) <= 1200
    assert [stage["name"] for stage in reflex_trace["stages"]] == ["reflex"]
    assert [stage["name"] for stage in deep_trace["stages"]] == ["reflex", "working", "subsurface"]
    assert "SUBSURFACE INDEX" in deep
    assert len(deep) >= len(reflex)
    await router.close()


@pytest.mark.asyncio
async def test_memory_tools_can_open_larger_surface(tmp_path):
    from humoid_tui.tools import ToolRegistry

    settings = Settings(
        humoid_workspace=tmp_path / "workspace",
        humoid_memory_backend="sqlite",
        humoid_memory_db=tmp_path / "provider.sqlite3",
        humoid_rhomgaic_db=tmp_path / "rhomgaic.sqlite3",
    )
    router = MemoryRouter(settings)
    await router.initialize()
    await router.add("Fixed src/agent.py progressive memory recall", validation_status="verified")
    tools = ToolRegistry(settings)
    tools.attach_memory(router)

    packet = await tools.execute("recall_memory_context", {
        "query": "src/agent.py memory recall",
        "depth": "working",
        "max_chars": 2500,
    })
    status = await tools.execute("memory_surface_status", {})

    assert "working" in packet
    assert "rhomgaic" in status.lower()
    await tools.close()
    await router.close()


def test_chromatic_python_ast_surface_extracts_structural_memory():
    encoded = ChromaticCodeEmbedding.encode(
        "src/cache.py\n```python\n"
        "import asyncio\n"
        "class CacheIndex:\n"
        "    async def refresh(self):\n"
        "        await asyncio.sleep(0)\n"
        "```"
    )

    surface = set(encoded.ast_terms)
    assert ("ast_class", "cacheindex") in surface
    assert ("ast_def", "refresh") in surface
    assert any(kind == "ast_call" and value.endswith("sleep") for kind, value in surface)
    assert ("ast_import", "asyncio") in surface


@pytest.mark.asyncio
async def test_context_packet_hot_cache_hits_then_invalidates_on_memory_write(tmp_path):
    router = MemoryRouter(Settings(
        humoid_memory_backend="sqlite",
        humoid_memory_db=tmp_path / "provider.sqlite3",
        humoid_rhomgaic_db=tmp_path / "rhomgaic.sqlite3",
        humoid_rag_context_cache=True,
        humoid_rag_context_cache_ttl_seconds=3600,
    ))
    await router.initialize()
    await router.add(
        "Fixed src/cache.py CacheIndex invalidation ordering",
        validation_status="verified",
    )

    first = await router.context_packet("src/cache.py CacheIndex invalidation", depth="working")
    first_trace = dict(router.last_trace)
    second = await router.context_packet("src/cache.py CacheIndex invalidation", depth="working")
    second_trace = dict(router.last_trace)

    assert first == second
    assert first_trace.get("cache_hit") is False
    assert second_trace.get("cache_hit") is True

    await router.add(
        "Added tests/test_cache.py CacheIndex invalidation regression coverage",
        validation_status="verified",
    )
    third = await router.context_packet("src/cache.py CacheIndex invalidation", depth="working")
    third_trace = dict(router.last_trace)
    assert third
    assert third_trace.get("cache_hit") is False
    await router.close()


@pytest.mark.asyncio
async def test_rhomgaic_consolidation_aliases_duplicates_without_deleting_provider(tmp_path):
    router = MemoryRouter(Settings(
        humoid_memory_backend="sqlite",
        humoid_memory_db=tmp_path / "provider.sqlite3",
        humoid_rhomgaic_db=tmp_path / "rhomgaic.sqlite3",
    ))
    await router.initialize()
    text = "Fixed src/cache.py CacheIndex invalidation ordering and tests pass"
    first = await router.add(text, validation_status="verified")
    second = await router.add(text, validation_status="observed")

    result = await router.consolidate_memory_surface(similarity=0.90, max_aliases=8)
    assert result["provider_unchanged"] is True
    assert result["aliases_proposed"] >= 1

    sidecar_hits = await router.rhomgaic.search("src/cache.py CacheIndex invalidation", limit=8)
    returned = {hit.memory_id for hit in sidecar_hits}
    assert not ({first, second} <= returned)

    durable = await router.list_memories(limit=20)
    durable_ids = {hit.memory_id for hit in durable}
    assert first in durable_ids and second in durable_ids
    await router.close()


def test_retrieval_planner_deepens_debug_graph_and_narrows_exact_file():
    from humoid_tui.rhomgaic_memory import RhomgaicRetrievalPlanner

    planner = RhomgaicRetrievalPlanner(dimensions=96, default_graph_hops=2)
    plan = planner.plan("debug src/cache.py CacheIndex traceback", final_limit=8)

    assert plan.action == "debug"
    assert plan.code_signal >= 0.4
    assert plan.reflex_breadth == 1
    assert plan.deep_graph_hops == 3
    assert "exact-file surface" in plan.reasons


@pytest.mark.asyncio
async def test_rhomgaic_attached_code_surface_indexes_structure_without_raw_source(tmp_path):
    memory = RhomgaicMemory(tmp_path / "surface.sqlite3")
    await memory.initialize()
    await memory.add(
        "verified-fix",
        "Verified cache regression fix after pytest",
        validation_status="verified",
    )
    source = (
        "import asyncio\n\n"
        "class CacheIndex:\n"
        "    async def refresh(self):\n"
        "        await asyncio.sleep(0)\n"
    )
    receipt = await memory.attach_code_surface(
        "verified-fix",
        "src/cache.py",
        source,
    )

    assert receipt["indexed"] is True
    assert receipt["raw_source_persisted"] is False
    assert receipt["ast_terms"] >= 3
    hits = await memory.search("src/cache.py CacheIndex refresh", limit=3)
    assert hits and hits[0].memory_id == "verified-fix"
    stats = await memory.stats()
    assert stats["code_surfaces"] == 1


@pytest.mark.asyncio
async def test_rhomgaic_dependency_graph_links_definition_to_cross_file_call(tmp_path):
    memory = RhomgaicMemory(tmp_path / "dependencies.sqlite3")
    await memory.initialize()
    await memory.add(
        "definition",
        "Implemented src/core.py compute_total pricing primitive",
        validation_status="verified",
    )
    await memory.attach_code_surface(
        "definition",
        "src/core.py",
        "def compute_total(items):\n    return sum(items)\n",
    )
    await memory.add(
        "caller",
        "Updated src/service.py checkout pipeline",
        validation_status="verified",
    )
    await memory.attach_code_surface(
        "caller",
        "src/service.py",
        "from src.core import compute_total\n\ndef checkout(items):\n    return compute_total(items)\n",
    )

    anchors = await memory.search("src/core.py compute_total", limit=1)
    expanded = await memory.expand_graph(anchors, limit=8, hops=1)

    assert anchors and anchors[0].memory_id == "definition"
    related = next(hit for hit in expanded if hit.memory_id == "caller")
    assert any("dependency:" in item for item in related.metadata.get("graph_path", []))


@pytest.mark.asyncio
async def test_rhomgaic_importance_decay_prefers_fresher_equivalent_memory(tmp_path):
    memory = RhomgaicMemory(
        tmp_path / "importance.sqlite3",
        importance_half_life_days=90,
    )
    await memory.initialize()
    text = "Fixed src/cache.py CacheIndex invalidation regression"
    await memory.add(
        "old",
        text,
        validation_status="verified",
        created_at="2015-01-01T00:00:00+00:00",
    )
    await memory.add(
        "fresh",
        text,
        validation_status="verified",
        created_at="2026-09-03T12:00:00+00:00",
    )

    hits = await memory.search("src/cache.py CacheIndex invalidation", limit=2)

    assert [hit.memory_id for hit in hits] == ["fresh", "old"]
    assert hits[0].metadata["importance_score"] > hits[1].metadata["importance_score"]


@pytest.mark.asyncio
async def test_rhomgaic_distills_semantic_capsules_and_uses_them_as_support(tmp_path):
    memory = RhomgaicMemory(tmp_path / "capsules.sqlite3", semantic_capsules=True)
    await memory.initialize()
    await memory.add(
        "impl",
        "Fixed src/cache.py CacheIndex invalidation ordering",
        task_id="cache-regression",
        validation_status="verified",
    )
    await memory.add(
        "test",
        "Added tests/test_cache.py CacheIndex invalidation regression coverage",
        task_id="cache-regression",
        validation_status="verified",
    )
    await memory.add(
        "trace",
        "Debugged src/cache.py CacheIndex invalidation traceback",
        task_id="cache-regression",
        validation_status="observed",
    )

    result = await memory.distill_semantic_capsules(min_support=2, max_capsules=8)
    stats = await memory.stats()
    hits = await memory.search("fix src/cache.py CacheIndex invalidation", limit=6)

    assert result["capsules"] >= 1
    assert result["provider_unchanged"] is True
    assert stats["semantic_capsules"] >= 1
    assert any(hit.metadata.get("semantic_capsules") for hit in hits)


@pytest.mark.asyncio
async def test_rhomgaic_policy_adapts_only_from_auto_recall_geometry(tmp_path):
    from humoid_tui.rhomgaic_memory import RagStage, RhomgaicRetrievalPlanner
    from humoid_tui.models import MemoryHit

    memory = RhomgaicMemory(tmp_path / "policy.sqlite3", policy_adaptation=True)
    await memory.initialize()
    hit = MemoryHit(
        memory_id="m1",
        text="debug src/cache.py CacheIndex traceback",
        score=0.1,
        memory_tier="episodic",
        channel="agent",
        task_id="",
        validation_status="observed",
        created_at="2026-09-03T12:00:00+00:00",
        metadata={},
    )
    stages = [
        RagStage("reflex", [hit], 1.8, novelty=1.0, breadth=1),
        RagStage("working", [hit], 1.6, novelty=0.8, breadth=3),
        RagStage("subsurface", [hit], 1.5, novelty=0.7, breadth=6),
    ]
    query = "debug src/cache.py CacheIndex traceback"
    for _ in range(6):
        await memory.record_recall_outcome(query, stages, forced=False)

    hint = await memory.policy_hint(query)
    plan = RhomgaicRetrievalPlanner(dimensions=96, default_graph_hops=2).plan(
        query, 8, policy_hint=hint
    )

    assert hint["observations"] == 6
    assert hint["graph_delta"] == 1
    assert plan.deep_graph_hops == 4
    assert plan.policy_observations == 6
    assert any("adaptive policy" in reason for reason in plan.reasons)

    # Explicit/forced recalls are logged but must not train the adaptive policy.
    await memory.record_recall_outcome(query, stages, forced=True)
    after = await memory.policy_hint(query)
    assert after["observations"] == 6


@pytest.mark.asyncio
async def test_memory_tool_can_distill_rebuildable_semantic_capsules(tmp_path):
    from humoid_tui.tools import ToolRegistry

    settings = Settings(
        humoid_workspace=tmp_path / "workspace",
        humoid_memory_backend="sqlite",
        humoid_memory_db=tmp_path / "provider.sqlite3",
        humoid_rhomgaic_db=tmp_path / "rhomgaic.sqlite3",
    )
    router = MemoryRouter(settings)
    await router.initialize()
    await router.add(
        "Fixed src/cache.py CacheIndex regression",
        task_id="cache",
        validation_status="verified",
    )
    await router.add(
        "Added tests/test_cache.py CacheIndex regression coverage",
        task_id="cache",
        validation_status="verified",
    )
    tools = ToolRegistry(settings)
    tools.attach_memory(router)

    payload = await tools.execute("distill_memory_surface", {
        "min_support": 2,
        "max_capsules": 8,
        "dry_run": False,
    })

    assert '"capsules": 1' in payload
    assert '"provider_unchanged": true' in payload.lower()
    await tools.close()
    await router.close()
