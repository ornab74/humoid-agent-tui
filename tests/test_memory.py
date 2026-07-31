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
