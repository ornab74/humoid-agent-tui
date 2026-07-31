import pytest
from humoid_tui.memory import SQLiteMemory

@pytest.mark.asyncio
async def test_memory_search(tmp_path):
    m=SQLiteMemory(tmp_path/"m.sqlite")
    await m.initialize()
    await m.add("alpha beta gamma", memory_tier="semantic")
    hits=await m.search("alpha gamma")
    assert hits and hits[0].memory_tier=="semantic"
