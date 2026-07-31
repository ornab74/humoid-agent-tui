from pathlib import Path
import pytest
from humoid_tui.config import Settings
from humoid_tui.tools import ToolRegistry, ToolError

@pytest.mark.asyncio
async def test_workspace_containment(tmp_path):
    s=Settings(humoid_workspace=tmp_path)
    t=ToolRegistry(s)
    await t.execute("write_file",{"path":"a.txt","content":"ok"})
    assert (tmp_path/"a.txt").read_text()=="ok"
    with pytest.raises(ToolError):
        await t.execute("read_file",{"path":"../escape.txt"})
