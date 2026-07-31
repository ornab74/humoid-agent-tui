from __future__ import annotations

from humoid_tui.config import Settings
from humoid_tui.perspective import ProjectPerspectiveIndex
from humoid_tui.tools import ToolRegistry


def local_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("HUMOID_PERSPECTIVE_BACKEND", "local")
    monkeypatch.setenv("HUMOID_PERSPECTIVE_CHUNK_CHARS", "512")
    monkeypatch.setenv("HUMOID_PERSPECTIVE_CHUNK_OVERLAP", "64")
    monkeypatch.setenv("HUMOID_PERSPECTIVE_PACKET_MAX_CHARS", "3000")
    return Settings(humoid_workspace=tmp_path, humoid_memory_backend="sqlite")


async def test_build_search_expand_and_clear(tmp_path, monkeypatch):
    settings = local_settings(tmp_path, monkeypatch)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "engine.py").write_text(
        "class PerspectiveEngine:\n"
        "    def search(self, query):\n"
        "        return self.vector_store.search(query)\n\n"
        "def build_context_accordion(hits):\n"
        "    return hits[:8]\n"
    )
    (tmp_path / "README.md").write_text("# Demo\n\nPerspective search documentation.\n")

    perspective = ProjectPerspectiveIndex(settings, tmp_path)
    built = await perspective.build("review perspective retrieval")
    assert "local-vector-fallback" in built
    assert perspective.manifest["files"] == 2
    assert perspective.manifest["chunks"] >= 2

    packet = await perspective.search("Where is vector search and context folding implemented?", limit=4)
    assert "PROJECT PERSPECTIVE" in packet
    assert "src/engine.py" in packet
    assert "PerspectiveEngine" in packet

    chunk_id = next(
        chunk.chunk_id for chunk in perspective.chunks.values()
        if chunk.path == "src/engine.py"
    )
    expanded = await perspective.expand([chunk_id], radius=0)
    assert chunk_id in expanded
    assert "vector_store.search" in expanded

    cleared = await perspective.clear()
    assert "Cleared temporary project perspective" in cleared
    assert perspective.chunks == {}


async def test_context_packet_folds_lower_ranked_chunks(tmp_path, monkeypatch):
    settings = local_settings(tmp_path, monkeypatch)
    monkeypatch.setenv("HUMOID_PERSPECTIVE_PACKET_MAX_CHARS", "1200")
    for index in range(8):
        (tmp_path / f"module_{index}.py").write_text(
            f"def feature_{index}():\n"
            f"    return 'shared architecture retrieval context feature {index}'\n" * 20
        )

    perspective = ProjectPerspectiveIndex(settings, tmp_path)
    await perspective.build("review architecture")
    packet = await perspective.search("shared architecture retrieval context", limit=20, max_chars=1200)

    assert len(packet) <= 1200
    assert "CONTEXT ACCORDION" in packet
    assert "ids=" in packet


async def test_tool_registry_synchronizes_writes_and_undo(tmp_path, monkeypatch):
    settings = local_settings(tmp_path, monkeypatch)
    path = tmp_path / "app.py"
    path.write_text("def old_behavior():\n    return 'old'\n")
    tools = ToolRegistry(settings)

    await tools.execute("build_project_perspective", {"objective": "change behavior"})
    original_ids = set(tools.perspective.path_chunks["app.py"])

    await tools.execute("write_file", {
        "path": "app.py",
        "content": "def new_behavior():\n    return 'new perspective result'\n",
    })
    changed_ids = set(tools.perspective.path_chunks["app.py"])
    assert changed_ids
    assert changed_ids != original_ids
    assert "app.py" in tools.perspective.dirty_paths

    packet = await tools.execute("search_project_perspective", {"query": "new perspective result"})
    assert "new_behavior" in packet

    await tools.execute("undo_file_change", {})
    restored = await tools.execute("search_project_perspective", {"query": "old behavior"})
    assert "old_behavior" in restored


def test_perspective_tools_are_exposed(tmp_path, monkeypatch):
    settings = local_settings(tmp_path, monkeypatch)
    names = {schema["function"]["name"] for schema in ToolRegistry(settings).schemas()}
    assert {
        "build_project_perspective",
        "search_project_perspective",
        "expand_project_perspective",
        "project_perspective_status",
        "clear_project_perspective",
    } <= names
