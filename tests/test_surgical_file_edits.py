import hashlib

import pytest

from humoid_tui.config import Settings
from humoid_tui.tools import ToolError, ToolRegistry


@pytest.mark.asyncio
async def test_append_file_does_not_require_full_content(tmp_path):
    path = tmp_path / "ideas.md"
    path.write_text("# Ideas\n\n1. Existing\n")
    tools = ToolRegistry(Settings(humoid_workspace=tmp_path))

    result = await tools.execute("append_file", {
        "path": "ideas.md",
        "content": "\n2. New idea\n",
    })

    assert path.read_text() == "# Ideas\n\n1. Existing\n\n2. New idea\n"
    assert "sha256=" in result


@pytest.mark.asyncio
async def test_insert_text_requires_unambiguous_anchor(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("## Summary\nold\n## Summary\nold again\n")
    tools = ToolRegistry(Settings(humoid_workspace=tmp_path))

    with pytest.raises(ToolError, match="match count"):
        await tools.execute("insert_text", {
            "path": "README.md",
            "anchor": "## Summary\n",
            "content": "## New section\n",
            "position": "before",
        })

    await tools.execute("insert_text", {
        "path": "README.md",
        "anchor": "## Summary\n",
        "content": "## New section\n",
        "position": "before",
        "occurrence": 2,
        "expected_matches": 2,
    })
    assert path.read_text().count("## New section") == 1


@pytest.mark.asyncio
async def test_replace_text_honors_sha_precondition(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("MODE = 'old'\n")
    tools = ToolRegistry(Settings(humoid_workspace=tmp_path))
    stale_hash = hashlib.sha256(b"different").hexdigest()

    with pytest.raises(ToolError, match="precondition failed"):
        await tools.execute("replace_text", {
            "path": "app.py",
            "old_text": "MODE = 'old'",
            "new_text": "MODE = 'new'",
            "expected_sha256": stale_hash,
        })
    assert path.read_text() == "MODE = 'old'\n"


@pytest.mark.asyncio
async def test_atomic_multi_edit_and_single_undo(tmp_path):
    path = tmp_path / "notes.md"
    original = "# Notes\n\n## Summary\nold\n"
    path.write_text(original)
    tools = ToolRegistry(Settings(humoid_workspace=tmp_path))

    await tools.execute("apply_file_edits", {
        "path": "notes.md",
        "edits": [
            {"operation": "insert_before", "anchor": "## Summary", "content": "## Added\nnew\n\n"},
            {"operation": "replace", "old_text": "old", "content": "updated"},
            {"operation": "append", "content": "footer\n", "ensure_newline": True},
        ],
    })

    assert "## Added" in path.read_text()
    assert "updated" in path.read_text()
    assert path.read_text().endswith("footer\n")

    await tools.execute("undo_file_change", {})
    assert path.read_text() == original


@pytest.mark.asyncio
async def test_atomic_edit_failure_writes_nothing(tmp_path):
    path = tmp_path / "config.toml"
    original = "enabled = false\n"
    path.write_text(original)
    tools = ToolRegistry(Settings(humoid_workspace=tmp_path))

    with pytest.raises(ToolError, match="no file changes were written"):
        await tools.execute("apply_file_edits", {
            "path": "config.toml",
            "edits": [
                {"operation": "replace", "old_text": "false", "content": "true"},
                {"operation": "insert_after", "anchor": "missing-anchor", "content": "x"},
            ],
        })

    assert path.read_text() == original


def test_surgical_tools_are_exposed(tmp_path):
    tools = ToolRegistry(Settings(humoid_workspace=tmp_path))
    names = {schema["function"]["name"] for schema in tools.schemas()}
    assert {"append_file", "insert_text", "replace_text", "apply_file_edits", "read_file_range"} <= names
