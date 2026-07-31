import sys

from humoid_tui.config import Settings
from humoid_tui.context_accordion import ContextAccordion
from humoid_tui.env_store import EnvStore
from humoid_tui.i18n import LOCALES, translate
from humoid_tui.local_models import HardwareProfile, LocalModelManager
from humoid_tui.memory import EmbeddedWeaviateMemory, SQLiteMemory
from humoid_tui.tools import ToolError, ToolRegistry


def test_env_store_preserves_template_and_updates_values(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# local\nFOO=old\n")
    store = EnvStore(path)
    store.set("foo", "new value")
    store.set("BAR", "yes")
    assert store.values() == {"FOO": "new value", "BAR": "yes"}
    assert path.read_text().startswith("# local\n")


def test_context_accordion_retains_expandable_source():
    accordion = ContextAccordion(context_limit=10, threshold=0.5)
    messages = [{"role": "system", "content": "rules"}] + [
        {"role": "user", "content": f"message {index} src/file.py unresolved"}
        for index in range(8)
    ]
    folded = accordion.fold(messages)
    assert len(folded) < len(messages)
    assert accordion.archives
    assert accordion.expanded(0)[0] == messages[1]


def test_cuda_wheel_selection_is_bounded_to_official_matrix(tmp_path):
    manager = LocalModelManager(tmp_path)
    supported = manager.install_command(HardwareProfile("cuda", "GPU", "12.4"))
    unsupported = manager.install_command(HardwareProfile("cuda", "GPU", "12.8"))
    if sys.version_info[:2] in {(3, 10), (3, 11), (3, 12)}:
        assert any("cu124" in item for item in supported)
    assert "llama-cpp-python[server]" in supported
    assert not any("cu128" in item for item in unsupported)


def test_african_and_asian_language_catalog():
    for code in ("sw", "am", "ha", "yo", "zu", "ar", "hi", "bn", "ur", "zh", "ja", "ko", "id"):
        assert code in LOCALES
        assert translate(code, "memory") != ""
    assert translate("sw", "memory") == "KUMBUKUMBU"
    assert translate("zh", "sessions") == "会话"


async def test_memory_crud(tmp_path):
    memory = SQLiteMemory(tmp_path / "memory.sqlite3")
    await memory.initialize()
    memory_id = await memory.add("draft")
    assert (await memory.list_memories())[0].text == "draft"
    await memory.update_memory(memory_id, text="edited", validation_status="verified")
    updated = (await memory.list_memories())[0]
    assert updated.text == "edited"
    assert updated.validation_status == "verified"
    await memory.delete_memory(memory_id)
    assert await memory.list_memories() == []


async def test_file_preview_edit_and_undo(tmp_path):
    tools = ToolRegistry(Settings(humoid_workspace=tmp_path))

    async def approve(_path, diff, _content):
        assert "+draft" in diff
        return True, "edited"

    tools.approve_write = approve
    await tools.execute("write_file", {"path": "note.txt", "content": "draft"})
    assert (tmp_path / "note.txt").read_text() == "edited"
    await tools.execute("undo_file_change", {})
    assert not (tmp_path / "note.txt").exists()


async def test_benchmark_reports_missing_model_without_launching_module(tmp_path):
    manager = LocalModelManager(tmp_path)
    result = await manager.benchmark(tmp_path / "missing.gguf")
    assert result["ok"] is False
    assert "Model file not found" in result["error"]


async def test_local_launch_uses_low_memory_server_flags(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    manager = LocalModelManager(tmp_path)
    captured = []

    class Process:
        pid = 42
        returncode = None

    async def create_process(*command, **_kwargs):
        captured.extend(command)
        return Process()

    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)
    await manager.launch(model)

    assert captured[captured.index("--use_mlock") + 1] == "false"
    assert captured[captured.index("--n_batch") + 1] == "128"
    assert captured[captured.index("--n_ubatch") + 1] == "64"
    assert captured[captured.index("--logits_all") + 1] == "false"


async def test_temporary_tool_requires_read_only_validated_steps(tmp_path):
    (tmp_path / "known.txt").write_text("known value")
    tools = ToolRegistry(Settings(humoid_workspace=tmp_path))
    result = await tools.execute("invent_tool", {
        "name": "read_known",
        "description": "Read the known fixture",
        "steps": [{"tool": "read_file", "arguments": {"path": "known.txt"}}],
        "validation_cases": [{"contains": "known value"}],
    })
    assert "validated" in result
    assert await tools.execute("read_known", {}) == "known value"
    try:
        await tools.execute("invent_tool", {
            "name": "unsafe", "description": "unsafe",
            "steps": [{"tool": "run_command", "arguments": {"command": "true"}}],
            "validation_cases": [{"contains": ""}],
        })
    except ToolError:
        pass
    else:
        raise AssertionError("unsafe invented tool was accepted")


def test_embedded_weaviate_adopts_healthy_existing_process(tmp_path):
    class Client:
        def __init__(self):
            self.closed = False

        def is_ready(self):
            return True

        def close(self):
            self.closed = True

    client = Client()

    class Weaviate:
        @staticmethod
        def connect_to_local(**kwargs):
            assert kwargs == {"host": "127.0.0.1", "port": 8079, "grpc_port": 50050}
            return client

    memory = EmbeddedWeaviateMemory(Settings(
        weaviate_embedded_data_path=tmp_path / "data",
        weaviate_embedded_binary_path=tmp_path / "bin",
    ))
    memory._ensure_collection = lambda: None
    memory._crud_roundtrip = lambda: None

    assert memory._adopt_existing(Weaviate)
    assert memory.client is client
    assert memory.connection_mode == "adopted-existing"
    assert not client.closed
