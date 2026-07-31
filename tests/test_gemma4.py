from humoid_tui.agent import AgentHarness
from humoid_tui.config import Settings
from humoid_tui.gemma4 import (
    Gemma4RuntimePolicy,
    gemma_system_suffix,
    render_tool_call,
    render_tool_response,
    strip_thought_channels,
)
from humoid_tui.tool_protocols import Gemma4Protocol


def test_runtime_policy():
    assert Gemma4RuntimePolicy(mode="auto").active("gemma-4-4b-it")
    assert not Gemma4RuntimePolicy(mode="off").active("gemma-4-4b-it")
    assert Gemma4RuntimePolicy(mode="native").active("anything")


def test_native_roundtrip_parse():
    text = render_tool_call("read_file", {"path": "src/a,b.py", "line": 7})
    calls = Gemma4Protocol().parse(text=text, structured=[], provider="llamacpp", model="gemma-4")
    assert calls[0].name == "read_file"
    assert calls[0].arguments["path"] == "src/a,b.py"
    assert calls[0].arguments["line"] == 7


def test_response_and_thought_strip():
    assert "<|tool_response>" in render_tool_response("x", "ok", True)
    text = "<|channel>thought\nsecret<channel|>Visible"
    assert strip_thought_channels(text) == "Visible"


def test_declarations_include_native_tokens():
    suffix = gemma_system_suffix([{"type":"function","function":{"name":"x","description":"d","parameters":{"type":"object","properties":{}}}}], "low")
    assert "<|tool>declaration:x" in suffix
    assert "<|think|>" in suffix


async def _ignore_event(_event):
    pass


def test_request_messages_normalize_null_assistant_content(tmp_path):
    harness = AgentHarness(
        Settings(
            humoid_provider="llamacpp",
            humoid_workspace=tmp_path,
            llamacpp_model="gemma-4-test",
        ),
        _ignore_event,
    )
    harness.messages.append({"role": "assistant", "content": None})

    outgoing = harness._request_messages()

    assert outgoing[-1] == {"role": "assistant", "content": ""}
    # Request normalization must not mutate persisted session history.
    assert harness.messages[-1]["content"] is None


def test_request_messages_normalize_llamacpp_tool_arguments(tmp_path):
    harness = AgentHarness(
        Settings(
            humoid_provider="llamacpp",
            humoid_workspace=tmp_path,
            llamacpp_model="gemma-4-test",
        ),
        _ignore_event,
    )
    harness.messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "list_files", "arguments": "{}"},
        }],
    })

    outgoing = harness._request_messages()

    assert outgoing[-1]["tool_calls"][0]["function"]["arguments"] == {}
    assert harness.messages[-1]["tool_calls"][0]["function"]["arguments"] == "{}"
