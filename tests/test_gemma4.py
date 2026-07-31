from humoid_tui.gemma4 import (
    Gemma4RuntimePolicy, render_tool_call, render_tool_response,
    strip_thought_channels, gemma_system_suffix,
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
