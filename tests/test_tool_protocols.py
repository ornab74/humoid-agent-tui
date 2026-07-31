from humoid_tui.tool_protocols import (
    Gemma4Protocol,
    ProtocolRegistry,
    TaggedJSONProtocol,
)


def test_openai_structured_glm():
    protocol = ProtocolRegistry().resolve("digitalocean", "glm-5.2")
    calls = protocol.parse(
        text="",
        structured=[{
            "id": "x",
            "function": {
                "name": "read_file",
                "arguments": '{"path":"a.py"}',
            },
        }],
        provider="digitalocean",
        model="glm-5.2",
    )
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.py"}
    assert "TOOL.CALL/v1" in calls[0].envelope()


def test_gemma4_native_tokens():
    text = (
        '<|tool_call>call:read_file{path:<|"|>src/main.py<|"|>}'
        '<tool_call|><|tool_response>'
    )
    calls = Gemma4Protocol().parse(
        text=text,
        structured=[],
        provider="llamacpp",
        model="gemma-4-e2b-it",
    )
    assert calls[0].arguments["path"] == "src/main.py"


def test_tagged_json_local():
    text = (
        '<tool_call>{"name":"list_files",'
        '"arguments":{"path":"."}}</tool_call>'
    )
    calls = TaggedJSONProtocol().parse(
        text=text,
        structured=[],
        provider="llamacpp",
        model="qwen",
    )
    assert calls[0].name == "list_files"


def test_structured_arguments_accept_mapping():
    protocol = ProtocolRegistry().resolve("digitalocean", "glm-5.2")
    calls = protocol.parse(
        text="",
        structured=[{
            "id": "mapping",
            "function": {
                "name": "write_file",
                "arguments": {
                    "path": "scratchpad/agentconvos.md",
                    "content": "hello",
                },
            },
        }],
        provider="digitalocean",
        model="glm-5.2",
    )
    assert calls[0].arguments["path"] == "scratchpad/agentconvos.md"


def test_structured_arguments_unwrap_double_encoded_json():
    protocol = ProtocolRegistry().resolve("digitalocean", "glm-5.2")
    arguments = '"{\\"path\\":\\"scratchpad/agentconvos.md\\",\\"content\\":\\"hello\\"}"'
    calls = protocol.parse(
        text="",
        structured=[{
            "id": "double",
            "function": {
                "name": "write_file",
                "arguments": arguments,
            },
        }],
        provider="digitalocean",
        model="glm-5.2",
    )
    assert calls[0].arguments == {
        "path": "scratchpad/agentconvos.md",
        "content": "hello",
    }


def test_structured_arguments_recover_cumulative_snapshots():
    protocol = ProtocolRegistry().resolve("digitalocean", "glm-5.2")
    arguments = (
        '{"path":"scratchpad/agentconvos.md"}'
        '{"path":"scratchpad/agentconvos.md","content":"complete"}'
    )
    calls = protocol.parse(
        text="",
        structured=[{
            "id": "cumulative",
            "function": {
                "name": "write_file",
                "arguments": arguments,
            },
        }],
        provider="digitalocean",
        model="glm-5.2",
    )
    assert calls[0].arguments == {
        "path": "scratchpad/agentconvos.md",
        "content": "complete",
    }


def test_structured_arguments_accept_markdown_fence_and_literal_newline():
    protocol = ProtocolRegistry().resolve("digitalocean", "glm-5.2")
    arguments = '```json\n{"path":"note.md","content":"line one\nline two"}\n```'
    calls = protocol.parse(
        text="",
        structured=[{
            "id": "fenced",
            "function": {
                "name": "write_file",
                "arguments": arguments,
            },
        }],
        provider="digitalocean",
        model="glm-5.2",
    )
    assert calls[0].arguments["content"] == "line one\nline two"


def test_malformed_structured_call_falls_back_to_tagged_text():
    protocol = ProtocolRegistry().resolve("digitalocean", "glm-5.2")
    calls = protocol.parse(
        text=(
            '<tool_call>{"name":"write_file","arguments":'
            '{"path":"fallback.md","content":"ok"}}</tool_call>'
        ),
        structured=[{
            "id": "bad",
            "function": {
                "name": "write_file",
                "arguments": "{not valid",
            },
        }],
        provider="digitalocean",
        model="glm-5.2",
    )
    assert calls[0].arguments == {"path": "fallback.md", "content": "ok"}
