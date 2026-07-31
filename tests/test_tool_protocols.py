from humoid_tui.tool_protocols import ProtocolRegistry, Gemma4Protocol, TaggedJSONProtocol


def test_openai_structured_glm():
    p=ProtocolRegistry().resolve("digitalocean","glm-5.2")
    calls=p.parse(text="", structured=[{"id":"x","function":{"name":"read_file","arguments":"{\"path\":\"a.py\"}"}}], provider="digitalocean",model="glm-5.2")
    assert calls[0].name=="read_file"
    assert calls[0].arguments=={"path":"a.py"}
    assert "TOOL.CALL/v1" in calls[0].envelope()


def test_gemma4_native_tokens():
    text='<|tool_call>call:read_file{path:<|"|>src/main.py<|"|>}<tool_call|><|tool_response>'
    calls=Gemma4Protocol().parse(text=text,structured=[],provider="llamacpp",model="gemma-4-e2b-it")
    assert calls[0].arguments["path"]=="src/main.py"


def test_tagged_json_local():
    text='<tool_call>{"name":"list_files","arguments":{"path":"."}}</tool_call>'
    calls=TaggedJSONProtocol().parse(text=text,structured=[],provider="llamacpp",model="qwen")
    assert calls[0].name=="list_files"
