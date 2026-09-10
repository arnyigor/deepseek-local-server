from deepseek_local_server.openai.tool_protocol import parse_output


def test_json_tool_call():
    parsed = parse_output('{"tool_call":{"name":"read_file","arguments":{"path":"a.kt"}}}', {"read_file"})
    assert parsed.content == ""
    assert parsed.tool_calls[0]["function"]["name"] == "read_file"
    assert 'a.kt' in parsed.tool_calls[0]["function"]["arguments"]


def test_unknown_tool_stays_text():
    raw = '{"tool_call":{"name":"delete_world","arguments":{}}}'
    parsed = parse_output(raw, {"read_file"})
    assert parsed.content == raw
    assert parsed.tool_calls == []
