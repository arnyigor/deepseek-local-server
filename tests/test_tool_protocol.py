import json

import pytest

from deepseek_local_server.errors import ToolProtocolError
from deepseek_local_server.openai.tool_protocol import parse_assistant_output


def test_plain_text_without_tools() -> None:
    parsed = parse_assistant_output("Hello", set())
    assert parsed.content == "Hello"
    assert parsed.finish_reason == "stop"
    assert parsed.tool_calls == ()


def test_parses_single_tool_call() -> None:
    parsed = parse_assistant_output(
        '<<<DEEPSEEK_TOOL_CALL>>>{"name":"read","arguments":{"path":"README.md"}}<<<END_DEEPSEEK_TOOL_CALL>>>',
        {"read"},
    )
    assert parsed.content is None
    assert parsed.finish_reason == "tool_calls"
    assert parsed.tool_calls[0].name == "read"
    assert json.loads(parsed.tool_calls[0].arguments_json) == {"path": "README.md"}


def test_parses_multiple_tool_calls_and_residual_content() -> None:
    parsed = parse_assistant_output(
        'Checking.\n<<<DEEPSEEK_TOOL_CALL>>>{"name":"read","arguments":{"path":"a"}}<<<END_DEEPSEEK_TOOL_CALL>>>\n'
        '<<<DEEPSEEK_TOOL_CALL>>>{"name":"read","arguments":{"path":"b"}}<<<END_DEEPSEEK_TOOL_CALL>>>',
        {"read"},
    )
    assert parsed.content == "Checking."
    assert len(parsed.tool_calls) == 2


def test_rejects_unknown_tool() -> None:
    with pytest.raises(ToolProtocolError):
        parse_assistant_output(
            '<<<DEEPSEEK_TOOL_CALL>>>{"name":"delete_everything","arguments":{}}<<<END_DEEPSEEK_TOOL_CALL>>>',
            {"read"},
        )
