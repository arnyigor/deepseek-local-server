from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from deepseek_local_server.errors import ToolProtocolError

_TOOL_BLOCK = re.compile(r"<<<DEEPSEEK_TOOL_CALL>>>\s*(.*?)\s*<<<END_DEEPSEEK_TOOL_CALL>>>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True, slots=True)
class ParsedAssistantOutput:
    content: str | None
    tool_calls: tuple[ParsedToolCall, ...]
    finish_reason: str


def _decode_payload(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    fence = _CODE_FENCE.match(candidate)
    if fence:
        candidate = fence.group(1).strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ToolProtocolError("Tool-call payload must be a JSON object")
    return value


def _normalise_call(payload: dict[str, Any], allowed_tools: set[str]) -> ParsedToolCall:
    name = payload.get("name")
    arguments = payload.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise ToolProtocolError("Tool-call payload is missing a valid name")
    if name not in allowed_tools:
        raise ToolProtocolError(f"DeepSeek attempted unknown tool: {name}")
    if isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolProtocolError(f"Tool arguments for {name} are not valid JSON") from exc
    else:
        parsed_arguments = arguments
    if not isinstance(parsed_arguments, dict):
        raise ToolProtocolError(f"Tool arguments for {name} must be an object")
    return ParsedToolCall(
        id=f"call_{uuid4().hex[:24]}",
        name=name,
        arguments_json=json.dumps(parsed_arguments, ensure_ascii=False, separators=(",", ":")),
    )


def _parse_json_fallback(output: str, allowed_tools: set[str]) -> tuple[ParsedToolCall, ...]:
    candidate = output.strip()
    fence = _CODE_FENCE.match(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return ()
    if not isinstance(payload, dict) or "tool_calls" not in payload:
        return ()
    calls = payload["tool_calls"]
    if not isinstance(calls, list):
        return ()
    return tuple(_normalise_call(call, allowed_tools) for call in calls if isinstance(call, dict))


def parse_assistant_output(output: str, allowed_tool_names: set[str]) -> ParsedAssistantOutput:
    if not allowed_tool_names:
        return ParsedAssistantOutput(content=output.strip(), tool_calls=(), finish_reason="stop")

    matches = list(_TOOL_BLOCK.finditer(output))
    if matches:
        calls = tuple(_normalise_call(_decode_payload(match.group(1)), allowed_tool_names) for match in matches)
        content = _TOOL_BLOCK.sub("", output).strip() or None
        return ParsedAssistantOutput(content=content, tool_calls=calls, finish_reason="tool_calls")

    fallback = _parse_json_fallback(output, allowed_tool_names)
    if fallback:
        return ParsedAssistantOutput(content=None, tool_calls=fallback, finish_reason="tool_calls")

    return ParsedAssistantOutput(content=output.strip(), tool_calls=(), finish_reason="stop")
