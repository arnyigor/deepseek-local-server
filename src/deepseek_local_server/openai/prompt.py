from __future__ import annotations

import json
from typing import Any

from deepseek_local_server.openai.content import content_to_text
from deepseek_local_server.openai.schemas import ChatCompletionRequest, ChatMessage, ToolDefinition

_TOOL_START = "<<<DEEPSEEK_TOOL_CALL>>>"
_TOOL_END = "<<<END_DEEPSEEK_TOOL_CALL>>>"


def _pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _tool_choice_instruction(tool_choice: Any) -> str:
    if tool_choice is None or tool_choice == "auto":
        return "Use a tool only when you actually need it; otherwise just answer normally."
    if tool_choice == "none":
        return "Don't use any tool this turn, just answer in plain text."
    if tool_choice == "required":
        return "Use at least one of the tools below before giving your final answer."
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if isinstance(function, dict) and function.get("name"):
            return f"Use the {function['name']!r} tool for this turn."
    return "Follow the requested tool choice."


def _format_tools(tools: list[ToolDefinition]) -> str:
    lines: list[str] = []
    for tool in tools:
        fn = tool.function
        lines.append(f"- {fn.name}: {fn.description or '(no description)'}")
        lines.append(f"  arguments schema: {json.dumps(fn.parameters, ensure_ascii=False)}")
    return "\n".join(lines)


def _format_assistant_tool_calls(message: ChatMessage) -> str:
    if not message.tool_calls:
        return ""
    lines: list[str] = []
    for call in message.tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments", "{}")
        try:
            decoded = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            decoded = arguments
        lines.append(f"  (called {function.get('name')} with {json.dumps(decoded, ensure_ascii=False)})")
    return "\n".join(lines)


def _format_message(message: ChatMessage) -> str:
    text = content_to_text(message.content)

    if message.role == "assistant" and message.tool_calls:
        extra = _format_assistant_tool_calls(message)
        return f"{text}\n{extra}" if text else extra

    return text


def build_prompt(
    request: ChatCompletionRequest,
    *,
    messages: list[ChatMessage] | None = None,
    start_index: int = 0,
    continuation: bool = False,
) -> str:
    tools = request.tools or []
    transcript = messages if messages is not None else request.messages

    sections: list[str] = []

    # The tool contract was already given at the start of this chat; DeepSeek still sees it
    # above in the visible conversation, so repeating it on every follow-up turn only
    # adds noise.
    if not continuation:
        if tools and request.tool_choice != "none":
            parallel = request.parallel_tool_calls is not False
            sections.append(
                "\n".join(
                    [
                        "You have access to these tools:",
                        _format_tools(tools),
                        "",
                        _tool_choice_instruction(request.tool_choice),
                        "To call one, write exactly this and nothing else around it:",
                        _TOOL_START,
                        '{"name":"tool_name","arguments":{"arg":"value"}}',
                        _TOOL_END,
                        "Use the exact tool name from the list above, valid JSON arguments, and don't say the"
                        " tool already ran until you see its result."
                        + (" You can call more than one in a turn." if parallel else " Call at most one per turn."),
                    ]
                )
            )
        elif tools:
            sections.append("Tools are available but don't use any this turn — just answer in plain text.")

    sections.extend(_format_message(message) for message in transcript)

    return "\n\n".join(section for section in sections if section)
