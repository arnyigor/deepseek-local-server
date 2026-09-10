from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from deepseek_local_server.openai.tool_protocol import ParsedOutput


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def completion_payload(completion_id: str, model: str, parsed: ParsedOutput, reasoning: str, usage: Usage) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": parsed.content or None}
    if reasoning:
        message["reasoning_content"] = reasoning
    finish = "stop"
    if parsed.tool_calls:
        message["tool_calls"] = parsed.tool_calls
        finish = "tool_calls"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens + usage.reasoning_tokens,
            "total_tokens": usage.total_tokens,
            "completion_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
        },
    }


def sse(data: dict[str, Any] | str) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"data: {payload}\n\n".encode("utf-8")
