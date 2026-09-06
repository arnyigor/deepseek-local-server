from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from deepseek_local_server.openai.tool_protocol import ParsedAssistantOutput


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Deliberately marked as an estimate: DeepSeek Web does not expose tokenizer usage.
    return max(1, (len(text) + 2) // 3)


def completion_payload(
    *,
    completion_id: str,
    model: str,
    parsed: ParsedAssistantOutput,
    usage: Usage,
    created: int | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": parsed.content}
    if parsed.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json},
            }
            for call in parsed.tool_calls
        ]
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": parsed.finish_reason,
            }
        ],
        "usage": usage.as_dict(),
        "system_fingerprint": "deepseek-web-playwright-adapter",
    }


def _sse(data: dict[str, Any] | str) -> bytes:
    body = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"data: {body}\n\n".encode("utf-8")


async def buffered_stream(
    *,
    completion_id: str,
    model: str,
    parsed: ParsedAssistantOutput,
    usage: Usage,
    include_usage: bool,
) -> AsyncIterator[bytes]:
    created = int(time.time())
    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": "deepseek-web-playwright-adapter",
    }
    yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

    if parsed.tool_calls:
        for index, call in enumerate(parsed.tool_calls):
            delta = {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments_json},
                    }
                ]
            }
            yield _sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
    elif parsed.content:
        # The browser response is available only after completion. Chunking here preserves
        # protocol compatibility but is not real-time token streaming.
        chunk_size = 512
        for offset in range(0, len(parsed.content), chunk_size):
            delta = {"content": parsed.content[offset:offset + chunk_size]}
            yield _sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})

    yield _sse(
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": parsed.finish_reason}],
        }
    )
    if include_usage:
        yield _sse({**base, "choices": [], "usage": usage.as_dict()})
    yield _sse("[DONE]")
