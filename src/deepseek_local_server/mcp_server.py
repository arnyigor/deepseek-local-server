from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import time
from pathlib import Path
from uuid import uuid4

import httpx
from mcp.server.mcpserver import Context, MCPServer

from deepseek_local_server.auth import read_api_token
from deepseek_local_server.config import Settings

_settings = Settings.from_env()
_DEFAULT_TIMEOUT = _settings.request_timeout_seconds + 30
_history: list[dict[str, str]] = []
_lock = asyncio.Lock()
_session_id = f"mcp-{uuid4().hex}"

server = MCPServer(
    name="deepseek-web",
    instructions=(
        "Use DeepSeek Web as a second-opinion/research model through a local gateway. "
        "All requests go through the direct DeepSeek Web API. "
        "DeepSeek merged its Instant/Expert/Vision modes into a single model (2026-09); "
        "reasoning and web search are always on and the full reasoning is returned with the answer. "
        "The merged model natively understands images: pass image_path to a local "
        "image file to ask about it. Pass all required context in the question."
    ),
)


MODEL = "deepseek-reasoner-search"  # reasoning + web search are always on

# Live reasoning ticker: only sent when the caller can receive progress at all.
NOTIFY_INTERVAL_SECONDS = 1.5
NOTIFY_MAX_CHARS = 12_000

# The chain stored in the result must stay small: hosts truncate oversized tool
# output from the head, which would eat the answer if the chain came first.
RESULT_REASONING_HEAD_CHARS = 4_000
RESULT_REASONING_TAIL_CHARS = 2_000


def _can_receive_progress(ctx: Context | None) -> bool:
    """True when the request carried a progress token (i.e. notifications land somewhere).

    Hosts only inject it for proxy-style calls; direct tool calls get nothing, so
    there the result alone carries the reasoning.
    """
    if ctx is None:
        return False
    try:
        meta = ctx.request_context.meta
    except Exception:
        return False
    # the framework normalises _meta.progressToken into meta["progress_token"]
    return bool(isinstance(meta, dict) and (meta.get("progress_token") or meta.get("progressToken")))


_GRAY = "\x1b[90m"
_RESET_FG = "\x1b[39m"


def _trim_reasoning(text: str) -> str:
    """Bound the chain kept in the result so the answer can never be pushed out."""
    if len(text) <= RESULT_REASONING_HEAD_CHARS + RESULT_REASONING_TAIL_CHARS:
        return text
    omitted = len(text) - RESULT_REASONING_HEAD_CHARS - RESULT_REASONING_TAIL_CHARS
    head = text[:RESULT_REASONING_HEAD_CHARS]
    tail = text[-RESULT_REASONING_TAIL_CHARS:]
    return f"{head}\n… [{omitted} chars of reasoning omitted] …\n{tail}"


def _dim(text: str) -> str:
    """Paint text grey inside a terminal line.

    MCP gives no styling channel, and hosts paint every result line with their
    own "tool output" colour, so the reasoning is dimmed with SGR codes that win
    over the outer colour; the final reset must not leak into the answer.
    """
    return "\n".join(f"{_GRAY}{line}{_RESET_FG}" if line else line for line in text.split("\n"))


def _build_content(question: str, image_path: str | None) -> str | list[dict[str, object]]:
    if not image_path:
        return question
    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"image_path does not exist: {image_path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
    ]


@server.tool()
async def ask_deepseek(
    question: str,
    ctx: Context,
    new_conversation: bool = False,
    image_path: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT,
) -> str:
    """Ask DeepSeek Web through the local gateway.

    Reasoning and web search are always on. While the model thinks, the running
    reasoning is reported as progress messages (hosts that support progress show
    it live, e.g. pi's status line). The result then carries the whole chain
    first (dimmed) and the answer last:
    '<reasoning>...</reasoning>' + final answer.
    """
    async with _lock:
        if new_conversation:
            _history.clear()
        try:
            model = MODEL
            token = read_api_token(_settings)
            content = _build_content(question, image_path)
        except Exception as exc:
            return f"deepseek-local-server configuration error: {exc}"

        user = {"role": "user", "content": content}
        messages = [*_history, user]

        reasoning_acc: list[str] = []
        answer_acc: list[str] = []
        tool_markup: list[str] = []
        error_text: str | None = None
        live = _can_receive_progress(ctx)
        last_notify = 0.0
        thinking_done = False

        async def _tick(text: str, *, force: bool = False) -> None:
            """Show the reasoning so far; hosts update the status block in place."""
            nonlocal last_notify
            if not live or not text:
                return
            now = time.monotonic()
            if not force and now - last_notify < NOTIFY_INTERVAL_SECONDS:
                return
            last_notify = now
            shown = text if len(text) <= NOTIFY_MAX_CHARS else f"…\n{text[-NOTIFY_MAX_CHARS:]}"
            try:
                await ctx.report_progress(progress=float(len(text)), message=f"thinking:\n{shown}")
            except Exception:
                pass  # notifications are best-effort

        async def _consume(client: httpx.AsyncClient) -> None:
            nonlocal error_text, thinking_done
            async with client.stream(
                "POST",
                f"{_settings.api_base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}", "x-agent-session": _session_id},
                json={"model": model, "messages": messages, "stream": True},
            ) as response:
                if response.is_error:
                    body = (await response.aread()).decode("utf-8", "replace")
                    error_text = f"deepseek-local-server error {response.status_code}: {body}"
                    return
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(event.get("error"), dict):
                            error_text = f"deepseek-local-server error: {event['error'].get('message', data)}"
                            return
                        choices = event.get("choices") or [{}]
                        delta = choices[0].get("delta") or {}
                        if delta.get("reasoning_content"):
                            reasoning_acc.append(delta["reasoning_content"])
                            await _tick("".join(reasoning_acc))
                        if delta.get("content"):
                            if not thinking_done:
                                thinking_done = True
                                await _tick("".join(reasoning_acc), force=True)  # thinking finished
                            answer_acc.append(delta["content"])
                        if delta.get("tool_calls"):
                            tool_markup.append(json.dumps(delta["tool_calls"], ensure_ascii=False))

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds + 10) as client:
                task = asyncio.create_task(_consume(client))
                try:
                    await asyncio.wait({task})
                    task.result()
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        except httpx.ConnectError:
            return f"Could not reach deepseek-local-server at {_settings.api_base_url}. Start `deepseek-local-server serve`."
        except httpx.TimeoutException:
            return f"Timed out after {timeout_seconds:.0f}s."

        if error_text:
            return error_text
        answer = "".join(answer_acc) or "".join(tool_markup)
        if answer:
            _history.extend([user, {"role": "assistant", "content": answer}])
        answer = answer or "(DeepSeek returned an empty response)"
        reasoning_text = "".join(reasoning_acc)
        if reasoning_text:
            block = f"<reasoning>\n{_trim_reasoning(reasoning_text)}\n</reasoning>"
            return f"{_dim(block)}\n\n{answer}"
        return answer


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
