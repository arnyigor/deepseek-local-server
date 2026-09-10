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
        "reasoning is always on and is streamed back to the caller, only 'search' remains a toggle. "
        "The merged model natively understands images: pass image_path to a local "
        "image file to ask about it. Pass all required context in the question."
    ),
)


def _model(search: bool) -> str:
    return "deepseek-reasoner-search" if search else "deepseek-reasoner"


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
    search: bool = False,
    new_conversation: bool = False,
    image_path: str | None = None,
    include_reasoning: bool = True,
    timeout_seconds: float = _DEFAULT_TIMEOUT,
) -> str:
    """Ask DeepSeek Web through the local gateway.

    Reasoning is always on. It streams to the client via progress notifications
    (message carries the accumulated reasoning tail) and is included in the
    returned text before the answer: <reasoning>...</reasoning> + final answer.
    Pass include_reasoning=False to get the bare answer only.
    """
    async with _lock:
        if new_conversation:
            _history.clear()
        try:
            model = _model(search)
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
        state = {"last_emit": time.monotonic(), "reasoning_flushed": False}

        async def _emit(text: str, *, force: bool = False) -> None:
            """Push the visible tail of streamed text to the MCP client."""
            now = time.monotonic()
            if not force and now - state["last_emit"] < 1.0:
                return
            state["last_emit"] = now
            tail = text[-500:]
            try:
                await ctx.report_progress(progress=float(len(text)), message=tail)
                await ctx.info(tail)
            except Exception:
                pass  # notifications are best-effort

        async def _consume(client: httpx.AsyncClient) -> None:
            nonlocal error_text
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
                            await _emit("".join(reasoning_acc), force=len(reasoning_acc) == 1)
                        if delta.get("content"):
                            if reasoning_acc and not state["reasoning_flushed"]:
                                state["reasoning_flushed"] = True
                                await _emit("".join(reasoning_acc), force=True)  # final flush of short reasoning
                            answer_acc.append(delta["content"])
                        if delta.get("tool_calls"):
                            tool_markup.append(json.dumps(delta["tool_calls"], ensure_ascii=False))

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds + 10) as client:
                task = asyncio.create_task(_consume(client))
                try:
                    while not task.done():
                        done, _ = await asyncio.wait({task}, timeout=5)
                        if done:
                            break
                        if time.monotonic() - state["last_emit"] >= 5:
                            await _emit("".join(reasoning_acc) or "Waiting for DeepSeek...", force=True)
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
        if include_reasoning and reasoning_text:
            return f"<reasoning>\n{reasoning_text}\n</reasoning>\n\n{answer}"
        return answer


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
