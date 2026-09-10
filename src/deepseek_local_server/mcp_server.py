from __future__ import annotations

import asyncio
import base64
import mimetypes
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
        "Use DeepSeek Web as a second-opinion/research model through a local hybrid gateway. "
        "The direct Web API is primary and browser automation is only a fallback. "
        "DeepSeek merged its Instant/Expert/Vision modes into a single model (2026-09); "
        "there is no model choice left, only 'reasoning' and 'search' toggles. "
        "The merged model natively understands images: pass image_path to a local "
        "image file to ask about it. Pass all required context in the question."
    ),
)


def _model(reasoning: bool, search: bool) -> str:
    if search:
        return "deepseek-reasoner-search" if reasoning else "deepseek-chat-search"
    return "deepseek-reasoner" if reasoning else "deepseek-chat"


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
    reasoning: bool = True,
    search: bool = False,
    new_conversation: bool = False,
    image_path: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT,
) -> str:
    """Ask DeepSeek Web through the local gateway."""
    async with _lock:
        if new_conversation:
            _history.clear()
        try:
            model = _model(reasoning, search)
            token = read_api_token(_settings)
            content = _build_content(question, image_path)
        except Exception as exc:
            return f"deepseek-local-server configuration error: {exc}"

        user = {"role": "user", "content": content}
        messages = [*_history, user]
        task = asyncio.create_task(_post(model, messages, token, timeout_seconds))
        elapsed = 0.0
        try:
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=5)
                if done:
                    break
                elapsed += 5
                try:
                    await ctx.report_progress(progress=elapsed, total=timeout_seconds, message="Waiting for DeepSeek...")
                except Exception:
                    pass
            response = task.result()
        except httpx.ConnectError:
            return f"Could not reach deepseek-local-server at {_settings.api_base_url}. Start `deepseek-local-server serve`."
        except httpx.TimeoutException:
            return f"Timed out after {timeout_seconds:.0f}s."
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        if response.is_error:
            return f"deepseek-local-server error {response.status_code}: {response.text}"
        payload = response.json()
        message = payload["choices"][0]["message"]
        answer = message.get("content") or ""
        if not answer and message.get("tool_calls"):
            answer = str(message["tool_calls"])
        if answer:
            _history.extend([user, {"role": "assistant", "content": answer}])
        return answer or "(DeepSeek returned an empty response)"


async def _post(model: str, messages: list[dict[str, str]], token: str, timeout: float) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout + 10) as client:
        return await client.post(
            f"{_settings.api_base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}", "x-agent-session": _session_id},
            json={"model": model, "messages": messages, "stream": False},
        )


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
