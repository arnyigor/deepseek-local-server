from __future__ import annotations

import asyncio

import httpx
from mcp.server.mcpserver import Context, MCPServer

from deepseek_local_server.auth import read_api_token
from deepseek_local_server.config import Settings

_settings = Settings.from_env()

# Web-search-augmented and DeepThink (reasoning) answers can take much longer than a plain
# reply, so the default needs real headroom above the browser's own generation timeout.
_DEFAULT_TIMEOUT_SECONDS = _settings.request_timeout_seconds + 120

# stdio has one client per process. History lives until that client disconnects.
_history: list[dict[str, str]] = []
_conversation_lock = asyncio.Lock()

server = MCPServer(
    name="deepseek-web",
    instructions=(
        "Ask a plain-text question to DeepSeek (chat.deepseek.com) through a local browser "
        "bridge, with DeepThink (R1 reasoning / expert mode) enabled by default. It has no "
        "access to your files, shell, or other tools -- put all needed context directly in "
        "the question. Good for a second opinion, brainstorming, or explaining something; "
        "not for multi-step agentic work. DeepThink answers can take a few minutes -- don't "
        "lower timeout_seconds below the default. Calls continue the current conversation; "
        "set new_conversation=true to start a fresh chat."
    ),
)


@server.tool()
async def ask_deepseek(
    question: str,
    ctx: Context,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    new_conversation: bool = False,
) -> str:
    """Ask DeepSeek Web (DeepThink) a plain-text question and return its answer.

    Calls remember earlier questions and answers in this MCP process. Set
    `new_conversation=True` to discard that history and start a new chat. Reconnecting
    the MCP process also clears history. DeepSeek cannot read your project files or run
    commands, so paste any relevant code or context directly into the question. Replies
    can take a few minutes with DeepThink (reasoning mode) enabled -- the default timeout
    already accounts for that.
    """
    async with _conversation_lock:
        if new_conversation:
            _history.clear()
        return await _ask_in_conversation(question, ctx, timeout_seconds)


async def _ask_in_conversation(question: str, ctx: Context, timeout_seconds: float) -> str:
    try:
        token = read_api_token(_settings)
    except (FileNotFoundError, RuntimeError) as exc:
        return f"deepseek-local-server is not initialized: {exc}"

    user_message = {"role": "user", "content": question}
    messages = [*_history, user_message]

    async def do_request() -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout_seconds + 10) as client:
            return await client.post(
                f"{_settings.api_base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "model": _settings.model_id,
                    "messages": messages,
                    "stream": False,
                },
            )

    task = asyncio.create_task(do_request())
    elapsed = 0.0
    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=5)
            if done:
                break
            elapsed += 5
            try:
                await ctx.report_progress(progress=elapsed, total=timeout_seconds, message="Waiting for DeepSeek Web...")
            except Exception:
                pass
        response = task.result()
    except httpx.ConnectError:
        return (
            f"Could not reach deepseek-local-server at {_settings.api_base_url}. "
            "Start it first with `deepseek-local-server serve`."
        )
    except httpx.TimeoutException:
        return f"Timed out waiting for DeepSeek Web after {timeout_seconds:.0f}s."
    finally:
        # Do not leave an orphan HTTP request after an MCP cancellation.
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    if response.is_error:
        return f"deepseek-local-server returned an error ({response.status_code}): {response.text}"

    payload = response.json()
    answer = payload["choices"][0]["message"].get("content")
    if not answer:
        return "(DeepSeek returned an empty response)"
    _history.extend([user_message, {"role": "assistant", "content": answer}])
    return answer


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
