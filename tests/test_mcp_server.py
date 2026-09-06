import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from deepseek_local_server import mcp_server
from deepseek_local_server.config import Settings
from deepseek_local_server.openai.schemas import ChatCompletionRequest
from deepseek_local_server.service import CompletionService


@pytest.fixture
def bridge(monkeypatch):
    requests = []
    replies = []

    async def handle(request):
        requests.append(json.loads(request.content))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            return httpx.Response(reply, text="test failure")
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(mcp_server.httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(handle), **kwargs
    ))
    monkeypatch.setattr(mcp_server, "read_api_token", lambda _: "test-token")
    monkeypatch.setattr(mcp_server, "_history", [])
    monkeypatch.setattr(mcp_server, "_conversation_lock", asyncio.Lock())
    return requests, replies, AsyncMock()


def test_continuation_and_explicit_reset(bridge, tmp_path):
    requests, replies, ctx = bridge
    replies.extend(["accepted", "BLUE-CAT-42", "new topic", "continued"])

    async def run():
        assert await mcp_server.ask_deepseek("Remember BLUE-CAT-42", ctx) == "accepted"
        assert await mcp_server.ask_deepseek("Which code?", ctx) == "BLUE-CAT-42"
        assert await mcp_server.ask_deepseek("Hello", ctx, new_conversation=True) == "new topic"
        assert await mcp_server.ask_deepseek("Continue", ctx) == "continued"

    asyncio.run(run())
    assert requests[1]["messages"] == [
        {"role": "user", "content": "Remember BLUE-CAT-42"},
        {"role": "assistant", "content": "accepted"},
        {"role": "user", "content": "Which code?"},
    ]
    assert requests[2]["messages"] == [{"role": "user", "content": "Hello"}]
    assert requests[3]["messages"][0:2] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "new topic"},
    ]

    # Verify the payloads actually select continuation in the existing backend.
    service = CompletionService(Settings(home=tmp_path))
    page = type("Page", (), {"is_closed": lambda self: False})()
    service._session_page = page
    service._session_messages = ChatCompletionRequest(**requests[0]).messages
    assert service._continuation_delta(ChatCompletionRequest(**requests[1]).messages) is not None
    assert service._continuation_delta(ChatCompletionRequest(**requests[2]).messages) is None


@pytest.mark.parametrize("failure", [503, httpx.ConnectError("offline"), httpx.ReadTimeout("slow"), None])
def test_failed_call_does_not_pollute_history(bridge, failure):
    requests, replies, ctx = bridge
    replies.extend(["first answer", failure, "retry answer"])

    async def run():
        await mcp_server.ask_deepseek("first", ctx)
        await mcp_server.ask_deepseek("failed question", ctx)
        await mcp_server.ask_deepseek("retry", ctx)

    asyncio.run(run())
    assert requests[2]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "retry"},
    ]


def test_failed_reset_does_not_restore_old_history(bridge):
    requests, replies, ctx = bridge
    replies.extend(["old answer", 503, "fresh answer"])

    async def run():
        await mcp_server.ask_deepseek("old question", ctx)
        await mcp_server.ask_deepseek("new question", ctx, new_conversation=True)
        await mcp_server.ask_deepseek("retry", ctx)

    asyncio.run(run())
    assert requests[2]["messages"] == [{"role": "user", "content": "retry"}]


def test_concurrent_calls_preserve_order(bridge):
    requests, replies, ctx = bridge
    replies.extend(["first answer", "second answer"])

    async def run():
        await asyncio.gather(
            mcp_server.ask_deepseek("first", ctx),
            mcp_server.ask_deepseek("second", ctx),
        )

    asyncio.run(run())
    assert requests[1]["messages"][0:2] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer"},
    ]


def test_cancellation_stops_request_without_updating_history(monkeypatch):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def post(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.side_effect = post
    monkeypatch.setattr(mcp_server.httpx, "AsyncClient", lambda **_: client)
    monkeypatch.setattr(mcp_server, "read_api_token", lambda _: "test-token")
    monkeypatch.setattr(mcp_server, "_history", [])
    monkeypatch.setattr(mcp_server, "_conversation_lock", asyncio.Lock())

    async def run():
        task = asyncio.create_task(mcp_server.ask_deepseek("cancel me", AsyncMock()))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()
        assert not mcp_server._conversation_lock.locked()
        assert mcp_server._history == []

    asyncio.run(run())
