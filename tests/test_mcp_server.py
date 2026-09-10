import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from deepseek_local_server import mcp_server


def test_build_content_without_image_is_plain_string():
    assert mcp_server._build_content("hello", None) == "hello"


def test_build_content_with_image_embeds_base64_data_url(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes")
    content = mcp_server._build_content("what is this?", str(image))
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_content_rejects_missing_image_path():
    with pytest.raises(ValueError):
        mcp_server._build_content("q", "/no/such/file.png")


def _sse(reply) -> httpx.Response:
    """Reply in the shape the gateway's SSE stream uses."""
    if isinstance(reply, Exception):
        raise reply
    if isinstance(reply, int):
        return httpx.Response(reply, text="test failure")
    if isinstance(reply, dict):
        deltas = reply["deltas"]
    else:
        deltas = [{"content": reply}]
    events = [json.dumps({"choices": [{"delta": d, "finish_reason": None}]}) for d in deltas]
    return httpx.Response(200, text="".join(f"data: {e}\n\n" for e in events) + "data: [DONE]\n\n")


@pytest.fixture
def bridge(monkeypatch):
    requests = []
    replies = []

    async def handle(request):
        requests.append(json.loads(request.content))
        return _sse(replies.pop(0))

    client_class = httpx.AsyncClient
    monkeypatch.setattr(mcp_server.httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(handle), **kwargs
    ))
    monkeypatch.setattr(mcp_server, "read_api_token", lambda _: "test-token")
    monkeypatch.setattr(mcp_server, "_history", [])
    monkeypatch.setattr(mcp_server, "_lock", asyncio.Lock())
    ctx = AsyncMock()
    # no progressToken -> the caller cannot receive progress notifications
    ctx.request_context.meta = None
    return requests, replies, ctx


def test_continuation_and_explicit_reset(bridge):
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


def test_model_is_always_reasoning_plus_search(bridge):
    requests, replies, ctx = bridge
    replies.extend(["detailed", "searched"])

    async def run():
        await mcp_server.ask_deepseek("first", ctx)
        await mcp_server.ask_deepseek("continue", ctx)

    asyncio.run(run())
    assert requests[0]["model"] == "deepseek-reasoner-search"
    assert requests[1]["model"] == "deepseek-reasoner-search"
    assert len(requests[1]["messages"]) == 3


def _dimmed(text: str) -> str:
    return "\n".join(f"\x1b[90m{line}\x1b[39m" if line else line for line in text.split("\n"))


def test_reasoning_comes_first_dimmed_and_answer_last(bridge):
    requests, replies, ctx = bridge
    replies.extend([
        {"deltas": [{"reasoning_content": "think"}, {"reasoning_content": "ing"}, {"content": "the answer"}]},
        {"deltas": [{"content": "next"}]},
    ])

    async def run():
        result = await mcp_server.ask_deepseek("question", ctx)
        assert result == f"{_dimmed('<reasoning>\nthinking\n</reasoning>')}\n\nthe answer"
        assert result.endswith("the answer")  # answer keeps the host's normal colour
        again = await mcp_server.ask_deepseek("follow-up", ctx)
        assert again == "next"  # no reasoning deltas this time -> nothing to prepare

    asyncio.run(run())
    # history stores the clean answer, not the reasoning wrapper
    assert requests[1]["messages"][1] == {"role": "assistant", "content": "the answer"}


def test_no_ticker_without_a_progress_token(bridge):
    """Direct-tool callers cannot receive progress, so nothing is sent but the result."""
    requests, replies, ctx = bridge
    replies.append({"deltas": [{"reasoning_content": "think"}, {"content": "the answer"}]})

    async def run():
        result = await mcp_server.ask_deepseek("question", ctx)
        assert result == f"{_dimmed('<reasoning>\nthink\n</reasoning>')}\n\nthe answer"
        assert ctx.report_progress.await_count == 0

    asyncio.run(run())


def test_ticker_streams_reasoning_when_the_caller_supports_progress(bridge):
    requests, replies, ctx = bridge
    ctx.request_context.meta = {"progress_token": 5}
    replies.append(
        {
            "deltas": [
                {"reasoning_content": "first thought"},
                {"reasoning_content": " second thought"},
                {"content": "the answer"},
            ]
        }
    )

    async def run():
        result = await mcp_server.ask_deepseek("question", ctx)
        assert ctx.report_progress.await_count >= 1
        messages = [call.kwargs["message"] for call in ctx.report_progress.await_args_list]
        assert all(m.startswith("thinking: ") for m in messages)
        assert "second thought" in messages[-1]  # newest slice is reported
        assert result == f"{_dimmed('<reasoning>\nfirst thought second thought\n</reasoning>')}\n\nthe answer"

    asyncio.run(run())


def test_cancellation_stops_request_without_updating_history(monkeypatch):
    started = asyncio.Event()
    stopped = asyncio.Event()

    class FakeStream:
        async def __aenter__(self):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def __aexit__(self, *exc):
            return False

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.stream = lambda *args, **kwargs: FakeStream()  # used as async CM, not awaited
    monkeypatch.setattr(mcp_server.httpx, "AsyncClient", lambda **_: client)
    monkeypatch.setattr(mcp_server, "read_api_token", lambda _: "test-token")
    monkeypatch.setattr(mcp_server, "_history", [])
    monkeypatch.setattr(mcp_server, "_lock", asyncio.Lock())
    ctx = AsyncMock()
    ctx.request_context.meta = {"progress_token": 1}

    async def run():
        task = asyncio.create_task(mcp_server.ask_deepseek("cancel me", ctx))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()
        assert not mcp_server._lock.locked()
        assert mcp_server._history == []

    asyncio.run(run())
