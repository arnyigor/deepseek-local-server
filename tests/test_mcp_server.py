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


def test_answer_comes_first_with_the_dimmed_chain_appended(bridge):
    requests, replies, ctx = bridge
    replies.extend([
        {"deltas": [{"reasoning_content": "think"}, {"reasoning_content": "ing"}, {"content": "the answer"}]},
        {"deltas": [{"content": "next"}]},
    ])

    async def run():
        result = await mcp_server.ask_deepseek("question", ctx)
        assert result == f"the answer\n\n---\n{_dimmed('<reasoning>\nthinking\n</reasoning>')}"
        assert result.startswith("the answer")  # collapsed hosts show these lines first
        again = await mcp_server.ask_deepseek("follow-up", ctx)
        assert again == "next"  # no reasoning deltas this time -> nothing to append

    asyncio.run(run())
    # history stores the clean answer, not the reasoning wrapper
    assert requests[1]["messages"][1] == {"role": "assistant", "content": "the answer"}


def test_markdown_tables_become_box_drawing_tables(bridge):
    requests, replies, ctx = bridge
    table = [
        "| Поезд | Плацкарт, ₽ | Купе, ₽ |",
        "| :--- | ---: | ---: |",
        "| 002Й | 1 916 | 2 220 |",
    ]
    replies.append({"deltas": [{"content": "\n".join(table)}]})

    async def run():
        result = await mcp_server.ask_deepseek("question", ctx)
        lines = result.split("\n")
        assert lines[0].startswith("┌")
        assert lines[0].endswith("┐")
        assert "│ Поезд" in lines[1] and "│ 002Й" in lines[3]
        assert "       1 916 │" in lines[3]  # right-aligned numeric column
        assert lines[-1].startswith("└")
        assert "|" not in result  # no raw pipes left over

    asyncio.run(run())


def test_table_box_stays_within_the_width_budget(bridge):
    requests, replies, ctx = bridge
    table = "\n".join(
        [
            "| Параметр | Значение |",
            "| --- | --- |",
            f"| длинное описание параметра | {'x' * 120} |",
        ]
    )
    replies.append({"deltas": [{"content": table}]})

    async def run():
        result = await mcp_server.ask_deepseek("question", ctx)
        assert max(len(line) for line in result.split("\n")) <= mcp_server.TABLE_MAX_WIDTH_CHARS

    asyncio.run(run())


def test_table_inside_a_code_fence_is_left_alone(bridge):
    requests, replies, ctx = bridge
    fenced = "```\n| a | b |\n|---|---|\n| 1 | 2 |\n```"
    replies.append({"deltas": [{"content": fenced}]})

    async def run():
        assert await mcp_server.ask_deepseek("question", ctx) == fenced

    asyncio.run(run())


def test_escaped_pipe_stays_inside_one_cell(bridge):
    requests, replies, ctx = bridge
    replies.append({"deltas": [{"content": "| Выражение | Значение |\n|---|---|\n| a \\| b | 1 |"}]})

    async def run():
        result = await mcp_server.ask_deepseek("question", ctx)
        assert "a | b" in result
        content_rows = [line for line in result.split("\n") if line.startswith("│")]
        assert content_rows and all(row.count("│") == 3 for row in content_rows)

    asyncio.run(run())


def test_latex_backslashes_survive_in_table_cells(bridge):
    requests, replies, ctx = bridge
    replies.append(
        {
            "deltas": [
                {
                    "content": "| Тело | mu |\n|---|---|\n| Земля | 3{,}986\\cdot10^{14} |",
                }
            ]
        }
    )

    async def run():
        result = await mcp_server.ask_deepseek("question", ctx)
        assert "3{,}986\\cdot10^{14}" in result

    asyncio.run(run())


def test_wide_characters_keep_the_box_aligned(bridge):
    requests, replies, ctx = bridge
    table = "| 名前 | Флаг | Значение |\n|---|---|---|\n| 中文 | 🚀 | 1 |\n| ab | x | 2 |"
    replies.append({"deltas": [{"content": table}]})

    async def run():
        result = await mcp_server.ask_deepseek("question", ctx)
        widths = {mcp_server._display_width(line) for line in result.split("\n") if line}
        assert len(widths) == 1  # every row occupies the same number of cells

    asyncio.run(run())


def test_markdown_markers_are_cleaned_but_code_and_math_survive(bridge):
    requests, replies, ctx = bridge
    answer = "\n".join(
        [
            "### Заголовок",
            "",
            "1. **Жирный** пункт и [ссылка](http://x.org/a).",
            "",
            "\\[",
            "v_1 = \\sqrt{\\frac{\\mu}{R}}",
            "\\]",
            "",
            "Код: `**literal**` и $v_c$ остаются.",
            "",
            "зависит от \\(\\mu\\) и \\(R\\)",
            "",
            "```python",
            "s = '**not bold**'",
            "```",
        ]
    )
    replies.append({"deltas": [{"content": answer}]})

    async def run():
        result = await mcp_server.ask_deepseek("question", ctx)
        assert "### " not in result and result.startswith("Заголовок")
        assert "**Жирный**" not in result and "Жирный пункт" in result
        assert "[ссылка]" not in result and "ссылка (http://x.org/a)" in result
        assert "\\[" not in result and "v_1 = \\sqrt{\\frac{\\mu}{R}}" in result
        assert "\\(\\mu\\)" not in result and "зависит от \\mu" in result  # inline math delimiters dropped
        assert "`**literal**`" in result  # code spans are untouched
        assert "s = '**not bold**'" in result  # fenced code is untouched

    asyncio.run(run())


def test_pipe_lines_that_are_not_a_table_stay_untouched(bridge):
    requests, replies, ctx = bridge
    replies.append({"deltas": [{"content": "| not a table\nsecond line"}]})

    async def run():
        result = await mcp_server.ask_deepseek("question", ctx)
        assert result == "| not a table\nsecond line"

    asyncio.run(run())


def test_no_ticker_without_a_progress_token(bridge):
    """Direct-tool callers cannot receive progress, so nothing is sent but the result."""
    requests, replies, ctx = bridge
    replies.append({"deltas": [{"reasoning_content": "think"}, {"content": "the answer"}]})

    async def run():
        result = await mcp_server.ask_deepseek("question", ctx)
        assert result == f"the answer\n\n---\n{_dimmed('<reasoning>\nthink\n</reasoning>')}"
        assert ctx.report_progress.await_count == 0

    asyncio.run(run())


def test_huge_reasoning_is_trimmed_so_the_answer_survives(bridge):
    """Hosts cut oversized tool output from the head, so the chain must stay small."""
    requests, replies, ctx = bridge
    long_reasoning = "x" * 40_000
    replies.append({"deltas": [{"reasoning_content": long_reasoning}, {"content": "THE-ANSWER"}]})

    async def run():
        result = await mcp_server.ask_deepseek("question", ctx)
        assert result.startswith("THE-ANSWER")  # the answer survives any head truncation
        assert len(result) < 10_000
        assert "chars of reasoning omitted" in result

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
        assert all("\n" not in m for m in messages)  # one compact line, never a wall
        assert all(len(m) <= mcp_server.NOTIFY_TAIL_CHARS + 32 for m in messages)
        assert "second thought" in messages[-1]  # newest slice is reported
        # the chain travelled by ticker, so the result must be the bare answer
        assert result == "the answer"

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
