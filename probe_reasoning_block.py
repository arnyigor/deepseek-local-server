"""Live check: the whole reasoning is delivered before the answer.

Runs the exact tool code path (mcp_server.ask_deepseek) in-process against the
running local server, with a Context stub that records notifications. Expected:
exactly ONE progress notification carrying the complete reasoning (sent when
the thinking phase ends, i.e. before the answer), plus the same chain as a
single '<reasoning>...</reasoning>' block in the returned text.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepseek_local_server import mcp_server  # noqa: E402


class RecordingCtx:
    def __init__(self):
        self.events = []

    async def report_progress(self, **kw):
        self.events.append(("progress", str(kw.get("message", ""))))

    async def info(self, data, **kw):
        self.events.append(("info", str(data)))


async def main():
    ctx = RecordingCtx()
    expected = str(86400 * 7 + 365)
    answer = await mcp_server.ask_deepseek(
        "Посчитай: сколько секунд в сутках? Умножь на 7 и прибавь 365. Дай только число без пояснений.",
        ctx,
        new_conversation=True,
    )
    print(f"ANSWER:\n{answer}\n")
    print(f"notifications: {len(ctx.events)}")
    for kind, msg in ctx.events:
        print(f"  {kind}: {msg[:90]!r}")

    one_block = answer.startswith("<reasoning>\n") and answer.count("<reasoning>") == 1
    closed = "</reasoning>\n\n" in answer
    body = answer.split("</reasoning>\n\n", 1)[-1] if closed else ""
    # exactly one notification, carrying the full reasoning, before the result
    notified_full = len(ctx.events) == 1 and ctx.events[0][0] == "progress"
    notified_text = ctx.events[0][1] if notified_full else ""
    chain = answer.split("</reasoning>", 1)[0].removeprefix("<reasoning>\n") if one_block else ""
    ok = one_block and closed and expected in body and notified_full and notified_text.strip() == chain.strip()
    print("\nRESULT:", "PASS" if ok else "FAIL")

asyncio.run(main())
