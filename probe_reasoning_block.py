"""Live check: the tool returns the WHOLE reasoning chain as one block.

Runs the exact tool code path (mcp_server.ask_deepseek) in-process against the
running local server, with a Context stub that records any notifications:
there must be none (no chunked progress updates), and the returned text must be
one complete '<reasoning>...</reasoning>' block followed by the answer.
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
    ok = one_block and closed and expected in body and not ctx.events
    print("\nRESULT:", "PASS" if ok else "FAIL")

asyncio.run(main())
