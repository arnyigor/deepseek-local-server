"""Live check: reasoning must stream to the MCP client BEFORE the final answer.

Runs the exact tool code path (mcp_server.ask_deepseek) in-process against the
running local server, with a Context stub that records notifications.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepseek_local_server import mcp_server  # noqa: E402


class RecordingCtx:
    def __init__(self):
        self.t0 = time.monotonic()
        self.events = []

    def _rec(self, kind):
        def fn(**kw):
            dt = time.monotonic() - self.t0
            msg = kw.get("message") or (kw.get("data") if kw else "")
            self.events.append((dt, kind, str(msg)))
        return fn

    async def report_progress(self, **kw):
        self._rec("progress")(**kw)

    async def info(self, data, **kw):
        dt = time.monotonic() - self.t0
        self.events.append((dt, "info", str(data)))


async def main():
    ctx = RecordingCtx()
    expected = str(86400 * 7 + 365)
    answer = await mcp_server.ask_deepseek(
        f"Посчитай: сколько секунд в сутках? Умножь на 7 и прибавь 365. Дай только число без пояснений.",
        ctx,
        new_conversation=True,
    )
    print(f"t={time.monotonic() - ctx.t0:5.1f}s  ANSWER: {answer!r}")
    print(f"\nnotifications: {len(ctx.events)}")
    for dt, kind, msg in ctx.events:
        print(f"  t={dt:5.1f}s  {kind:8s}  {msg[:90]!r}")
    ok = bool(answer) and expected in answer and len(ctx.events) >= 1
    print("\nRESULT:", "PASS" if ok else "FAIL")

asyncio.run(main())
