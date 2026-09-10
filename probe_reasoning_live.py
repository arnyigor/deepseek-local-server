"""Live check: reasoning ticks arrive WHILE the model is thinking.

Simulates the caller pi produces on its proxy path (a request that carries a
progress token) and records notification timestamps against the total call time.

Run manually: .venv/Scripts/python.exe probe_reasoning_live.py
"""
import asyncio
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Cyrillic output on cp1251 consoles
sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepseek_local_server import mcp_server  # noqa: E402


class ProgressCtx:
    """Stand-in for pi's proxy call: request carries a progress token."""

    class _RequestContext:
        meta = {"progress_token": 1}

    request_context = _RequestContext()

    def __init__(self):
        self.ticks: list[tuple[float, str]] = []
        self.t0 = time.monotonic()

    async def report_progress(self, progress=None, total=None, message=None):
        self.ticks.append((time.monotonic() - self.t0, message or ""))


async def main():
    ctx = ProgressCtx()
    question = (
        "Найди в интернете и посчитай: сколько сейчас стоит билет Москва — Мурманск в плацкарте и купе, "
        "во сколько обойдётся поездка для 2 человек в плацкарте, и сколько идёт поезд. Покажи расчёт."
    )
    result = await mcp_server.ask_deepseek(question, ctx, new_conversation=True)
    total = time.monotonic() - ctx.t0

    print(f"call took {total:.1f}s, {len(ctx.ticks)} reasoning ticks")
    for dt, message in ctx.ticks:
        print(f"  t={dt:5.1f}s  {message[:100]!r}")

    first_tick_before_end = bool(ctx.ticks) and ctx.ticks[0][0] < total - 0.5
    mid_call_ticks = [t for t, _ in ctx.ticks if t < total - 0.5]
    answer_body = result.strip()

    checks = {
        "several ticks during thinking": len(mid_call_ticks) >= 1,
        "first tick well before the call ended": first_tick_before_end,
        "reasoning stayed out of the result": "<reasoning>" not in result and "\x1b[" not in result,
        "result is the answer": len(answer_body) > 0,
        "ticks are compact single lines": all(
            m.startswith("thinking: ") and "\n" not in m and len(m) <= mcp_server.NOTIFY_TAIL_CHARS + 32
            for _, m in ctx.ticks
        ),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("\nRESULT:", "PASS" if all(checks.values()) else "FAIL")
    raise SystemExit(0 if all(checks.values()) else 1)

asyncio.run(main())
