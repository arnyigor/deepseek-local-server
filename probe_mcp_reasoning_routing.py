"""End-to-end check over real MCP stdio: reasoning routing depends on the caller.

Spawns `python -m deepseek_local_server mcp` and calls ask_deepseek twice against
the running gateway:

  1. with progress_callback  -> the SDK adds _meta.progressToken, so the complete
     reasoning must arrive as ONE progress notification BEFORE the result, and
     the result must be the bare answer (this mirrors pi's proxy path).
  2. without progress_callback -> no progressToken, so the result must carry the
     reasoning block instead (this mirrors pi's direct-tools path).

Run manually: .venv/Scripts/python.exe probe_mcp_reasoning_routing.py
"""
import asyncio
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).parent
QUESTION = "Посчитай: (12 + 8) * 3 - 5. Ответь только числом."


async def call(session: ClientSession, *, with_progress: bool):
    events: list[tuple[float, str, str]] = []
    t0 = time.monotonic()

    def on_progress(progress: float, total: float | None, message: str | None) -> None:
        events.append((time.monotonic() - t0, "progress", message or ""))

    async def on_progress_async(progress: float, total: float | None, message: str | None) -> None:
        on_progress(progress, total, message)

    started = time.monotonic()
    result = await session.call_tool(
        "ask_deepseek",
        {"question": QUESTION, "new_conversation": True},
        progress_callback=on_progress_async if with_progress else None,
    )
    answer = "\n".join(block.text for block in result.content if getattr(block, "type", "") == "text")
    return answer, events, time.monotonic() - started


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "deepseek_local_server", "mcp"], cwd=str(ROOT)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            answer, events, took = await call(session, with_progress=True)
            bare = answer.strip() == "55" and "<reasoning>" not in answer
            notified = len(events) == 1 and bool(events[0][2].strip())
            early = bool(events) and events[0][0] < took  # notification before the result
            print(f"[progress-capable] {took:4.1f}s answer={answer.strip()!r} notifications={len(events)}")
            if events:
                print(f"  first notification at t={events[0][0]:.1f}s (result at t={took:.1f}s)")
                print(f"  reasoning in notification: {events[0][2][:80]!r}")

            answer2, events2, took2 = await call(session, with_progress=False)
            embedded = answer2.startswith("55") and "<reasoning>" in answer2 and answer2.rstrip().endswith("</reasoning>")
            print(f"[progress-less]    {took2:4.1f}s answer starts: {answer2[:40]!r} notifications={len(events2)}")
            print(f"  reasoning appended to result: {embedded}")

    ok = bare and notified and early and embedded and not events2
    print("\nRESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

asyncio.run(main())
