"""Live check of the no-ticker layout: answer first, dimmed chain appended.

Callers without a progress channel (pi's direct-tools path) get the reasoning
inside the result, after the answer, so a collapsed block still shows the answer.
"""
import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Cyrillic output on cp1251 consoles
sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepseek_local_server import mcp_server  # noqa: E402


class NoProgressCtx:
    """Stand-in for a caller whose request carries no progress token."""

    class RC:
        meta = None

    request_context = RC()

    async def report_progress(self, **kw):  # pragma: no cover
        raise AssertionError("no notifications should be sent without a progress token")


async def main():
    expected = str(86400 * 7 + 365)
    answer = await mcp_server.ask_deepseek(
        "Посчитай: сколько секунд в сутках? Умножь на 7 и прибавь 365. Дай только число без пояснений.",
        NoProgressCtx(),
        new_conversation=True,
    )
    print(answer)
    print()

    head, separator, tail = answer.partition("\n\n---\n")
    checks = {
        "answer comes first": head.strip() == expected,
        "chain appended after a separator": separator != "" and tail.startswith("\x1b[90m<reasoning>"),
        "chain is dimmed": tail.rstrip().endswith("\x1b[39m"),
        "no progress sent": True,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("\nRESULT:", "PASS" if all(checks.values()) else "FAIL")
    raise SystemExit(0 if all(checks.values()) else 1)


asyncio.run(main())
