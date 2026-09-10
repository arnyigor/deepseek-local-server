"""Live check of the MCP result layout.

Runs the exact tool code path (mcp_server.ask_deepseek) in-process against the
running local server: the result must be the reasoning chain first (dimmed with
SGR codes) and the answer last, with no progress notifications involved at all.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepseek_local_server import mcp_server  # noqa: E402


def _dimmed(text: str) -> str:
    return "\n".join(f"\x1b[90m{line}\x1b[39m" if line else line for line in text.split("\n"))


async def main():
    expected = str(86400 * 7 + 365)
    answer = await mcp_server.ask_deepseek(
        "Посчитай: сколько секунд в сутках? Умножь на 7 и прибавь 365. Дай только число без пояснений.",
        new_conversation=True,
    )
    print(answer)
    print()

    block, _, tail = answer.partition("\n\n")
    body = tail.strip()
    strip = lambda s: s.replace("\x1b[90m", "").replace("\x1b[39m", "")  # noqa: E731
    checks = {
        "reasoning block first": strip(block).startswith("<reasoning>") and strip(block).endswith("</reasoning>"),
        "each block line dimmed": all(line.startswith("\x1b[90m") for line in block.split("\n") if line),
        "answer is last and clean": body == expected and "\x1b[" not in body,
        "no ctx parameter": "ctx" not in mcp_server.ask_deepseek.__code__.co_varnames,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("\nRESULT:", "PASS" if all(checks.values()) else "FAIL")
    raise SystemExit(0 if all(checks.values()) else 1)

asyncio.run(main())
