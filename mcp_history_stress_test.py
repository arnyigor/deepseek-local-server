"""Live stress test for ask_deepseek's conversation-history behavior.

Not part of the pytest suite (hits the real DeepSeek API through the running
`deepseek-local-server serve` process) — run manually:

    python mcp_history_stress_test.py [n]

mcp_server.py keeps history in one process-wide list and only clears it on
new_conversation=True; the server session behind it (chat_session_id chain)
resets independently whenever the incoming message list is shorter than what
it last saw. Three checks, each with randomized secrets so runs can't pass by
memorizing a fixed answer:

  recall     - tell a fact, ask for it back in the very next turn.
  reset      - tell a fact, start new_conversation, confirm it's forgotten.
  depth      - tell fact A, then fact B, then ask for fact A (2 turns back).
  image      - show an image with a number in turn 1, recall the number by
               memory alone (no image attached) in turn 2.
"""
from __future__ import annotations

import asyncio
import random
import re
import string
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from deepseek_local_server import mcp_server  # noqa: E402


class _Ctx:
    async def report_progress(self, **kw):
        pass


def _code(rng: random.Random) -> str:
    word = "".join(rng.choices(string.ascii_uppercase, k=5))
    return f"{word}-{rng.randint(100, 999)}"


def _has_word(answer: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", answer.upper()) is not None


async def _ask(question: str, **kw) -> str:
    return await mcp_server.ask_deepseek(question, _Ctx(), timeout_seconds=90, **kw)


def _draw_number_card(number: int) -> Path:
    img = Image.new("RGB", (200, 200), (30, 90, 180))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except OSError:
        font = ImageFont.load_default()
    text = str(number)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((200 - tw) / 2, (200 - th) / 2 - bbox[1]), text, fill="white", font=font)
    path = Path(tempfile.gettempdir()) / f"mcp_history_test_{random.randint(0, 999999)}.png"
    img.save(path)
    return path


async def _run_recall(rng: random.Random, index: int) -> bool:
    code = _code(rng)
    await _ask(f"Remember this secret code: {code}. Reply with only the word OK.", new_conversation=True, reasoning=False)
    answer = await _ask("What secret code did I just give you? Reply with only the code, nothing else.", reasoning=False)
    ok = _has_word(answer, code)
    print(f"[{index}] {'PASS' if ok else 'FAIL'}  recall  code={code!r}  answer={answer[:150]!r}")
    return ok


async def _run_reset(rng: random.Random, index: int) -> bool:
    old_code = _code(rng)
    await _ask(f"Remember this secret code: {old_code}. Reply with only the word OK.", new_conversation=True, reasoning=False)
    answer = await _ask(
        "What secret code did I give you earlier in this conversation? "
        "If you were not told one, reply with only the word UNKNOWN.",
        new_conversation=True,
        reasoning=False,
    )
    ok = not _has_word(answer, old_code)
    print(f"[{index}] {'PASS' if ok else 'FAIL'}  reset   old_code={old_code!r}  answer={answer[:150]!r}")
    return ok


async def _run_depth(rng: random.Random, index: int) -> bool:
    fact_a = str(rng.randint(1000, 9999))
    fact_b = rng.choice(["crimson", "azure", "amber", "violet", "teal"])
    await _ask(f"Remember fact A: the magic number is {fact_a}. Reply with only the word OK.", new_conversation=True, reasoning=False)
    await _ask(f"Remember fact B: the secret color is {fact_b}. Reply with only the word OK.", reasoning=False)
    answer = await _ask("Going back to fact A: what was the magic number? Reply with only the number.", reasoning=False)
    ok = fact_a in answer
    print(f"[{index}] {'PASS' if ok else 'FAIL'}  depth   fact_a={fact_a!r} (2 turns back)  answer={answer[:150]!r}")
    return ok


async def _run_image(rng: random.Random, index: int) -> bool:
    number = rng.randint(1000, 9999)
    path = _draw_number_card(number)
    try:
        await _ask(
            "This image shows a number on a blue card. Remember that number. Reply with only the word OK.",
            new_conversation=True,
            reasoning=False,
            image_path=str(path),
        )
    finally:
        path.unlink(missing_ok=True)
    answer = await _ask(
        "Without looking at any image again, what number was on the blue card I showed you? Reply with only the number.",
        reasoning=False,
    )
    ok = str(number) in answer
    print(f"[{index}] {'PASS' if ok else 'FAIL'}  image   number={number}  answer={answer[:150]!r}")
    return ok


CASES = [_run_recall, _run_reset, _run_depth, _run_image]


async def main(n: int) -> None:
    rng = random.Random()
    results = []
    for i in range(n):
        case = rng.choice(CASES)
        results.append(await case(rng, i + 1))
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    asyncio.run(main(count))
