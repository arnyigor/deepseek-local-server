"""Live stress test for the ask_deepseek MCP tool: vision + multi-step logic.

Not part of the pytest suite (hits the real DeepSeek API through the running
`deepseek-local-server serve` process) — run manually:

    python mcp_vision_stress_test.py

Generates a synthetic image of colored, numbered squares with a known ground
truth, asks DeepSeek a question that requires (a) reading the numbers off the
image and (b) a multi-step arithmetic/logic chain on top of them, then checks
the final answer against the value computed at image-generation time. Runs
several randomized variations back to back to check for stability.
"""
from __future__ import annotations

import asyncio
import random
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from deepseek_local_server import mcp_server  # noqa: E402

COLORS = {"RED": (220, 40, 40), "BLUE": (40, 90, 220), "GREEN": (40, 160, 70)}


def _has_word(answer: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", answer.upper()) is not None


@dataclass
class Case:
    name: str
    question: str
    expected: str
    check: "callable[[str], bool]"


class _Ctx:
    async def report_progress(self, **kw):
        pass


def _draw_squares(squares: list[tuple[str, int]]) -> Path:
    size = 120
    img = Image.new("RGB", (size * len(squares), size + 20), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    for i, (color_name, value) in enumerate(squares):
        x0 = i * size + 10
        draw.rectangle([x0, 10, x0 + size - 20, size - 10], fill=COLORS[color_name])
        text = str(value)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x0 + (size - 20 - tw) / 2, 10 + (size - 20 - th) / 2 - bbox[1]), text, fill="white", font=font)
    path = Path(tempfile.gettempdir()) / f"mcp_vision_test_{random.randint(0, 999999)}.png"
    img.save(path)
    return path


def _make_squares(rng: random.Random, color_names: list[str]) -> list[tuple[str, int]]:
    return [(rng.choice(color_names), rng.randint(1, 20)) for _ in range(rng.randint(4, 6))]


def _make_case(rng: random.Random) -> tuple[Path, Case]:
    color_names = list(COLORS)
    kind = rng.choice(["parity", "threshold", "compare"])

    squares = _make_squares(rng, color_names)
    if kind == "compare":
        # Avoid a tie for the top sum — DeepSeek could legitimately pick either
        # color, which would make the check flaky rather than wrong.
        for _ in range(20):
            sums = [sum(v for cc, v in squares if cc == c) for c in color_names if any(cc == c for cc, _ in squares)]
            if len(sums) == len(set(sums)):
                break
            squares = _make_squares(rng, color_names)
    path = _draw_squares(squares)
    if kind == "parity":
        target = rng.choice(color_names)
        total = sum(v for c, v in squares if c == target)
        parity = "EVEN" if total % 2 == 0 else "ODD"
        question = (
            f"This image shows colored squares, each with a number written on it. "
            f"Add up the numbers written on only the {target} squares. "
            f"Then answer with exactly one word: EVEN if that sum is even, ODD if it is odd. "
            f"Reply with only that one word, nothing else."
        )
        return path, Case(f"parity-{target}", question, parity, lambda a: _has_word(a, parity))

    if kind == "threshold":
        target = rng.choice(color_names)
        total = sum(v for c, v in squares if c == target)
        threshold = rng.randint(10, 40)
        expected = "YES" if total > threshold else "NO"
        question = (
            f"This image shows colored squares, each with a number written on it. "
            f"Add up the numbers written on only the {target} squares, then check if that sum "
            f"is strictly greater than {threshold}. Reply with exactly one word: YES or NO."
        )
        return path, Case(f"threshold-{target}>{threshold}", question, expected, lambda a: _has_word(a, expected))

    sums = {c: sum(v for cc, v in squares if cc == c) for c in color_names if any(cc == c for cc, _ in squares)}
    winner = max(sums, key=sums.get)
    question = (
        "This image shows colored squares, each with a number written on it. "
        "For each color present, sum the numbers written on squares of that color. "
        "Then tell me which color has the highest sum. Reply with exactly one word: "
        "the color name in English, uppercase (RED, BLUE, or GREEN)."
    )
    return path, Case(f"compare-winner={winner}", question, winner, lambda a: _has_word(a, winner))


async def _run_case(rng: random.Random, index: int) -> bool:
    path, case = _make_case(rng)
    try:
        answer = await mcp_server.ask_deepseek(
            case.question,
            _Ctx(),
            reasoning=True,
            search=False,
            new_conversation=True,
            image_path=str(path),
            timeout_seconds=120,
        )
    finally:
        path.unlink(missing_ok=True)

    ok = case.check(answer)
    status = "PASS" if ok else "FAIL"
    print(f"[{index}] {status}  case={case.name}  expected={case.expected!r}  answer={answer[:200]!r}")
    return ok


async def main(n: int) -> None:
    rng = random.Random()
    results = [await _run_case(rng, i + 1) for i in range(n)]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    asyncio.run(main(count))
