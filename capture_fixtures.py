"""Capture real DeepSeek answers as rendering fixtures.

Writes tests/fixtures/<name>.raw.txt (what the model returned) and
tests/fixtures/<name>.expected.txt (what the renderer produced) for the
snapshot test in tests/test_render_fixtures.py.

Run manually (needs the gateway running): .venv/Scripts/python.exe capture_fixtures.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepseek_local_server import mcp_server  # noqa: E402

FIXTURES = Path(__file__).parent / "tests" / "fixtures"

CASES = {
    "tables": "Компактно: таблица 3 строки (Мерлин 1D / Raptor 2 / RD-180: тяга, удельный импульс) и одна формула.",
    "math": "Компактно: формула Циолковского с пояснением символов, формула первой космической скорости и период круговой орбиты.",
    "latex_doc": "Дай расчёт в виде LaTeX-документа: первая космическая скорость для Земли и Луны с таблицей значений.",
    "code": "Компактно: короткий код на Python для расчёта Δv по формуле Циолковского и один пример вывода.",
}


class Ctx:
    """Caller that can receive progress, so the result stays the bare answer."""

    class RC:
        meta = {"progress_token": 1}

    request_context = RC()

    async def report_progress(self, **kw):  # pragma: no cover - ticker sink
        pass


async def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, question in CASES.items():
        render = mcp_server._render_markdown
        mcp_server._render_markdown = lambda text: text
        raw = await mcp_server.ask_deepseek(question, Ctx(), new_conversation=True)
        mcp_server._render_markdown = render
        rendered = render(raw)
        (FIXTURES / f"{name}.raw.txt").write_text(raw, encoding="utf-8")
        (FIXTURES / f"{name}.expected.txt").write_text(rendered, encoding="utf-8")
        print(f"===== {name} ({len(raw)} raw chars) =====")
        print(rendered)
        print()


asyncio.run(main())
