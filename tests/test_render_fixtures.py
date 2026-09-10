"""Snapshot tests over real DeepSeek answers captured in tests/fixtures/.

Each fixture is a pair: <name>.raw.txt (exactly what the model returned) and
<name>.expected.txt (what the renderer must produce). Regenerate the raw files
with capture_fixtures.py (needs the gateway) and the expectations with
refresh_fixtures.py (offline).
"""
from pathlib import Path

import pytest

from deepseek_local_server import mcp_server

FIXTURE_DIR = Path(__file__).parent / "fixtures"
RAW_FIXTURES = sorted(FIXTURE_DIR.glob("*.raw.txt"))
FIXTURE_IDS = [path.stem for path in RAW_FIXTURES]

LATEX_SCAFFOLDING = ("\\documentclass", "\\usepackage", "\\begin{", "\\end{", "\\toprule", "\\caption{", "\\section{")


def _expected_path(raw_path: Path) -> Path:
    return raw_path.with_suffix("").with_suffix(".expected.txt")


def test_fixtures_exist():
    assert RAW_FIXTURES, "no rendering fixtures captured"
    for raw_path in RAW_FIXTURES:
        assert _expected_path(raw_path).is_file(), f"missing expectation for {raw_path.name}"


@pytest.mark.parametrize("raw_path", RAW_FIXTURES, ids=FIXTURE_IDS)
def test_real_answer_matches_saved_snapshot(raw_path: Path):
    raw = raw_path.read_text(encoding="utf-8")
    expected = _expected_path(raw_path).read_text(encoding="utf-8")
    assert mcp_server._render_markdown(raw) == expected


@pytest.mark.parametrize("raw_path", RAW_FIXTURES, ids=FIXTURE_IDS)
def test_real_answer_is_idempotent(raw_path: Path):
    rendered = mcp_server._render_markdown(raw_path.read_text(encoding="utf-8"))
    assert mcp_server._render_markdown(rendered) == rendered


@pytest.mark.parametrize("raw_path", RAW_FIXTURES, ids=FIXTURE_IDS)
def test_real_answer_has_no_latex_scaffolding(raw_path: Path):
    rendered = mcp_server._render_markdown(raw_path.read_text(encoding="utf-8"))
    for token in LATEX_SCAFFOLDING:
        assert token not in rendered


@pytest.mark.parametrize("raw_path", RAW_FIXTURES, ids=FIXTURE_IDS)
def test_real_answer_tables_are_aligned(raw_path: Path):
    rendered = mcp_server._render_markdown(raw_path.read_text(encoding="utf-8"))
    columns = [line for line in rendered.split("\n") if line.startswith(("┌", "│", "├", "└"))]
    if not columns:
        pytest.skip("fixture has no box table")
    widths = {mcp_server._display_width(line) for line in columns}
    assert len(widths) == 1, f"box table is ragged: {widths}"
