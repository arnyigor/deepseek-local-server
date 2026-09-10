"""Regenerate tests/fixtures/*.expected.txt from the saved raw answers (offline).

Use it after intentionally changing the renderer:

    .venv/Scripts/python.exe refresh_fixtures.py

Review the diff with git before committing.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from deepseek_local_server import mcp_server  # noqa: E402

FIXTURES = pathlib.Path(__file__).parent / "tests" / "fixtures"

refreshed = 0
for raw_path in sorted(FIXTURES.glob("*.raw.txt")):
    raw = raw_path.read_text(encoding="utf-8")
    expected_path = raw_path.with_suffix("").with_suffix(".expected.txt")
    expected_path.write_text(mcp_server._render_markdown(raw), encoding="utf-8")
    print(f"refreshed {expected_path.name}")
    refreshed += 1

print(f"{refreshed} fixture(s) refreshed")
