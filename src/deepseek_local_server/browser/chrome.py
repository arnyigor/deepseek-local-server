from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
from playwright.async_api import Browser, BrowserContext, Playwright

from deepseek_local_server.errors import BrowserProtocolError


def resolve_chrome_path() -> str:
    """Find a real, already-installed Chrome/Chromium executable.

    Deliberately avoids Playwright's own bundled browser download: driving the
    user's existing Chrome over CDP means there is nothing to fetch and nothing
    tied to a specific Playwright package revision.
    """
    env_path = os.getenv("DEEPSEEK_LOCAL_SERVER_CHROME_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    candidates: list[str] = []
    if sys.platform == "win32":
        local_app_data = os.getenv("LOCALAPPDATA", "")
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.join(local_app_data, "Google", "Chrome", "Application", "chrome.exe") if local_app_data else "",
        ]
    elif sys.platform == "darwin":
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium"):
            found = shutil.which(name)
            if found:
                candidates.append(found)

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate

    raise BrowserProtocolError(
        "No system Chrome/Chromium found. Install Google Chrome, or set "
        "DEEPSEEK_LOCAL_SERVER_CHROME_PATH to the executable path."
    )


async def _wait_for_devtools(port: int, timeout: float = 20.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    async with httpx.AsyncClient() as client:
        while asyncio.get_running_loop().time() < deadline:
            try:
                response = await client.get(f"http://127.0.0.1:{port}/json/version", timeout=1.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.25)
    raise BrowserProtocolError(f"Chrome DevTools endpoint on port {port} did not start in time")


def _launch_chrome_process(chrome_path: str, profile_dir: Path, *, headless: bool, port: int) -> subprocess.Popen:
    args = [
        chrome_path,
        f"--user-data-dir={profile_dir}",
        f"--remote-debugging-port={port}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-features=Translate",
    ]
    if headless:
        args.append("--headless=new")
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" and headless else 0
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)


async def launch_and_connect(
    playwright: Playwright, profile_dir: Path, *, headless: bool, port: int
) -> tuple[subprocess.Popen, Browser, BrowserContext]:
    """Launch the user's real Chrome with a debug port and drive it over CDP.

    Replaces ``chromium.launch_persistent_context`` (which requires Playwright's
    own downloaded browser build) with ``connect_over_cdp`` against a Chrome
    process we spawn ourselves, so no ``playwright install`` is ever required.
    """
    profile_dir.mkdir(parents=True, exist_ok=True)
    chrome_path = resolve_chrome_path()
    proc = _launch_chrome_process(chrome_path, profile_dir, headless=headless, port=port)
    try:
        await _wait_for_devtools(port)
        browser = await playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
    except Exception:
        proc.terminate()
        raise
    context = browser.contexts[0] if browser.contexts else await browser.new_context()
    return proc, browser, context


def terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        pass
