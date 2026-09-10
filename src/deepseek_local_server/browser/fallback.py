from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from deepseek_local_server.browser.chrome import launch_and_connect, terminate
from deepseek_local_server.config import Settings
from deepseek_local_server.errors import AuthenticationRequiredError, BrowserProtocolError
from deepseek_local_server.models import ModelSpec

LOGGER = logging.getLogger("deepseek_local_server.browser_fallback")

ASSISTANT_SELECTOR = ".ds-markdown.ds-assistant-message-main-content"
COMPOSERS = ['textarea[placeholder="Message DeepSeek"]', "textarea:not([disabled])"]
SEND_SELECTOR = "div.ds-button--primary"
DEEPTHINK_SELECTOR = 'div.ds-toggle-button:has-text("DeepThink"), div.ds-toggle-button:has-text("Глубокое мышление")'


@dataclass(frozen=True, slots=True)
class BrowserAnswer:
    content: str


class BrowserFallback:
    """Serialized emergency fallback through the real DeepSeek UI.

    This is intentionally not the normal transport. It exists for periods where
    the private Web API contract changes but the public web UI still works.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()
        self._pw: Playwright | None = None
        self._proc: subprocess.Popen | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    async def _reset(self) -> None:
        terminate(self._proc)
        self._proc = None
        self._browser = None
        self._context = None
        if self._pw is not None:
            await self._pw.stop()
            self._pw = None

    async def _launch(self) -> None:
        self._pw = await async_playwright().start()
        self._proc, self._browser, self._context = await launch_and_connect(
            self._pw,
            self.settings.profile_dir,
            headless=self.settings.headless_fallback,
            port=self.settings.chrome_debug_port,
        )

    async def _page(self) -> Page:
        if self._browser is not None and not self._browser.is_connected():
            # Chrome can die between requests (crash, manual close, profile conflict).
            # Reconnecting to a fresh process is cheaper than failing every call forever.
            await self._reset()
        if self._context is None:
            await self._launch()
        try:
            page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        except Exception:
            await self._reset()
            await self._launch()
            page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        if "deepseek.com" not in page.url:
            await page.goto(self.settings.deepseek_url, wait_until="domcontentloaded", timeout=60_000)
        return page

    async def _composer(self, page: Page):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            for selector in COMPOSERS:
                loc = page.locator(selector).last
                if await loc.count() and await loc.is_visible():
                    return loc
            await asyncio.sleep(0.25)
        body = (await page.locator("body").inner_text()).lower()
        if "sign in" in body or "log in" in body:
            raise AuthenticationRequiredError("Browser fallback needs login. Run `deepseek-local-server auth`.")
        raise BrowserProtocolError("DeepSeek composer was not found; UI selectors may have changed")

    async def _new_chat(self, page: Page) -> None:
        """Best-effort reset so the fallback does not inherit stale browser context."""
        candidates = (
            'button[aria-label*="New chat" i]',
            '[title*="New chat" i]',
            'button:has-text("New chat")',
            'button:has-text("Новый чат")',
        )
        for selector in candidates:
            loc = page.locator(selector).first
            if await loc.count() and await loc.is_visible():
                try:
                    await loc.click(timeout=3000)
                    await asyncio.sleep(0.4)
                    return
                except Exception:
                    continue

    async def _configure_toggles(self, page: Page, spec: ModelSpec) -> None:
        # DeepSeek merged Instant/Expert/Vision into one model (2026-09); there is no
        # mode selector left to click, only the DeepThink toggle below.
        if spec.thinking:
            toggle = page.locator(DEEPTHINK_SELECTOR).first
            if await toggle.count() and await toggle.is_visible():
                if await toggle.get_attribute("aria-pressed") != "true":
                    try:
                        await toggle.click(timeout=3000)
                    except Exception:
                        pass

    async def query(self, prompt: str, spec: ModelSpec) -> BrowserAnswer:
        if spec.search:
            raise BrowserProtocolError("Browser fallback does not emulate web-search mode; direct backend is required")
        async with self._lock:
            page = await self._page()
            await self._new_chat(page)
            await self._configure_toggles(page, spec)
            composer = await self._composer(page)
            baseline = await page.locator(ASSISTANT_SELECTOR).count()
            await composer.fill(prompt)
            send = page.locator(SEND_SELECTOR).last
            if await send.count() and await send.is_visible():
                await send.click()
            else:
                await composer.press("Enter")

            deadline = time.monotonic() + self.settings.request_timeout_seconds
            last = ""
            stable_since: float | None = None
            while time.monotonic() < deadline:
                items = page.locator(ASSISTANT_SELECTOR)
                count = await items.count()
                if count > baseline:
                    current = (await items.last.inner_text()).strip()
                    if current:
                        if current == last:
                            stable_since = stable_since or time.monotonic()
                        else:
                            last = current
                            stable_since = None
                        if stable_since and time.monotonic() - stable_since >= self.settings.stable_seconds:
                            return BrowserAnswer(last)
                await asyncio.sleep(self.settings.poll_interval_seconds)
            raise BrowserProtocolError("Browser fallback timed out waiting for a stable assistant response")

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        self._context = None
        if self._pw is not None:
            await self._pw.stop()
            self._pw = None
        terminate(self._proc)
        self._proc = None
