from __future__ import annotations

import asyncio
from collections.abc import Sequence

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

from deepseek_local_server.config import Settings


class BrowserManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._start_lock = asyncio.Lock()

    @property
    def started(self) -> bool:
        return self._context is not None

    @staticmethod
    def _live_pages(pages: Sequence[Page]) -> list[Page]:
        return [page for page in pages if not page.is_closed()]

    async def start(self) -> Page:
        async with self._start_lock:
            # OAuth/login flows may close the original tab and leave another tab in the same
            # persistent context. Reuse that tab instead of starting a second Chromium process.
            if self._context is not None:
                try:
                    pages = self._live_pages(self._context.pages)
                    if pages:
                        self._page = pages[-1]
                        return self._page

                    page = await self._context.new_page()
                    self._configure_page(page)
                    self._page = page
                    return page
                except Exception:
                    # The browser context itself was closed. Dispose stale Playwright state and
                    # reopen the persistent profile below.
                    await self._close_unlocked()

            self._settings.ensure_directories()
            self._playwright = await async_playwright().start()
            try:
                self._context = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(self._settings.profile_dir),
                    headless=self._settings.headless,
                    viewport={"width": 1440, "height": 1000},
                    locale="en-US",
                    args=["--disable-blink-features=AutomationControlled"],
                )
                pages = self._live_pages(self._context.pages)
                self._page = pages[-1] if pages else await self._context.new_page()
                self._configure_page(self._page)
                return self._page
            except Exception:
                await self._close_unlocked()
                raise

    def _configure_page(self, page: Page) -> None:
        page.set_default_timeout(15_000)

    async def _ensure_routing(self, page: Page) -> None:
        if self._settings.block_heavy_resources:
            # route() is idempotent enough for newly created pages, but callers only invoke this
            # once for each page returned by new_page().
            await page.route("**/*", self._route_resource)

    async def _route_resource(self, route) -> None:  # type: ignore[no-untyped-def]
        if route.request.resource_type in {"font", "media"}:
            await route.abort()
        else:
            await route.continue_()

    async def get_page(self) -> Page:
        page = await self.start()
        await self._ensure_routing(page)
        return page

    async def get_live_page(self, *, prefer_url_contains: str | None = None) -> Page | None:
        """Return a currently open page, preferring a URL fragment when supplied.

        Login providers may close the original page and replace it with another page. This method
        deliberately does not create a new page, which lets the caller distinguish a replaced tab
        from a fully closed browser window.
        """
        context = self._context
        if context is None:
            return None
        try:
            pages = self._live_pages(context.pages)
        except Exception:
            return None
        if not pages:
            return None
        if prefer_url_contains:
            needle = prefer_url_contains.lower()
            for page in reversed(pages):
                if needle in page.url.lower():
                    self._page = page
                    return page
        self._page = pages[-1]
        return self._page

    async def new_page(self) -> Page:
        await self.start()
        assert self._context is not None
        page = await self._context.new_page()
        self._configure_page(page)
        await self._ensure_routing(page)
        return page

    async def _close_unlocked(self) -> None:
        context, playwright = self._context, self._playwright
        self._page = None
        self._context = None
        self._playwright = None
        try:
            if context is not None:
                try:
                    await context.close()
                except Exception:
                    # It may already have been closed by the user or an OAuth flow.
                    pass
        finally:
            if playwright is not None:
                try:
                    await playwright.stop()
                except Exception:
                    pass

    async def close(self) -> None:
        async with self._start_lock:
            await self._close_unlocked()

    async def reset(self) -> Page:
        await self.close()
        return await self.get_page()
