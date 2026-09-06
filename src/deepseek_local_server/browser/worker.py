from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from playwright.async_api import (
    Error as PlaywrightError,
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)

from deepseek_local_server.browser.dom import (
    ASSISTANT_SELECTORS,
    COMPOSER_SELECTORS,
    DEEPTHINK_TOGGLE_SELECTOR,
    EXPERT_MODEL_TEXTS,
    NEW_CHAT_SELECTORS,
    PAGE_ERROR_JS,
    SEND_SELECTORS,
    SNAPSHOT_MESSAGES_JS,
    STOP_SELECTORS,
    USER_MESSAGE_COUNT_SELECTOR,
    clean_assistant_text,
    is_junk_snapshot_text,
)
from deepseek_local_server.browser.manager import BrowserManager
from deepseek_local_server.config import Settings
from deepseek_local_server.errors import AuthenticationRequiredError, BrowserProtocolError, DeepSeekPageError

LOGGER = logging.getLogger("deepseek_local_server.browser")
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class BrowserResult:
    answer: str
    browser_url: str
    partial: bool = False


class DeepSeekBrowserWorker:
    def __init__(self, settings: Settings, manager: BrowserManager) -> None:
        self._settings = settings
        self._manager = manager

    async def login_interactively(self) -> None:
        page = await self._manager.get_page()
        await page.goto(self._settings.deepseek_url, wait_until="domcontentloaded", timeout=60_000)
        print("\nA Chromium window has been opened.")
        print("Sign in to DeepSeek and wait until the chat input is visible.")
        print("Do not close the Chromium window. The persistent profile is saved automatically.")
        await asyncio.to_thread(
            input,
            "Return to this terminal and press Enter while Chromium is still open... ",
        )

        page = await self._manager.get_live_page(prefer_url_contains="deepseek.com")
        if page is None:
            print("Chromium was closed or replaced. Reopening the saved profile for verification...")
            page = await self._manager.reset()

        try:
            if "deepseek.com" not in page.url.lower():
                await page.goto(self._settings.deepseek_url, wait_until="domcontentloaded", timeout=60_000)
            await self._find_composer(page, timeout_ms=30_000)
        except (PlaywrightError, BrowserProtocolError):
            if not page.is_closed():
                raise
            print("The login tab changed during verification. Reopening the saved profile once...")
            page = await self._manager.reset()
            await page.goto(self._settings.deepseek_url, wait_until="domcontentloaded", timeout=60_000)
            await self._find_composer(page, timeout_ms=30_000)

    async def query(
        self,
        *,
        request_id: str,
        prompt: str,
        timeout_seconds: float,
        progress: ProgressCallback | None = None,
        page: Page | None = None,
        keep_open: bool = False,
    ) -> tuple[BrowserResult, Page]:
        reuse = page is not None
        if not reuse:
            self._progress(progress, "opening_browser_page")
            page = await self._manager.new_page()
        assert page is not None
        success = False
        try:
            if not reuse:
                LOGGER.info("[%s] opening DeepSeek page", request_id)
                await page.goto(self._settings.deepseek_url, wait_until="domcontentloaded", timeout=60_000)
                try:
                    # The chat SPA finishes hydrating (event handlers wired up) shortly after its
                    # initial API calls settle. Acting before that point silently no-ops clicks/keys.
                    await page.wait_for_load_state("networkidle", timeout=8_000)
                except PlaywrightTimeoutError:
                    pass
                await self._try_start_new_chat(page)
            await self._ensure_expert(page)
            await self._ensure_deepthink(page)
            composer = await self._find_composer(page, timeout_ms=20_000)
            self._progress(progress, "composer_ready")

            baseline = await self._message_snapshot(page, ASSISTANT_SELECTORS)
            baseline_last = baseline[-1]["text"] if baseline else ""
            baseline_message_count = await self._count(page, USER_MESSAGE_COUNT_SELECTOR)

            await self._fill_composer(composer, prompt)
            self._progress(progress, "composer_filled")
            LOGGER.info("[%s] composer filled with %d characters", request_id, len(prompt))

            submit_method = await self._submit_prompt(
                page=page,
                composer=composer,
                baseline_assistants=len(baseline),
                baseline_message_count=baseline_message_count,
            )
            self._progress(progress, f"submitted:{submit_method}")
            LOGGER.info("[%s] prompt submitted via %s", request_id, submit_method)

            self._progress(progress, "waiting_for_deepseek")
            result = await self._wait_for_response(
                page=page,
                request_id=request_id,
                baseline_count=len(baseline),
                baseline_last=baseline_last,
                timeout_seconds=timeout_seconds,
            )
            self._progress(progress, "response_received")
            success = True
            return result, page
        except AuthenticationRequiredError:
            raise
        except Exception:
            await self._capture_debug(page, request_id)
            raise
        finally:
            if not (success and keep_open):
                try:
                    await page.close()
                except Exception:
                    pass

    @staticmethod
    def _progress(callback: ProgressCallback | None, stage: str) -> None:
        if callback is not None:
            callback(stage)

    async def _try_start_new_chat(self, page: Page) -> None:
        for selector in NEW_CHAT_SELECTORS:
            locator = page.locator(selector).first
            try:
                if await locator.count() and await locator.is_visible():
                    await locator.click(timeout=2_000)
                    await asyncio.sleep(0.5)
                    return
            except Exception:
                continue

    async def _ensure_expert(self, page: Page) -> None:
        # The top-bar model picker (Instant / Expert) decides which model answers; the
        # DeepThink toggle alone does not switch models. Selecting Expert is idempotent,
        # so no state pre-check is needed.
        try:
            for label in EXPERT_MODEL_TEXTS:
                option = page.get_by_text(label, exact=True).first
                if await option.count():
                    await option.click(timeout=3_000)
                    await asyncio.sleep(0.3)
                    LOGGER.info("Expert model selected (%s)", label)
                    return
            LOGGER.info("Expert model option not found in the model picker")
        except Exception:
            # Model preference, not a hard requirement; never break the send flow over it.
            LOGGER.info("could not select Expert model", exc_info=True)

    async def _ensure_deepthink(self, page: Page) -> None:
        if not self._settings.deepthink_enabled:
            return
        try:
            toggle = page.locator(DEEPTHINK_TOGGLE_SELECTOR).first
            if not await toggle.count():
                return
            if (await toggle.get_attribute("aria-pressed")) == "true":
                return
            await toggle.click(timeout=3_000)
            await asyncio.sleep(0.3)
            state = await toggle.get_attribute("aria-pressed")
            if state != "true":
                LOGGER.info("DeepThink toggle did not confirm as pressed (state=%r)", state)
        except Exception:
            # DeepThink is a preference, not a requirement; never break the send flow over it.
            LOGGER.info("could not enable DeepThink toggle", exc_info=True)

    async def _find_composer(self, page: Page, timeout_ms: int) -> Locator:
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            for selector in COMPOSER_SELECTORS:
                locator = page.locator(selector).last
                try:
                    if await locator.count() and await locator.is_visible():
                        return locator
                except PlaywrightTimeoutError:
                    continue
                except PlaywrightError as exc:
                    if page.is_closed():
                        raise BrowserProtocolError(
                            "The DeepSeek tab was closed while waiting for the chat input."
                        ) from exc
                    continue
            await asyncio.sleep(0.25)

        if page.is_closed():
            raise BrowserProtocolError(
                "The DeepSeek tab was closed while waiting for the chat input. "
                "Run `deepseek-local-server auth` again and keep Chromium open until verification completes."
            )

        try:
            body = (await page.locator("body").inner_text()).lower()
        except PlaywrightError as exc:
            if page.is_closed():
                raise BrowserProtocolError(
                    "The DeepSeek tab was closed while verifying authentication."
                ) from exc
            raise
        if "sign in" in body or "log in" in body:
            raise AuthenticationRequiredError("DeepSeek login is required. Run `deepseek-local-server auth`.")
        raise BrowserProtocolError("Could not find the DeepSeek chat input. The web UI may have changed.")

    async def _fill_composer(self, composer: Locator, prompt: str) -> None:
        await composer.scroll_into_view_if_needed()
        await composer.click()
        try:
            await composer.fill(prompt)
        except PlaywrightError:
            # Some rich-text editors do not implement fill() consistently. Update the native
            # value/content and dispatch input/change events in one browser-side operation; this
            # avoids one Playwright command per character for large prompts.
            await composer.evaluate(
                """(el, value) => {
                  el.focus();
                  if (el instanceof HTMLTextAreaElement || el instanceof HTMLInputElement) {
                    const proto = el instanceof HTMLTextAreaElement
                      ? HTMLTextAreaElement.prototype
                      : HTMLInputElement.prototype;
                    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
                    if (setter) setter.call(el, value); else el.value = value;
                  } else {
                    el.textContent = value;
                  }
                  el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: value }));
                  el.dispatchEvent(new Event('change', { bubbles: true }));
                }""",
                prompt,
            )

        actual = await self._composer_text(composer)
        if not actual.strip():
            raise BrowserProtocolError("DeepSeek composer remained empty after inserting the prompt.")
        # Large rich-text editors may normalize whitespace, so only enforce a coarse size check.
        if len(prompt) >= 100 and len(actual) < min(80, len(prompt) // 4):
            raise BrowserProtocolError(
                f"DeepSeek composer accepted only {len(actual)} of {len(prompt)} prompt characters."
            )

    async def _composer_text(self, composer: Locator) -> str:
        try:
            tag = (await composer.evaluate("el => el.tagName.toLowerCase()")) or ""
            if tag in {"textarea", "input"}:
                return await composer.input_value()
            return await composer.evaluate(
                "el => (el.innerText || el.textContent || '').replace(/\\u00a0/g, ' ')"
            )
        except PlaywrightError:
            return ""

    async def _submit_prompt(
        self,
        *,
        page: Page,
        composer: Locator,
        baseline_assistants: int,
        baseline_message_count: int,
    ) -> str:
        # The send control is a styled <div>, not a real <button>; try it, but Enter is the
        # proven-reliable fallback (see plan.md).
        button = await self._find_send_button(page)
        if button is not None:
            try:
                await button.click(timeout=5_000)
                if await self._wait_until_submitted(
                    page, baseline_assistants, baseline_message_count, timeout_seconds=5.0
                ):
                    return "send_button"
            except PlaywrightError:
                pass

        for shortcut in ("Control+Enter", "Enter"):
            try:
                await composer.focus()
                await composer.press(shortcut)
                if await self._wait_until_submitted(
                    page, baseline_assistants, baseline_message_count, timeout_seconds=3.0
                ):
                    return shortcut.lower().replace("+", "_")
            except PlaywrightError:
                continue

        raise BrowserProtocolError(
            "The prompt was inserted into DeepSeek, but the web UI did not submit it. "
            "A screenshot and HTML snapshot were saved."
        )

    async def _find_send_button(self, page: Page) -> Locator | None:
        for selector in SEND_SELECTORS:
            locator = page.locator(selector).last
            try:
                if await locator.count() and await locator.is_visible():
                    return locator
            except PlaywrightError:
                continue
        return None

    async def _wait_until_submitted(
        self,
        page: Page,
        baseline_assistants: int,
        baseline_message_count: int,
        *,
        timeout_seconds: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if await self._generation_running(page):
                return True
            assistants = await self._message_snapshot(page, ASSISTANT_SELECTORS)
            if len(assistants) > baseline_assistants:
                return True
            if await self._count(page, USER_MESSAGE_COUNT_SELECTOR) > baseline_message_count:
                return True
            await asyncio.sleep(0.2)
        return False

    async def _count(self, page: Page, selector: str) -> int:
        try:
            return await page.locator(selector).count()
        except PlaywrightError:
            return 0

    async def _message_snapshot(self, page: Page, selectors: list[str]) -> list[dict[str, Any]]:
        result = await page.evaluate(SNAPSHOT_MESSAGES_JS, {"selectors": selectors})
        if not isinstance(result, list):
            return []
        return [
            item
            for item in result
            if isinstance(item, dict) and item.get("text") and not is_junk_snapshot_text(item["text"])
        ]

    async def _generation_running(self, page: Page) -> bool:
        for selector in STOP_SELECTORS:
            locator = page.locator(selector)
            try:
                for index in range(await locator.count()):
                    if await locator.nth(index).is_visible():
                        return True
            except PlaywrightTimeoutError:
                continue
        return False

    async def _wait_for_response(
        self,
        *,
        page: Page,
        request_id: str,
        baseline_count: int,
        baseline_last: str,
        timeout_seconds: float,
    ) -> BrowserResult:
        deadline = time.monotonic() + timeout_seconds
        last_candidate = ""
        stable_since: float | None = None
        saw_new_candidate = False

        while time.monotonic() < deadline:
            error_kind = await page.evaluate(PAGE_ERROR_JS)
            if error_kind == "auth":
                raise AuthenticationRequiredError("DeepSeek authentication expired. Run `deepseek-local-server auth`.")
            if error_kind == "deepseek":
                raise DeepSeekPageError("DeepSeek displayed a generation or network error")

            snapshot = await self._message_snapshot(page, ASSISTANT_SELECTORS)
            candidate = clean_assistant_text(snapshot[-1]["text"]) if snapshot else ""
            is_new = bool(candidate) and (len(snapshot) > baseline_count or candidate != baseline_last)

            if is_new:
                saw_new_candidate = True
                if candidate == last_candidate:
                    stable_since = stable_since or time.monotonic()
                else:
                    last_candidate = candidate
                    stable_since = None

                stable = stable_since is not None and time.monotonic() - stable_since >= self._settings.stable_seconds
                if stable and not await self._generation_running(page):
                    return BrowserResult(answer=last_candidate, browser_url=page.url)

            await asyncio.sleep(self._settings.poll_interval_seconds)

        await self._capture_debug(page, request_id)
        if saw_new_candidate and last_candidate:
            raise BrowserProtocolError(
                "DeepSeek generation timed out after producing partial text. "
                "The partial response was discarded; debug artifacts were saved."
            )
        raise BrowserProtocolError("No new assistant response was detected before timeout. Debug artifacts were saved.")

    async def _capture_debug(self, page: Page, request_id: str) -> None:
        self._settings.debug_dir.mkdir(parents=True, exist_ok=True)
        base = self._settings.debug_dir / request_id
        try:
            await page.screenshot(path=str(base.with_suffix(".png")), full_page=True)
        except Exception:
            pass
        try:
            base.with_suffix(".html").write_text(await page.content(), encoding="utf-8")
        except Exception:
            pass
