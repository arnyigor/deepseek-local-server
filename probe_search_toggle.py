"""One-off probe: find the web-search ("联网") toggle near the DeepSeek composer.

Run with the server STOPPED (profile lock):
    .venv\\Scripts\\python.exe probe_search_toggle.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from playwright.async_api import async_playwright

PROFILE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(os.environ["LOCALAPPDATA"]) / "deepseek-local-server" / "browser-profile"
)
URL = "https://chat.deepseek.com/"

DUMP_JS = r"""
() => {
  const visible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.visibility !== 'hidden' && style.display !== 'none' &&
      rect.width > 0 && rect.height > 0;
  };
  const out = [];
  // 1) All toggle-ish elements anywhere on the page.
  for (const el of document.querySelectorAll('[aria-pressed], [role="switch"], [class*="toggle"], [class*="switch"]')) {
    if (!visible(el)) continue;
    const rect = el.getBoundingClientRect();
    out.push({
      kind: 'toggle',
      tag: el.tagName.toLowerCase(),
      classes: (typeof el.className === 'string') ? el.className : null,
      ariaPressed: el.getAttribute('aria-pressed'),
      role: el.getAttribute('role'),
      ariaLabel: el.getAttribute('aria-label'),
      title: el.getAttribute('title'),
      dataTestId: el.getAttribute('data-test-id') || el.getAttribute('data-testid'),
      text: (el.innerText || el.textContent || '').trim().slice(0, 60),
      rect: { x: Math.round(rect.left), y: Math.round(rect.top), w: Math.round(rect.width), h: Math.round(rect.height) },
    });
  }
  // 2) All visible buttons with labels mentioning search/web/联网.
  for (const el of document.querySelectorAll('button, [role="button"]')) {
    if (!visible(el)) continue;
    const text = (el.innerText || el.textContent || '').trim();
    const label = [el.getAttribute('aria-label'), el.getAttribute('title'), text].filter(Boolean).join(' | ');
    if (/search|web|联网|网络|browse|internet/i.test(label)) {
      const rect = el.getBoundingClientRect();
      out.push({
        kind: 'search-button',
        tag: el.tagName.toLowerCase(),
        classes: (typeof el.className === 'string') ? el.className : null,
        ariaPressed: el.getAttribute('aria-pressed'),
        ariaLabel: el.getAttribute('aria-label'),
        title: el.getAttribute('title'),
        dataTestId: el.getAttribute('data-test-id') || el.getAttribute('data-testid'),
        text: text.slice(0, 60),
        rect: { x: Math.round(rect.left), y: Math.round(rect.top), w: Math.round(rect.width), h: Math.round(rect.height) },
      });
    }
  }
  // 3) Composer location for reference.
  const composer = document.querySelector('textarea[placeholder="Message DeepSeek"]')
    || document.querySelector('textarea:not([disabled])');
  if (composer && visible(composer)) {
    const rect = composer.getBoundingClientRect();
    out.push({ kind: 'composer', rect: { x: Math.round(rect.left), y: Math.round(rect.top), w: Math.round(rect.width), h: Math.round(rect.height) } });
  }
  // 4) Full HTML of the composer's ancestor toolbar (up to 4 levels), to catch
  //    icon-only toggles that carry no text/aria at all.
  if (composer) {
    let node = composer;
    for (let i = 0; i < 4 && node.parentElement; i++) node = node.parentElement;
    out.push({ kind: 'toolbar-html', html: node.outerHTML.slice(0, 12000) });
  }
  return out;
}
"""


async def main() -> None:
    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=True,
            viewport={"width": 1440, "height": 1000},
            locale="en-US",
            args=["--disable-blink-features=AutomationControlled"],
        )
        try:
            pages = [p for p in context.pages if not p.is_closed()]
            page = pages[-1] if pages else await context.new_page()
            page.set_default_timeout(15_000)
            await page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
            try:
                await page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass
            # Wait for the composer to hydrate.
            for _ in range(40):
                if await page.locator('textarea[placeholder="Message DeepSeek"]').count():
                    break
                await asyncio.sleep(0.5)
            data = await page.evaluate(DUMP_JS)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        finally:
            await context.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(1)
