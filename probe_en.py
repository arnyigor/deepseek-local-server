import asyncio
import json

from playwright.async_api import async_playwright

PROFILE = r"C:\Users\ArnyPC\AppData\Local\deepseek-local-server\browser-profile"


async def main() -> None:
    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            PROFILE,
            headless=False,
            locale="en-US",
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto("https://chat.deepseek.com/", wait_until="domcontentloaded", timeout=60_000)
        try:
            await page.wait_for_load_state("networkidle", timeout=8_000)
        except Exception:
            pass
        await page.wait_for_timeout(3_000)

        info = await page.evaluate(
            """() => {
              const out = { lang: document.documentElement.lang, toggles: [], topText: '' };
              for (const el of document.querySelectorAll('[aria-pressed], [class*="toggle"]')) {
                const text = (el.innerText || '').trim().slice(0, 120);
                if (!text) continue;
                out.toggles.push({
                  cls: String(el.className).slice(0, 160),
                  text,
                  ariaPressed: el.getAttribute('aria-pressed'),
                });
              }
              // top-left area where model picker usually lives
              const top = document.querySelector('[class*="topbar"], [class*="header"], header');
              if (top) out.topText = (top.innerText || '').slice(0, 500);
              // any element whose text mentions model names
              const found = [];
              for (const el of document.querySelectorAll('div, span, button')) {
                if (el.children.length > 0) continue;
                const t = (el.innerText || '').trim();
                if (!t || t.length > 80) continue;
                if (/deepseek|r1|v3|expert|instant/i.test(t)) found.push(t);
              }
              out.modelTexts = Array.from(new Set(found)).slice(0, 30);
              return out;
            }"""
        )

        with open("probe_en.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(json.dumps(info, ensure_ascii=False, indent=1)[:2000])
        await ctx.close()


asyncio.run(main())
