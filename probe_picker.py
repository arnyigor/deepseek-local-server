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
              const out = { candidates: [] };
              for (const el of document.querySelectorAll('div, span, button, [role="button"]')) {
                if (el.children.length > 0) continue;
                const t = (el.innerText || '').trim();
                if (t !== 'Instant' && t !== 'Expert') continue;
                let chain = [];
                let node = el;
                for (let i = 0; i < 6 && node; i++) {
                  chain.push({
                    tag: node.tagName,
                    cls: String(node.className || '').slice(0, 150),
                    role: node.getAttribute('aria-label') || node.getAttribute('title') || '',
                    ariaHasPopup: node.getAttribute('aria-haspopup') || '',
                  });
                  node = node.parentElement;
                }
                out.candidates.push({ text: t, chain });
              }
              return out;
            }"""
        )

        with open("probe_picker.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        for c in info["candidates"]:
            print("==", c["text"])
            for step in c["chain"]:
                print("   ", step)
        await ctx.close()


asyncio.run(main())
