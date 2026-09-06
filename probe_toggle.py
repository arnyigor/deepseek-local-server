"""One-off probe: dump the composer-area toggle markup from the saved profile."""
import asyncio
import json

from playwright.async_api import async_playwright

PROFILE = r"C:\Users\ArnyPC\AppData\Local\deepseek-local-server\browser-profile"


async def main() -> None:
    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            PROFILE, headless=False, args=["--no-sandbox"]
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
              const out = { toggles: [], modelButtons: [], composerHtml: null };
              // every element that looks like a toggle switch
              for (const el of document.querySelectorAll('[aria-pressed], [role="switch"], [class*="toggle"]')) {
                const text = (el.innerText || '').trim().slice(0, 120);
                if (!text) continue;
                out.toggles.push({
                  tag: el.tagName,
                  cls: el.className.toString().slice(0, 200),
                  text,
                  ariaPressed: el.getAttribute('aria-pressed'),
                  role: el.getAttribute('role'),
                  checked: el.getAttribute('aria-checked'),
                });
              }
              // anything mentioning model names / expert / deepthink
              for (const el of document.querySelectorAll('button, [role="button"], [role="menuitemradio"], [role="menuitem"]')) {
                const t = (el.innerText || '').trim();
                if (/deepthink|expert|r1|v3|deepseek-|reason/i.test(t) && t.length < 200) {
                  out.modelButtons.push({
                    tag: el.tagName,
                    cls: el.className.toString().slice(0, 200),
                    text: t.slice(0, 200),
                    ariaPressed: el.getAttribute('aria-pressed'),
                    ariaSelected: el.getAttribute('aria-selected'),
                  });
                }
              }
              const ta = document.querySelector('textarea');
              if (ta) {
                let box = ta.closest('[class*="composer"], [class*="input"], [class*="editor"]') || ta.parentElement;
                for (let i = 0; i < 4 && box && box.parentElement; i++) box = box.parentElement;
                out.composerHtml = box ? box.outerHTML.slice(0, 30000) : ta.parentElement.outerHTML.slice(0, 30000);
              }
              return out;
            }"""
        )

        with open("probe_toggle.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print("toggles:", len(info["toggles"]))
        for t in info["toggles"]:
            print(json.dumps(t, ensure_ascii=False)[:300])
        print("modelButtons:", len(info["modelButtons"]))
        for b in info["modelButtons"]:
            print(json.dumps(b, ensure_ascii=False)[:300])
        await ctx.close()


asyncio.run(main())
