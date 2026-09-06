import asyncio
import json

from playwright.async_api import async_playwright

PROFILE = r"C:\Users\ArnyPC\AppData\Local\deepseek-local-server\browser-profile"

JS = r"""
() => {
  const needles = ['v3', 'r1', 'expert', 'deepthink', 'search', 'model'];
  const cyr = ['\u044d\u043a\u0441\u043f\u0435\u0440\u0442', '\u043c\u044b\u0448\u043b\u0435\u043d\u0438\u0435', '\u043f\u043e\u0438\u0441\u043a', '\u043c\u043e\u0434\u0435\u043b\u044c'];
  const out = { hits: [], topbar: null, htmlLang: document.documentElement.lang };
  const all = document.querySelectorAll('div, span, button, [role="button"]');
  for (const el of all) {
    if (el.children.length > 0) continue;
    const t = (el.innerText || '').trim();
    if (!t || t.length > 60) continue;
    const low = t.toLowerCase();
    const hit = needles.concat(cyr).some((n) => low.includes(n));
    if (!hit) continue;
    let chain = '';
    let node = el;
    for (let i = 0; i < 4 && node; i++) {
      chain = (node.className ? String(node.className).slice(0, 100) : node.tagName) + ' | ' + chain;
      node = node.parentElement;
    }
    out.hits.push({ text: t, tag: el.tagName, chain: chain });
  }
  const header = document.querySelector('header') || document.querySelector('nav');
  if (header) out.topbar = header.outerHTML.slice(0, 12000);
  return out;
}
"""


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

        info = await page.evaluate(JS)

        with open("probe_model.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print("lang:", info["htmlLang"], "hits:", len(info["hits"]))
        for h in info["hits"][:40]:
            print(repr(h["text"]), "|", h["tag"], "|", h["chain"][:150])
        await ctx.close()


asyncio.run(main())
