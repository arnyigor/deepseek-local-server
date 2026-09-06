"""
deepseek_probe.py — Phase 0 feasibility probe for driving chat.deepseek.com
the same way qwen_local_server drives chat.qwen.ai.

This is NOT a server. It does not blindly trust any single hardcoded
selector for DeepSeek's current UI. Instead it:

  1. Opens a *persistent* Chromium profile (separate from your Qwen one),
     so you log into DeepSeek once and it's remembered.
  2. Navigates to chat.deepseek.com.
  3. Runs generic heuristics to find composer candidates (textarea /
     contenteditable / role=textbox) and send-button candidates (by
     visible text, aria-label, title, position relative to the composer).
  4. Additionally flags any element matching a specific pattern found in a
     public (Oct-2025) bookmarklet for DeepSeek -- `input + div[role="button"]`
     -- as a *hint*, not a certainty: DeepSeek's DOM may well have changed
     since then, so this is scored as a bonus, not trusted blindly.
  5. Prints a ranked report of what it found, with attributes you can turn
     into real selectors.
  6. Saves a full-page screenshot and HTML dump for offline inspection.
  7. Optionally (--send "...") actually fills the best composer candidate,
     clicks the best button candidate, and watches the DOM for a new
     stable text block appearing -- proving (or disproving) the full
     round trip, the same way qwen_local_server's worker.py does.

Usage:

    python deepseek_probe.py                       # just look around, no send
    python deepseek_probe.py --send "Reply with exactly: PROBE_OK"
    python deepseek_probe.py --headless             # after first login
    python deepseek_probe.py --url https://chat.deepseek.com/

First run needs a visible window (do not pass --headless) so you can sign
into DeepSeek manually (email/phone + code, or however your account works);
the profile is saved for next time.

Login handshake, two modes:
  - Interactive terminal (a real TTY): the script prints a prompt and blocks
    on input() until you press Enter after signing in. If reading stdin hits
    EOF anyway (some agent/background-process launchers report stdin as a
    TTY even when it's actually redirected from /dev/null or NUL), it does
    NOT crash -- it falls back to the polling mode below instead.
  - Non-interactive stdin (launched by an agent, CI, or with stdin
    redirected from /dev/null or NUL): input() would raise EOFError here, so
    instead the script polls the page every 5s, for up to --login-timeout
    seconds (default 300), for a real chat input to appear -- no keypress,
    no shell piping tricks needed. Just leave the visible Chromium window
    open and log in there; the script picks it up on its own.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from playwright.async_api import async_playwright, Page, Locator
except ImportError:  # pragma: no cover
    print("Playwright is not installed. Run: pip install playwright && playwright install chromium")
    raise


# --------------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------------

def _default_home() -> Path:
    explicit = os.getenv("DEEPSEEK_PROBE_HOME")
    if explicit:
        return Path(explicit).expanduser().resolve()
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "deepseek-probe"
    xdg_data_home = os.getenv("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "deepseek-probe"
    return Path.home() / ".local" / "share" / "deepseek-probe"


HOME = _default_home()
PROFILE_DIR = HOME / "browser-profile"
DEBUG_DIR = HOME / "debug"

DEFAULT_URL = "https://chat.deepseek.com/"

# From a public (last touched ~Oct 2025) bookmarklet automating several chat
# sites. Treated as a scoring HINT only -- DeepSeek's DOM may have moved on.
KNOWN_SEND_BUTTON_HINT_SELECTORS = [
    'input + div[role="button"]',
]


# --------------------------------------------------------------------------
# Generic (non-hardcoded) element discovery
# --------------------------------------------------------------------------

COMPOSER_CANDIDATE_SELECTORS = [
    "textarea",
    '[contenteditable="true"]',
    '[role="textbox"]',
]

BUTTON_TEXT_HINTS = re.compile(r"send|submit|\u043e\u0442\u043f\u0440\u0430\u0432", re.IGNORECASE)  # "send/submit/otprav"
BUTTON_NEGATIVE_HINTS = re.compile(
    r"attach|upload|file|image|voice|microphone|record|search|settings|"
    r"model|stop|cancel|close|clear|delete|menu|more|help|feedback|share|"
    r"copy|download|history|new chat|deep ?think|search the web",
    re.IGNORECASE,
)

SNAPSHOT_JS = r"""
(args) => {
  const { composerSelectors, hintSelectors } = args;

  const visible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.visibility !== 'hidden' && style.display !== 'none' &&
      rect.width > 0 && rect.height > 0;
  };

  const describe = (el) => {
    const rect = el.getBoundingClientRect();
    return {
      tag: el.tagName.toLowerCase(),
      id: el.id || null,
      classes: (el.className && typeof el.className === 'string') ? el.className : null,
      role: el.getAttribute('role'),
      ariaLabel: el.getAttribute('aria-label'),
      title: el.getAttribute('title'),
      placeholder: el.getAttribute('placeholder'),
      dataTestId: el.getAttribute('data-test-id') || el.getAttribute('data-testid'),
      text: (el.innerText || el.textContent || '').trim().slice(0, 80),
      // DeepSeek toggles the send button's enabled state by adding/removing
      // a '--disabled' CLASS token (not aria-disabled), so check both.
      disabled: !!(el.disabled || el.getAttribute('aria-disabled') === 'true' || (typeof el.className === 'string' && /(^|\s)--disabled(\s|$)/.test(el.className))),
      rect: { x: Math.round(rect.left), y: Math.round(rect.top), w: Math.round(rect.width), h: Math.round(rect.height) },
    };
  };

  const composers = [];
  const seenComposers = new Set();
  for (const sel of composerSelectors) {
    for (const el of document.querySelectorAll(sel)) {
      if (seenComposers.has(el) || !visible(el)) continue;
      seenComposers.add(el);
      composers.push({ ...describe(el), matchedSelector: sel });
    }
  }

  const hintMatches = new Set();
  for (const sel of hintSelectors) {
    try {
      for (const el of document.querySelectorAll(sel)) hintMatches.add(el);
    } catch (e) { /* selector might not be supported/valid in this DOM state */ }
  }

  const buttons = [];
  for (const el of document.querySelectorAll('button, [role="button"]')) {
    if (!visible(el)) continue;
    buttons.push({ ...describe(el), matchesKnownHint: hintMatches.has(el) });
  }

  return { composers, buttons, url: location.href, title: document.title };
}
"""

MESSAGE_HEURISTIC_JS = r"""
() => {
  // No known selector for DeepSeek's message bubbles, so this just grabs
  // every leaf-ish block of visible text longer than a few words, as a
  // rough "what changed after I clicked send" signal. Refine once you've
  // seen real output.
  const visible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.visibility !== 'hidden' && style.display !== 'none' &&
      rect.width > 0 && rect.height > 0;
  };
  const blocks = [];
  const all = document.querySelectorAll('div, p, span');
  for (const el of all) {
    if (!visible(el)) continue;
    const text = (el.innerText || el.textContent || '').trim();
    if (text.length < 8) continue;
    let redundant = false;
    for (const child of el.children) {
      const childText = (child.innerText || child.textContent || '').trim();
      if (childText.length >= text.length * 0.9) { redundant = true; break; }
    }
    if (redundant) continue;
    const rect = el.getBoundingClientRect();
    blocks.push({ text: text.slice(0, 200), top: Math.round(rect.top + window.scrollY), tag: el.tagName.toLowerCase() });
  }
  blocks.sort((a, b) => a.top - b.top);
  return blocks;
}
"""


@dataclass
class ScoredElement:
    info: dict[str, Any]
    score: float
    reason: list[str] = field(default_factory=list)


def _score_composer(info: dict[str, Any]) -> ScoredElement:
    score = 0.0
    reasons = []
    rect = info["rect"]
    if rect["w"] > 200:
        score += 20
        reasons.append("wide (>200px)")
    if rect["h"] > 30:
        score += 10
        reasons.append("tall enough for multi-line")
    label = " ".join(filter(None, [info.get("ariaLabel"), info.get("placeholder"), info.get("title")])).lower()
    if any(k in label for k in ("message", "ask", "type", "send", "\u0441\u043f\u0440\u043e\u0441", "\u0441\u043e\u043e\u0431\u0449")):
        score += 30
        reasons.append(f"label hints at chat input: {label!r}")
    if info["tag"] == "textarea":
        score += 20
        reasons.append("is a plain <textarea> (matches the known DeepSeek pattern)")
    if info.get("disabled"):
        score -= 100
        reasons.append("disabled")
    score += min(rect["w"] * rect["h"], 500_000) / 20_000
    return ScoredElement(info, score, reasons)


def _score_button(info: dict[str, Any], composer_rect: dict[str, int] | None) -> ScoredElement:
    score = 0.0
    reasons = []
    label = " ".join(
        filter(None, [info.get("ariaLabel"), info.get("title"), info.get("dataTestId"), info.get("text")])
    )
    if BUTTON_NEGATIVE_HINTS.search(label):
        score -= 200
        reasons.append(f"negative-hint label: {label!r}")
    if BUTTON_TEXT_HINTS.search(label):
        score += 100
        reasons.append(f"positive-hint label: {label!r}")
    if info.get("matchesKnownHint"):
        score += 80
        reasons.append("matches known DeepSeek send-button pattern (input + div[role=button]) -- unverified, from a ~1yr-old public script")
    classes = info.get("classes") or ""
    if "primary" in classes:
        score += 60
        reasons.append(f"'primary' in classes (ds-button--primary) -- send buttons are usually the primary action: {classes[:60]!r}")
    if info.get("disabled"):
        # Mild penalty only: a send button is EXPECTED to be disabled while
        # the composer is empty (the snapshot is taken before we type
        # anything). Playwright's click() auto-waits for enabled state.
        score -= 40
        reasons.append("disabled (expected while composer is empty)")
    if composer_rect is not None:
        rect = info["rect"]
        # Distance to the NEAREST POINT OF the composer rect, not to its
        # top-left corner (that made a button at the composer's bottom-right
        # corner look ~750px away and lose to sidebar icons).
        cx0, cy0 = composer_rect["x"], composer_rect["y"]
        cx1, cy1 = composer_rect["x"] + composer_rect["w"], composer_rect["y"] + composer_rect["h"]
        dx = max(cx0 - rect["x"], 0, rect["x"] + rect["w"] - cx1)
        dy = max(cy0 - rect["y"], 0, rect["y"] + rect["h"] - cy1)
        distance = (dx * dx + dy * dy) ** 0.5
        proximity_score = max(0.0, 50 - distance / 20)
        score += proximity_score
        if proximity_score > 10:
            reasons.append(f"near composer (distance to nearest edge={distance:.0f}px)")
        # Chat send buttons conventionally sit at the composer's bottom-right.
        if rect["y"] >= cy0 and rect["x"] + rect["w"] >= cx1 - 80:
            score += 30
            reasons.append("sits at the composer's bottom-right (typical send-button position)")
    return ScoredElement(info, score, reasons)


# --------------------------------------------------------------------------
# Main probe flow
# --------------------------------------------------------------------------

async def _composer_present(page: Page) -> bool:
    """The actual thing we need, rather than fuzzy 'sign in' text matching:
    is there a *visible, reasonably sized* chat input on the page right now?

    Note: a plain count() > 0 is NOT enough -- the DeepSeek sign_in page
    carries a hidden textarea, which made a count-only check fire on the
    login screen. Require visibility plus a minimum size (a real chat
    composer is wide and multi-line; login fields are narrow inputs)."""
    for sel in COMPOSER_CANDIDATE_SELECTORS:
        try:
            loc = page.locator(sel)
            n = await loc.count()
            for i in range(min(n, 5)):
                el = loc.nth(i)
                try:
                    if not await el.is_visible():
                        continue
                    box = await el.bounding_box()
                except Exception:
                    continue
                if box and box["width"] > 100 and box["height"] > 20:
                    return True
        except Exception:
            continue
    return False


async def _poll_for_login(page: Page, login_timeout: float) -> None:
    print(f">>> Polling for up to {login_timeout:.0f}s for a chat input to appear "
          f"-- sign in in the visible Chromium window, this will pick it up automatically.")
    deadline = time.monotonic() + login_timeout
    started = time.monotonic()
    last_report = 0.0
    while time.monotonic() < deadline:
        await asyncio.sleep(5.0)
        elapsed = time.monotonic() - started
        if elapsed - last_report >= 30:
            print(f">>> Still waiting for login... ({int(elapsed)}s elapsed, timeout {login_timeout:.0f}s)")
            last_report = elapsed
        if await _composer_present(page):
            print(">>> Chat input detected, continuing.")
            return
    print(f">>> Timed out after {login_timeout:.0f}s waiting for login. Continuing anyway -- "
          f"the snapshot below will likely still show the login page.")


async def _ensure_login(page: Page, url: str, login_timeout: float) -> None:
    if await _composer_present(page):
        return

    print("\n>>> No chat input detected yet -- you probably need to sign in to DeepSeek in this profile.")
    print(">>> A visible Chromium window should be open. Sign in there (email/phone + code).")

    if sys.stdin.isatty():
        # Interactive run: wait for an explicit Enter, same as before.
        try:
            await asyncio.to_thread(input, ">>> Press Enter here once you're signed in and see the chat UI... ")
            if "deepseek.com" not in page.url:
                await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            return
        except EOFError:
            # isatty() is not always trustworthy: some agent/background-process
            # launchers report stdin as a TTY even though it's actually
            # redirected from /dev/null or NUL, so input() hits EOF instead of
            # blocking. Don't crash -- fall through to polling like the
            # non-interactive branch below.
            print(">>> stdin claimed to be a terminal but reading it hit EOF "
                  "(seen under some agent/background-process launchers) -- "
                  "falling back to polling instead of waiting for Enter.")
    else:
        print(">>> No interactive terminal detected -- polling instead of waiting for Enter.")

    # Reached either because stdin isn't a TTY, or because it claimed to be
    # one but reading it hit EOF anyway.
    await _poll_for_login(page, login_timeout)


async def _snapshot(page: Page) -> dict[str, Any]:
    return await page.evaluate(
        SNAPSHOT_JS,
        {"composerSelectors": COMPOSER_CANDIDATE_SELECTORS, "hintSelectors": KNOWN_SEND_BUTTON_HINT_SELECTORS},
    )


def _print_report(snapshot: dict[str, Any]) -> tuple[ScoredElement | None, ScoredElement | None]:
    print(f"\n=== Page: {snapshot['title']!r} @ {snapshot['url']} ===")

    composers = [_score_composer(info) for info in snapshot["composers"]]
    composers.sort(key=lambda s: s.score, reverse=True)
    print(f"\n--- Composer candidates: {len(composers)} ---")
    for scored in composers[:8]:
        info = scored.info
        print(
            f"  score={scored.score:6.1f}  <{info['tag']}> "
            f"id={info['id']!r} class={str(info['classes'])[:60]!r} "
            f"aria-label={info['ariaLabel']!r} placeholder={info['placeholder']!r} "
            f"rect={info['rect']} matchedSelector={info.get('matchedSelector')!r}"
        )
        for reason in scored.reason:
            print(f"      - {reason}")

    best_composer = composers[0] if composers else None
    composer_rect = best_composer.info["rect"] if best_composer else None

    buttons = [_score_button(info, composer_rect) for info in snapshot["buttons"]]
    buttons.sort(key=lambda s: s.score, reverse=True)
    print(f"\n--- Button candidates (top 10 of {len(buttons)}) ---")
    for scored in buttons[:10]:
        info = scored.info
        print(
            f"  score={scored.score:6.1f}  <{info['tag']}> "
            f"aria-label={info['ariaLabel']!r} title={info['title']!r} "
            f"text={info['text']!r} data-test-id={info['dataTestId']!r} "
            f"knownHint={info.get('matchesKnownHint')} rect={info['rect']}"
        )
        for reason in scored.reason:
            print(f"      - {reason}")

    best_button = buttons[0] if buttons else None

    if best_composer:
        print(f"\n>>> Best composer guess: <{best_composer.info['tag']}> "
              f"aria-label={best_composer.info['ariaLabel']!r}")
    else:
        print("\n>>> No composer candidate found at all -- page may not have loaded, "
              "or the input lives inside an <iframe>/shadow DOM this probe doesn't pierce.")

    if best_button and best_button.score > 0:
        print(f">>> Best button guess: aria-label={best_button.info['ariaLabel']!r} "
              f"title={best_button.info['title']!r} text={best_button.info['text']!r} "
              f"knownHint={best_button.info.get('matchesKnownHint')}")
    else:
        print(">>> No confident send button candidate. Check the button list above manually.")

    return best_composer, best_button


async def _locator_for(page: Page, info: dict[str, Any]) -> Locator | None:
    """Best-effort: rebuild a Playwright locator for an element we only have
    a JS-side description of, using whatever is most specific."""
    if info.get("dataTestId"):
        loc = page.locator(f'[data-test-id="{info["dataTestId"]}"], [data-testid="{info["dataTestId"]}"]').first
        if await loc.count():
            return loc
    if info.get("ariaLabel"):
        loc = page.get_by_label(info["ariaLabel"]).first
        if await loc.count():
            return loc
        loc = page.locator(f'[aria-label="{info["ariaLabel"]}"]').first
        if await loc.count():
            return loc
    if info.get("matchesKnownHint"):
        for sel in KNOWN_SEND_BUTTON_HINT_SELECTORS:
            loc = page.locator(sel).first
            if await loc.count():
                return loc
    classes = info.get("classes")
    if classes:
        # DeepSeek's hashed class names are stable within a deploy, BUT the
        # class list mutates after interaction (e.g. the send button loses
        # ds-button--disabled once text is typed), so try selectors in
        # decreasing order of specificity:
        #   1. tag + ALL class tokens (exact match at snapshot time)
        #   2. tag + only the stable design-system 'ds-*' tokens
        cls_tokens = classes.split()
        full_sel = f"{info['tag']}" + "".join(f".{c}" for c in cls_tokens)
        ds_tokens = [c for c in cls_tokens if c.startswith("ds-")]
        ds_sel = f"{info['tag']}" + "".join(f".{c}" for c in ds_tokens) if ds_tokens else None
        for sel in filter(None, (full_sel, ds_sel)):
            loc = page.locator(sel).first
            if await loc.count():
                return loc
    if info.get("id"):
        loc = page.locator(f'#{info["id"]}').first
        if await loc.count():
            return loc
    if info["tag"] == "textarea":
        loc = page.locator("textarea").first
        if await loc.count():
            return loc
    if info.get("text"):
        loc = page.get_by_text(info["text"], exact=False).first
        if await loc.count():
            return loc
    return None


async def _try_send(page: Page, composer: ScoredElement, button: ScoredElement | None, prompt: str) -> None:
    print(f"\n=== Attempting a real round trip with prompt: {prompt!r} ===")
    composer_loc = await _locator_for(page, composer.info)
    if composer_loc is None:
        print("!!! Could not rebuild a locator for the chosen composer. Aborting send attempt.")
        return

    before = await page.evaluate(MESSAGE_HEURISTIC_JS)
    before_texts = {b["text"] for b in before}

    await composer_loc.click()
    await composer_loc.fill(prompt)
    await asyncio.sleep(0.3)

    submitted = False
    if button is not None and button.score > 0:
        button_loc = await _locator_for(page, button.info)
        if button_loc is not None:
            try:
                # The send button starts disabled (empty composer) and only
                # becomes enabled after we type -- wait for that explicitly.
                await button_loc.wait_for(state="visible", timeout=5_000)
                await button_loc.click(timeout=5_000)
                submitted = True
                print("--- Clicked the best button candidate.")
            except Exception as exc:  # noqa: BLE001 -- probe script, report and fall back
                print(f"--- Click failed ({exc!r}), falling back to keyboard.")

    if not submitted:
        for shortcut in ("Enter", "Control+Enter"):
            try:
                await composer_loc.press(shortcut)
                print(f"--- Sent via keyboard shortcut: {shortcut}")
                submitted = True
                break
            except Exception:
                continue

    if not submitted:
        print("!!! Could not submit the prompt by any method tried.")
        return

    print("--- Waiting up to 60s for new text to appear on the page...")
    deadline = time.monotonic() + 60
    new_text = None
    while time.monotonic() < deadline:
        await asyncio.sleep(1.5)
        current = await page.evaluate(MESSAGE_HEURISTIC_JS)
        fresh = [b for b in current if b["text"] not in before_texts and prompt[:30] not in b["text"]]
        if fresh:
            new_text = fresh
            await asyncio.sleep(2.0)
            current2 = await page.evaluate(MESSAGE_HEURISTIC_JS)
            fresh2 = [b for b in current2 if b["text"] not in before_texts and prompt[:30] not in b["text"]]
            if len(fresh2) == len(fresh) and all(a["text"] == b["text"] for a, b in zip(fresh, fresh2)):
                break

    if new_text:
        print("\n=== New content detected after send (candidate response blocks) ===")
        for block in new_text[:5]:
            print(f"  <{block['tag']}> top={block['top']}  {block['text']!r}")
        print("\n>>> If this looks like DeepSeek's reply, the round trip works end-to-end.")
    else:
        print("\n!!! No new stable text block detected within 60s. Either the response is still "
              "streaming/thinking, or the message heuristic missed the real response container "
              "-- check the saved screenshot/HTML.")


async def run(args: argparse.Namespace) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=args.headless,
            viewport={"width": 1440, "height": 1000},
            locale="en-US",
            args=["--disable-blink-features=AutomationControlled"],
        )
        try:
            pages = [p for p in context.pages if not p.is_closed()]
            page = pages[-1] if pages else await context.new_page()
            page.set_default_timeout(15_000)

            print(f"--- Navigating to {args.url}")
            await page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
            try:
                await page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass

            await _ensure_login(page, args.url, args.login_timeout)
            try:
                await page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass

            snapshot = await _snapshot(page)
            best_composer, best_button = _print_report(snapshot)

            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            screenshot_path = DEBUG_DIR / f"probe-{stamp}.png"
            html_path = DEBUG_DIR / f"probe-{stamp}.html"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            html_path.write_text(await page.content(), encoding="utf-8")
            print(f"\n--- Saved screenshot to {screenshot_path}")
            print(f"--- Saved HTML dump to {html_path}")

            if args.send:
                if best_composer is None:
                    print("\n!!! Skipping send attempt: no composer candidate found.")
                else:
                    await _try_send(page, best_composer, best_button, args.send)
                    stamp2 = datetime.now().strftime("%Y%m%d-%H%M%S")
                    await page.screenshot(path=str(DEBUG_DIR / f"probe-after-send-{stamp2}.png"), full_page=True)
                    (DEBUG_DIR / f"probe-after-send-{stamp2}.html").write_text(
                        await page.content(), encoding="utf-8"
                    )

            if not args.headless:
                print("\n--- Leaving the browser open for 20s so you can look around manually too...")
                await asyncio.sleep(20)
        finally:
            await context.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL, help="DeepSeek page to open")
    parser.add_argument("--send", default=None, help="If set, actually try to submit this prompt and watch for a reply")
    parser.add_argument("--headless", action="store_true", help="Run headless (only after you've already logged in once)")
    parser.add_argument(
        "--login-timeout",
        type=float,
        default=300.0,
        help="Seconds to poll for login when stdin is not interactive (e.g. run by an agent/harness). Ignored when stdin is a real TTY -- that case waits for Enter instead.",
    )
    return parser


def main() -> None:
    # Windows console defaults to cp1251 and crashes on CJK/emoji in the
    # model's reply; force UTF-8 with replacement for anything unmappable.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 -- not all streams support reconfigure
            pass
    args = build_parser().parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
