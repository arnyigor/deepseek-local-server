from __future__ import annotations

import asyncio

from playwright.async_api import async_playwright

from deepseek_local_server.browser.chrome import launch_and_connect, terminate
from deepseek_local_server.config import Settings
from deepseek_local_server.direct.auth_config import DeepSeekAuth, save_auth


async def capture_auth(settings: Settings) -> DeepSeekAuth:
    """Capture the exact auth headers used by the user's own logged-in web session."""
    settings.ensure_directories()
    found: dict[str, str] = {}
    event = asyncio.Event()

    async with async_playwright() as pw:
        proc, browser, context = await launch_and_connect(
            pw, settings.profile_dir, headless=False, port=settings.chrome_debug_port
        )
        page = context.pages[0] if context.pages else await context.new_page()

        async def inspect_request(request) -> None:
            url = request.url
            if url.endswith(".wasm") and "sha3" in url.lower():
                found["wasmUrl"] = url
            if "/api/v0/chat/completion" not in url:
                return
            try:
                headers = await request.all_headers()
            except Exception:
                return
            auth_header = headers.get("authorization", "")
            found["token"] = auth_header.removeprefix("Bearer ").strip()
            found["cookie"] = headers.get("cookie", "")
            found["hif_dliq"] = headers.get("x-hif-dliq", "")
            found["hif_leim"] = headers.get("x-hif-leim", "")
            found["userAgent"] = headers.get("user-agent", "Mozilla/5.0")
            found["clientVersion"] = headers.get("x-client-version", "2.0.0")
            found["appVersion"] = headers.get("x-app-version", "2.0.0")
            found["locale"] = headers.get("x-client-locale", "en_US")
            found["timezoneOffset"] = headers.get("x-client-timezone-offset", "0")
            if found.get("token") and found.get("cookie"):
                event.set()

        def on_request(request) -> None:
            asyncio.create_task(inspect_request(request))

        page.on("request", on_request)
        await page.goto(settings.deepseek_url, wait_until="domcontentloaded", timeout=60_000)
        print("\nDeepSeek opened in Chromium.")
        print("1) Log in if needed.")
        print("2) Send one short message such as: AUTH_OK")
        print("3) The terminal will capture the request automatically.\n")
        try:
            await asyncio.wait_for(event.wait(), timeout=600)
        except TimeoutError as exc:
            raise RuntimeError("No DeepSeek completion request was captured within 10 minutes") from exc

        if not found.get("wasmUrl"):
            try:
                entries = await page.evaluate("performance.getEntriesByType('resource').map(x => x.name)")
                for url in entries:
                    if isinstance(url, str) and url.endswith(".wasm") and "sha3" in url.lower():
                        found["wasmUrl"] = url
                        break
            except Exception:
                pass
        if not found.get("wasmUrl"):
            raise RuntimeError("Auth headers were captured, but the DeepSeek SHA3 PoW WASM URL was not found")

        auth = DeepSeekAuth(
            token=found.get("token", ""),
            cookie=found.get("cookie", ""),
            hif_dliq=found.get("hif_dliq", ""),
            hif_leim=found.get("hif_leim", ""),
            wasm_url=found.get("wasmUrl", ""),
            user_agent=found.get("userAgent", "Mozilla/5.0"),
            client_version=found.get("clientVersion", "2.0.0"),
            app_version=found.get("appVersion", "2.0.0"),
            locale=found.get("locale", "en_US"),
            timezone_offset=found.get("timezoneOffset", "0"),
        )
        save_auth(settings.auth_file, auth)
        await browser.close()
        terminate(proc)
        return auth
