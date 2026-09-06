from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Sequence

import httpx
import uvicorn

from deepseek_local_server.api.app import create_app
from deepseek_local_server.auth import ensure_api_token, read_api_token, token_path_for_display
from deepseek_local_server.browser.manager import BrowserManager
from deepseek_local_server.browser.worker import DeepSeekBrowserWorker
from deepseek_local_server.config import Settings


def _settings() -> Settings:
    try:
        return Settings.from_env()
    except Exception as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc


def command_init(settings: Settings) -> int:
    token = ensure_api_token(settings)
    print("DeepSeek Local Server initialized.")
    print(f"Home:    {settings.home}")
    print(f"Profile: {settings.profile_dir}")
    print(f"Token:   {token_path_for_display(settings.token_file)}")
    print(f"Token length: {len(token)} characters")
    return 0


async def _auth_async(settings: Settings) -> None:
    visible = settings.with_headless(False)
    manager = BrowserManager(visible)
    worker = DeepSeekBrowserWorker(visible, manager)
    try:
        await worker.login_interactively()
        print("DeepSeek browser profile is ready.")
    finally:
        await manager.close()


def command_auth(settings: Settings) -> int:
    ensure_api_token(settings)
    asyncio.run(_auth_async(settings))
    return 0


def command_serve(settings: Settings) -> int:
    ensure_api_token(settings)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(f"OpenAI-compatible endpoint: {settings.api_base_url}/v1")
    print(f"Model: {settings.model_id}")
    print(f"Browser profile: {settings.profile_dir}")
    print(f"Headless: {settings.headless}")
    print(f"DeepThink (expert mode): {settings.deepthink_enabled}")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")
    return 0


def command_doctor(settings: Settings) -> int:
    print(f"Home: {settings.home}")
    print(f"Profile exists: {settings.profile_dir.exists()}")
    print(f"Token exists: {settings.token_file.exists()}")
    print(f"Endpoint: {settings.api_base_url}/v1")
    try:
        health = httpx.get(f"{settings.api_base_url}/health", timeout=3)
        health.raise_for_status()
        print(json.dumps(health.json(), indent=2, ensure_ascii=False))
        token = read_api_token(settings)
        models = httpx.get(
            f"{settings.api_base_url}/v1/models",
            headers={"Authorization": f"Bearer {token}"},
            timeout=3,
        )
        models.raise_for_status()
        print(json.dumps(models.json(), indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"Server check failed: {exc}")
        return 1
    return 0


def command_chat(settings: Settings, prompt: str, timeout: float) -> int:
    token = read_api_token(settings)
    response = httpx.post(
        f"{settings.api_base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": settings.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=timeout,
    )
    if response.is_error:
        print(response.text, file=sys.stderr)
        return 1
    payload = response.json()
    print(payload["choices"][0]["message"].get("content") or json.dumps(payload["choices"][0]["message"], ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepseek-local-server",
        description="Expose DeepSeek Web (DeepThink) as an OpenAI-compatible local model server",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create local directories and API token")
    sub.add_parser("auth", help="Open Chromium and save a DeepSeek login profile")
    sub.add_parser("serve", help="Start the OpenAI-compatible local server")
    sub.add_parser("doctor", help="Check server health and model discovery")

    chat = sub.add_parser("chat", help="Send a direct OpenAI-compatible test request")
    chat.add_argument("prompt")
    chat.add_argument("--timeout", type=float, default=330.0)

    sub.add_parser("mcp", help="Run the ask_deepseek MCP server over stdio")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    # Windows consoles default to cp866/cp1251 and crash on characters DeepSeek answers
    # routinely contain (zero-width spaces, CJK). Never let a pretty answer kill the CLI.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    args = build_parser().parse_args(argv)
    settings = _settings()
    if args.command == "init":
        code = command_init(settings)
    elif args.command == "auth":
        code = command_auth(settings)
    elif args.command == "serve":
        code = command_serve(settings)
    elif args.command == "doctor":
        code = command_doctor(settings)
    elif args.command == "chat":
        code = command_chat(settings, args.prompt, args.timeout)
    elif args.command == "mcp":
        from deepseek_local_server.mcp_server import main as mcp_main

        mcp_main()
        code = 0
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    raise SystemExit(code)
