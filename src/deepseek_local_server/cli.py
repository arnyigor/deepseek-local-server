from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import httpx
import uvicorn

from deepseek_local_server.api.app import create_app
from deepseek_local_server.auth import ensure_api_token, read_api_token
from deepseek_local_server.browser.auth_capture import capture_auth
from deepseek_local_server.browser.chrome import resolve_chrome_path
from deepseek_local_server.config import Settings
from deepseek_local_server.direct.auth_config import load_auth
from deepseek_local_server.direct.client import DeepSeekDirectClient


def _settings() -> Settings:
    try:
        return Settings.from_env()
    except Exception as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc


def command_init(settings: Settings) -> int:
    token = ensure_api_token(settings)
    print("DeepSeek Local Server v2 initialized.")
    print(f"Home:      {settings.home}")
    print(f"API token: {settings.token_file} ({len(token)} chars)")
    print(f"Auth file: {settings.auth_file}")
    return 0


def command_auth(settings: Settings) -> int:
    ensure_api_token(settings)
    auth = asyncio.run(capture_auth(settings))
    print("DeepSeek Web auth captured successfully.")
    print(f"Saved: {settings.auth_file}")
    print(f"Token: {'yes' if auth.token else 'no'}; cookie: {'yes' if auth.cookie else 'no'}; wasm: {'yes' if auth.wasm_url else 'no'}")
    return 0


def command_serve(settings: Settings) -> int:
    ensure_api_token(settings)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(f"Endpoint: {settings.api_base_url}/v1")
    print("Primary:  direct DeepSeek Web API")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")
    return 0


def command_doctor(settings: Settings, online: bool) -> int:
    print("=== Local files ===")
    print(f"Home: {settings.home}")
    try:
        print(f"System Chrome: {resolve_chrome_path()}")
    except Exception as exc:
        print(f"System Chrome: FAIL: {exc}")
    print(f"Token file: {settings.token_file} ({'OK' if settings.token_file.exists() else 'MISSING'})")
    print(f"Auth file:  {settings.auth_file} ({'OK' if settings.auth_file.exists() else 'MISSING'})")
    auth_ok = False
    try:
        auth = load_auth(settings.auth_file)
        auth_ok = True
        print(f"DeepSeek auth: OK (cookie={bool(auth.cookie)}, wasm={bool(auth.wasm_url)})")
    except Exception as exc:
        print(f"DeepSeek auth: FAIL: {exc}")

    if online and auth_ok:
        print("\n=== Direct backend probe ===")
        probe = asyncio.run(DeepSeekDirectClient(settings).probe())
        print(json.dumps(probe, indent=2, ensure_ascii=False))

    print("\n=== Running server ===")
    try:
        health = httpx.get(f"{settings.api_base_url}/health", timeout=2)
        health.raise_for_status()
        print(json.dumps(health.json(), indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"Server not reachable: {exc}")
    return 0 if auth_ok else 1


def command_chat(settings: Settings, prompt: str, model: str, timeout: float) -> int:
    token = read_api_token(settings)
    response = httpx.post(
        f"{settings.api_base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}", "x-agent-session": "cli"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
        timeout=timeout,
    )
    if response.is_error:
        print(response.text, file=sys.stderr)
        return 1
    payload = response.json()
    msg = payload["choices"][0]["message"]
    if msg.get("reasoning_content"):
        print("[reasoning]\n" + msg["reasoning_content"] + "\n")
    print(msg.get("content") or json.dumps(msg.get("tool_calls"), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deepseek-local-server", description="Hybrid local gateway for DeepSeek Web")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("auth")
    sub.add_parser("serve")
    d = sub.add_parser("doctor")
    d.add_argument("--offline", action="store_true")
    c = sub.add_parser("chat")
    c.add_argument("prompt")
    c.add_argument("--model", default="deepseek-reasoner")
    c.add_argument("--timeout", type=float, default=330)
    sub.add_parser("mcp")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = _settings()
    if args.command == "init":
        code = command_init(settings)
    elif args.command == "auth":
        code = command_auth(settings)
    elif args.command == "serve":
        code = command_serve(settings)
    elif args.command == "doctor":
        code = command_doctor(settings, online=not args.offline)
    elif args.command == "chat":
        code = command_chat(settings, args.prompt, args.model, args.timeout)
    elif args.command == "mcp":
        from deepseek_local_server.mcp_server import main as mcp_main
        mcp_main()
        code = 0
    else:
        raise SystemExit(2)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
