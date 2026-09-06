from __future__ import annotations

import secrets
from pathlib import Path

from deepseek_local_server.config import Settings


def ensure_api_token(settings: Settings) -> str:
    settings.ensure_directories()
    if settings.token_file.exists():
        token = settings.token_file.read_text(encoding="utf-8").strip()
        if token:
            return token

    token = secrets.token_urlsafe(32)
    settings.token_file.write_text(token + "\n", encoding="utf-8")
    try:
        settings.token_file.chmod(0o600)
    except OSError:
        pass
    return token


def read_api_token(settings: Settings) -> str:
    if not settings.token_file.exists():
        raise FileNotFoundError("API token does not exist. Run `deepseek-local-server init`.")
    token = settings.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("API token file is empty")
    return token


def token_path_for_display(path: Path) -> str:
    return str(path)
