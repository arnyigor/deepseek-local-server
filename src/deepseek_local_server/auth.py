from __future__ import annotations

import secrets
from pathlib import Path

from deepseek_local_server.config import Settings


def ensure_api_token(settings: Settings) -> str:
    settings.ensure_directories()
    if settings.token_file.exists():
        value = settings.token_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    token = secrets.token_urlsafe(48)
    settings.token_file.write_text(token + "\n", encoding="utf-8")
    return token


def read_api_token(settings: Settings) -> str:
    value = settings.token_file.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("Local API token is empty")
    return value


def token_path_for_display(path: Path) -> str:
    return str(path)
