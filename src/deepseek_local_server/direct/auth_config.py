from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from deepseek_local_server.errors import AuthenticationRequiredError


@dataclass(frozen=True, slots=True)
class DeepSeekAuth:
    token: str
    cookie: str
    hif_dliq: str = ""
    hif_leim: str = ""
    wasm_url: str = ""
    user_agent: str = "Mozilla/5.0"
    client_version: str = "2.0.0"
    app_version: str = "2.0.0"
    locale: str = "en_US"
    timezone_offset: str = "0"

    def validate(self) -> None:
        if not self.token.strip():
            raise AuthenticationRequiredError("DeepSeek auth token is missing")
        if not self.cookie.strip():
            raise AuthenticationRequiredError("DeepSeek cookies are missing")
        if not self.wasm_url.strip():
            raise AuthenticationRequiredError("DeepSeek PoW WASM URL is missing")


def load_auth(path: Path) -> DeepSeekAuth:
    if not path.exists():
        raise AuthenticationRequiredError(f"DeepSeek auth file not found: {path}. Run `deepseek-local-server auth`.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuthenticationRequiredError(f"Could not parse DeepSeek auth file: {exc}") from exc
    auth = DeepSeekAuth(
        token=str(raw.get("token", "")),
        cookie=str(raw.get("cookie", "")),
        hif_dliq=str(raw.get("hif_dliq", raw.get("hifDliq", ""))),
        hif_leim=str(raw.get("hif_leim", raw.get("hifLeim", ""))),
        wasm_url=str(raw.get("wasmUrl", raw.get("wasm_url", ""))),
        user_agent=str(raw.get("userAgent", raw.get("user_agent", "Mozilla/5.0"))),
        client_version=str(raw.get("clientVersion", raw.get("client_version", "2.0.0"))),
        app_version=str(raw.get("appVersion", raw.get("app_version", "2.0.0"))),
        locale=str(raw.get("locale", "en_US")),
        timezone_offset=str(raw.get("timezoneOffset", raw.get("timezone_offset", "0"))),
    )
    auth.validate()
    return auth


def save_auth(path: Path, auth: DeepSeekAuth) -> None:
    auth.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token": auth.token,
        "hif_dliq": auth.hif_dliq,
        "hif_leim": auth.hif_leim,
        "cookie": auth.cookie,
        "wasmUrl": auth.wasm_url,
        "userAgent": auth.user_agent,
        "clientVersion": auth.client_version,
        "appVersion": auth.app_version,
        "locale": auth.locale,
        "timezoneOffset": auth.timezone_offset,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if os.name != "nt":
        try:
            path.chmod(0o600)
        except OSError:
            pass
