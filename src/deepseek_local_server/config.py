from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, replace
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_home() -> Path:
    explicit = os.getenv("DEEPSEEK_LOCAL_SERVER_HOME")
    if explicit:
        return Path(explicit).expanduser().resolve()
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "deepseek-local-server"
    xdg = os.getenv("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "deepseek-local-server"
    return Path.home() / ".local" / "share" / "deepseek-local-server"


@dataclass(frozen=True, slots=True)
class Settings:
    home: Path
    host: str = "127.0.0.1"
    port: int = 9874
    deepseek_url: str = "https://chat.deepseek.com/"
    chrome_debug_port: int = 9333
    request_timeout_seconds: float = 300.0
    fetch_timeout_seconds: float = 60.0
    max_prompt_chars: int = 200_000
    session_ttl_seconds: float = 7200.0
    max_session_messages: int = 100
    max_concurrent_direct: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls(
            home=_default_home(),
            host=os.getenv("DEEPSEEK_LOCAL_SERVER_HOST", "127.0.0.1"),
            port=int(os.getenv("DEEPSEEK_LOCAL_SERVER_PORT", "9874")),
            deepseek_url=os.getenv("DEEPSEEK_LOCAL_SERVER_DEEPSEEK_URL", "https://chat.deepseek.com/"),
            chrome_debug_port=int(os.getenv("DEEPSEEK_LOCAL_SERVER_CHROME_DEBUG_PORT", "9333")),
            request_timeout_seconds=float(os.getenv("DEEPSEEK_LOCAL_SERVER_TIMEOUT_SECONDS", "300")),
            fetch_timeout_seconds=float(os.getenv("DEEPSEEK_LOCAL_SERVER_FETCH_TIMEOUT_SECONDS", "60")),
            max_prompt_chars=int(os.getenv("DEEPSEEK_LOCAL_SERVER_MAX_PROMPT_CHARS", "200000")),
            session_ttl_seconds=float(os.getenv("DEEPSEEK_LOCAL_SERVER_SESSION_TTL_SECONDS", "7200")),
            max_session_messages=int(os.getenv("DEEPSEEK_LOCAL_SERVER_MAX_SESSION_MESSAGES", "100")),
            max_concurrent_direct=int(os.getenv("DEEPSEEK_LOCAL_SERVER_MAX_CONCURRENT_DIRECT", "8")),
        )
        s.validate()
        return s

    def validate(self) -> None:
        try:
            addr = ipaddress.ip_address(self.host)
        except ValueError as exc:
            raise ValueError("DEEPSEEK_LOCAL_SERVER_HOST must be a literal loopback IP") from exc
        if not addr.is_loopback:
            raise ValueError("DeepSeek Local Server only listens on a loopback address")
        if not 1 <= self.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        if not 1 <= self.chrome_debug_port <= 65535:
            raise ValueError("DEEPSEEK_LOCAL_SERVER_CHROME_DEBUG_PORT must be between 1 and 65535")
        if self.request_timeout_seconds <= 0 or self.fetch_timeout_seconds <= 0:
            raise ValueError("Timeouts must be positive")
        if self.max_prompt_chars < 10_000:
            raise ValueError("DEEPSEEK_LOCAL_SERVER_MAX_PROMPT_CHARS is too small")
        if self.max_session_messages < 1 or self.max_concurrent_direct < 1:
            raise ValueError("Session and concurrency limits must be positive")

    @property
    def api_base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def token_file(self) -> Path:
        return self.home / "token"

    @property
    def auth_file(self) -> Path:
        return self.home / "deepseek-auth.json"

    @property
    def profile_dir(self) -> Path:
        return self.home / "browser-profile"

    @property
    def debug_dir(self) -> Path:
        return self.home / "debug"

    @property
    def wasm_cache_dir(self) -> Path:
        return self.home / "wasm-cache"

    def ensure_directories(self) -> None:
        for p in (self.home, self.profile_dir, self.debug_dir, self.wasm_cache_dir):
            p.mkdir(parents=True, exist_ok=True)
