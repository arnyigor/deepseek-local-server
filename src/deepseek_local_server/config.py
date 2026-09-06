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

    xdg_data_home = os.getenv("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "deepseek-local-server"

    return Path.home() / ".local" / "share" / "deepseek-local-server"


@dataclass(frozen=True, slots=True)
class Settings:
    home: Path
    host: str = "127.0.0.1"
    port: int = 9874
    deepseek_url: str = "https://chat.deepseek.com/"
    headless: bool = False
    request_timeout_seconds: float = 300.0
    stable_seconds: float = 2.0
    poll_interval_seconds: float = 0.35
    max_prompt_chars: int = 900_000
    model_id: str = "deepseek-web"
    model_name: str = "DeepSeek Web (DeepThink, local adapter)"
    context_window: int = 65_536
    max_output_tokens: int = 8_192
    block_heavy_resources: bool = True
    deepthink_enabled: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            home=_default_home(),
            host=os.getenv("DEEPSEEK_LOCAL_SERVER_HOST", "127.0.0.1"),
            port=int(os.getenv("DEEPSEEK_LOCAL_SERVER_PORT", "9874")),
            deepseek_url=os.getenv("DEEPSEEK_LOCAL_SERVER_DEEPSEEK_URL", "https://chat.deepseek.com/"),
            headless=_env_bool("DEEPSEEK_LOCAL_SERVER_HEADLESS", False),
            request_timeout_seconds=float(os.getenv("DEEPSEEK_LOCAL_SERVER_TIMEOUT_SECONDS", "300")),
            stable_seconds=float(os.getenv("DEEPSEEK_LOCAL_SERVER_STABLE_SECONDS", "2.0")),
            poll_interval_seconds=float(os.getenv("DEEPSEEK_LOCAL_SERVER_POLL_INTERVAL_SECONDS", "0.35")),
            max_prompt_chars=int(os.getenv("DEEPSEEK_LOCAL_SERVER_MAX_PROMPT_CHARS", "900000")),
            model_id=os.getenv("DEEPSEEK_LOCAL_SERVER_MODEL_ID", "deepseek-web"),
            model_name=os.getenv("DEEPSEEK_LOCAL_SERVER_MODEL_NAME", "DeepSeek Web (DeepThink, local adapter)"),
            context_window=int(os.getenv("DEEPSEEK_LOCAL_SERVER_CONTEXT_WINDOW", "65536")),
            max_output_tokens=int(os.getenv("DEEPSEEK_LOCAL_SERVER_MAX_OUTPUT_TOKENS", "8192")),
            block_heavy_resources=_env_bool("DEEPSEEK_LOCAL_SERVER_BLOCK_HEAVY_RESOURCES", True),
            deepthink_enabled=_env_bool("DEEPSEEK_LOCAL_SERVER_DEEPTHINK", True),
        )
        settings.validate()
        return settings

    def with_headless(self, headless: bool) -> "Settings":
        return replace(self, headless=headless)

    def validate(self) -> None:
        try:
            address = ipaddress.ip_address(self.host)
        except ValueError as exc:
            raise ValueError("DEEPSEEK_LOCAL_SERVER_HOST must be a literal loopback IP") from exc
        if not address.is_loopback:
            raise ValueError("DeepSeek Local Server only listens on a loopback address")
        if not 1 <= self.port <= 65535:
            raise ValueError("DEEPSEEK_LOCAL_SERVER_PORT must be between 1 and 65535")
        if self.stable_seconds <= 0 or self.poll_interval_seconds <= 0:
            raise ValueError("Polling and stability intervals must be positive")
        if self.request_timeout_seconds <= self.stable_seconds:
            raise ValueError("Request timeout must exceed stability interval")
        if self.max_prompt_chars < 10_000:
            raise ValueError("DEEPSEEK_LOCAL_SERVER_MAX_PROMPT_CHARS is too small")
        if not self.model_id.strip():
            raise ValueError("DEEPSEEK_LOCAL_SERVER_MODEL_ID cannot be blank")

    @property
    def profile_dir(self) -> Path:
        return self.home / "browser-profile"

    @property
    def token_file(self) -> Path:
        return self.home / "token"

    @property
    def debug_dir(self) -> Path:
        return self.home / "debug"

    @property
    def api_base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def ensure_directories(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
