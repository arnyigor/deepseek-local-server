import os
from pathlib import Path

import pytest

from deepseek_local_server.config import Settings


def test_loopback_is_required(tmp_path: Path) -> None:
    Settings(home=tmp_path, host="127.0.0.1").validate()
    with pytest.raises(ValueError):
        Settings(home=tmp_path, host="0.0.0.0").validate()


def test_deepthink_defaults_to_enabled(tmp_path: Path) -> None:
    assert Settings(home=tmp_path).deepthink_enabled is True


def test_deepthink_env_var_disables_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DEEPSEEK_LOCAL_SERVER_HOME", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_LOCAL_SERVER_DEEPTHINK", "false")
    assert Settings.from_env().deepthink_enabled is False
