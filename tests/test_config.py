from pathlib import Path
import pytest
from deepseek_local_server.config import Settings


def test_loopback_allowed(tmp_path: Path):
    Settings(home=tmp_path, host="127.0.0.1").validate()


def test_non_loopback_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        Settings(home=tmp_path, host="0.0.0.0").validate()
