import pytest

from deepseek_local_server.models import resolve_model


def test_reasoning_mapping():
    spec = resolve_model("deepseek-reasoner")
    assert spec.model_type == "default"
    assert spec.thinking is True
    assert spec.search is False


def test_alias_mapping():
    # DeepSeek merged Instant/Expert/Vision into one model; old names alias to it.
    assert resolve_model("deepseek-v4-pro").id == "deepseek-reasoner"
    assert resolve_model("deepseek-expert").id == "deepseek-reasoner"
    assert resolve_model("deepseek-web").id == "deepseek-reasoner"
    assert resolve_model("deepseek-r1").id == "deepseek-reasoner"


def test_unknown_model():
    with pytest.raises(ValueError):
        resolve_model("nope")
