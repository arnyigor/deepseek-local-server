import pytest

from deepseek_local_server.errors import UnsupportedContentError
from deepseek_local_server.openai.content import content_to_text


def test_text_parts_are_joined() -> None:
    assert content_to_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"


def test_images_are_rejected() -> None:
    with pytest.raises(UnsupportedContentError):
        content_to_text([{"type": "image_url", "image_url": {"url": "x"}}])
