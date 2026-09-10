import base64

from deepseek_local_server.openai.prompt import extract_images
from deepseek_local_server.openai.schemas import ChatMessage

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_extracts_inline_base64_image():
    url = "data:image/png;base64," + base64.b64encode(TINY_PNG).decode()
    messages = [ChatMessage(role="user", content=[
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": url}},
    ])]
    images = extract_images(messages)
    assert len(images) == 1
    assert images[0].data == TINY_PNG
    assert images[0].content_type == "image/png"
    assert images[0].filename == "image.png"


def test_ignores_remote_and_plain_text_content():
    messages = [
        ChatMessage(role="user", content="just text"),
        ChatMessage(role="user", content=[
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ]),
    ]
    assert extract_images(messages) == []
