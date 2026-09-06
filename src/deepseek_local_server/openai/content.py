from __future__ import annotations

from typing import Any

from deepseek_local_server.errors import UnsupportedContentError


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            part_type = part.get("type")
            if part_type in {"text", "input_text", "output_text"}:
                parts.append(str(part.get("text", "")))
            elif part_type in {"image_url", "input_image", "image"}:
                raise UnsupportedContentError(
                    "Image messages are not supported by the first DeepSeek Web adapter version"
                )
            else:
                text = part.get("text")
                if text is not None:
                    parts.append(str(text))
        return "\n".join(value for value in parts if value)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return str(content)
    return str(content)
