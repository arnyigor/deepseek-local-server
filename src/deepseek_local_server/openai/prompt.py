from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Iterable

from deepseek_local_server.openai.schemas import ChatCompletionRequest, ChatMessage


@dataclass(frozen=True, slots=True)
class ImagePart:
    data: bytes
    filename: str
    content_type: str


def extract_images(messages: Iterable[ChatMessage]) -> list[ImagePart]:
    """Pull inline base64 images (OpenAI vision `image_url` data URLs) out of messages.

    Remote (http/https) image URLs are not fetched here — DeepSeek's endpoint wants
    the raw bytes uploaded directly, and this gateway has no case yet that sends one.
    """
    images: list[ImagePart] = []
    for msg in messages:
        content = msg.content
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not isinstance(url, str) or not url.startswith("data:"):
                continue
            header, _, b64data = url.partition(",")
            content_type = header.removeprefix("data:").split(";")[0] or "image/png"
            try:
                data = base64.b64decode(b64data)
            except (ValueError, TypeError):
                continue
            ext = content_type.split("/")[-1] or "png"
            images.append(ImagePart(data, f"image.{ext}", content_type))
    return images


def _content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in {"text", "input_text"} and isinstance(part.get("text"), str):
                    out.append(part["text"])
                elif isinstance(part.get("content"), str):
                    out.append(part["content"])
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return json.dumps(content, ensure_ascii=False)


def build_prompt(request: ChatCompletionRequest, messages: Iterable[ChatMessage] | None = None) -> str:
    selected = list(messages if messages is not None else request.messages)
    sections: list[str] = []
    for msg in selected:
        text = _content_to_text(msg.content)
        if msg.role == "tool":
            sections.append(f"[TOOL RESULT {msg.tool_call_id or msg.name or ''}]\n{text}")
        else:
            sections.append(f"[{msg.role.upper()}]\n{text}")

    if request.tools:
        tool_defs = [
            {
                "name": t.function.name,
                "description": t.function.description or "",
                "parameters": t.function.parameters,
            }
            for t in request.tools
        ]
        sections.append(
            "[TOOL ADAPTER]\n"
            "You may call one of the following tools when needed. When you want a tool call, output ONLY strict JSON "
            'in this shape: {"tool_call":{"name":"tool_name","arguments":{...}}}. '
            "Do not wrap that JSON in markdown. Otherwise answer normally.\n"
            + json.dumps(tool_defs, ensure_ascii=False)
        )
    return "\n\n".join(sections).strip()
