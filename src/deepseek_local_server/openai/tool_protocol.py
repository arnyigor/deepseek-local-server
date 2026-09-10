from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class ParsedOutput:
    content: str
    tool_calls: list[dict[str, Any]]


def _coerce(obj: object, allowed: set[str]) -> list[dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    candidate = obj.get("tool_call") or obj.get("function_call")
    candidates = obj.get("tool_calls") if isinstance(obj.get("tool_calls"), list) else None
    if candidates is None and candidate is not None:
        candidates = [candidate]
    if not candidates:
        return []
    out = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        if not isinstance(name, str) or name not in allowed:
            continue
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                json.loads(args)
                arg_text = args
            except Exception:
                arg_text = json.dumps({"value": args}, ensure_ascii=False)
        else:
            arg_text = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        out.append({
            "id": f"call_{uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": arg_text},
        })
    return out


def parse_output(text: str, allowed_tools: set[str]) -> ParsedOutput:
    stripped = text.strip()
    if not allowed_tools:
        return ParsedOutput(stripped, [])
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.I)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    for raw in candidates:
        try:
            calls = _coerce(json.loads(raw), allowed_tools)
        except Exception:
            calls = []
        if calls:
            return ParsedOutput("", calls)
    return ParsedOutput(stripped, [])
