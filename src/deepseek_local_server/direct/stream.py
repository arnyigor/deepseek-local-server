from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StreamPiece:
    kind: str  # content | reasoning
    text: str


class DeepSeekPatchParser:
    """Incrementally reconstructs DeepSeek Web's patch/SSE response.

    It deliberately accepts both full response snapshots and append-style patches,
    because the web contract has used both forms.
    """

    def __init__(self) -> None:
        self.fragments: list[dict[str, Any]] = []
        self.full_content = ""
        self.message_id: int | None = None
        self.finish_reason: str | None = None
        self.model_error: str | None = None
        self._emitted_content = ""
        self._emitted_reasoning = ""
        self._last_path: str | None = None

    @staticmethod
    def _append_fragment_list(target: list[dict[str, Any]], value: Any) -> None:
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, dict):
                target.append(dict(item))

    def _texts(self) -> tuple[str, str]:
        # SEARCH fragments are a "searching the web..." status marker (content is null,
        # the real text is a `queries` list) — not answer text, unlike RESPONSE fragments.
        response = "".join(
            str(f.get("content") or "") for f in self.fragments
            if str(f.get("type", "")).upper() == "RESPONSE"
        )
        reasoning = "".join(
            str(f.get("content") or "") for f in self.fragments
            if str(f.get("type", "")).upper() in {"THINK", "REASONING"}
        )
        if not response:
            response = self.full_content
        return response, reasoning

    @staticmethod
    def _suffix(previous: str, current: str) -> tuple[str, str]:
        if current.startswith(previous):
            return current[len(previous):], current
        # Upstream occasionally replaces a snapshot instead of appending. Avoid
        # silently dropping data; emit the current snapshot and restart tracking.
        return current, current

    def feed(self, event: dict[str, Any]) -> list[StreamPiece]:
        if event.get("response_message_id") is not None and not self.message_id:
            self.message_id = int(event["response_message_id"])
        if event.get("finish_reason") is not None:
            self.finish_reason = str(event["finish_reason"])
        if event.get("type") == "error":
            self.model_error = str(event.get("content", "DeepSeek model error"))

        # DeepSeek only repeats "p" on the first delta of a run; later deltas for the
        # same path arrive as bare {"v": ...}, implicitly continuing the last path.
        path = event.get("p")
        if path is not None:
            self._last_path = path
        else:
            path = self._last_path
        value = event.get("v")
        if isinstance(value, dict) and isinstance(value.get("response"), dict):
            response = value["response"]
            if response.get("message_id") is not None:
                self.message_id = int(response["message_id"])
            if response.get("content") is not None:
                self.full_content = str(response["content"])
            if isinstance(response.get("fragments"), list):
                self.fragments = [dict(x) for x in response["fragments"] if isinstance(x, dict)]
            if response.get("finish_reason") is not None:
                self.finish_reason = str(response["finish_reason"])

        if path == "response/fragments":
            self._append_fragment_list(self.fragments, value)
        elif path == "response/fragments/-1/content" and self.fragments and not isinstance(value, (dict, list)):
            self.fragments[-1]["content"] = str(self.fragments[-1].get("content", "")) + str(value)
        elif path == "response/content" and not isinstance(value, (dict, list)):
            self.full_content += str(value)
        elif path == "response/finish_reason" and value is not None:
            self.finish_reason = str(value)
        elif path == "response" and isinstance(value, list):
            for op in value:
                if not isinstance(op, dict):
                    continue
                if op.get("p") == "fragments" and str(op.get("o", "")).upper() == "APPEND":
                    self._append_fragment_list(self.fragments, op.get("v"))

        content, reasoning = self._texts()
        out: list[StreamPiece] = []
        r_delta, self._emitted_reasoning = self._suffix(self._emitted_reasoning, reasoning)
        c_delta, self._emitted_content = self._suffix(self._emitted_content, content)
        if r_delta:
            out.append(StreamPiece("reasoning", r_delta))
        if c_delta:
            out.append(StreamPiece("content", c_delta))
        return out

    @property
    def content(self) -> str:
        return self._texts()[0]

    @property
    def reasoning(self) -> str:
        return self._texts()[1]
