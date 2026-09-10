from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import AsyncIterator
from uuid import uuid4

from deepseek_local_server.config import Settings
from deepseek_local_server.direct.client import DeepSeekDirectClient
from deepseek_local_server.models import ModelSpec, resolve_model
from deepseek_local_server.openai.prompt import build_prompt, extract_images
from deepseek_local_server.openai.responses import Usage, estimate_tokens
from deepseek_local_server.openai.schemas import ChatCompletionRequest, ChatMessage
from deepseek_local_server.openai.tool_protocol import ParsedOutput, parse_output
from deepseek_local_server.session import AgentSession, SessionStore

LOGGER = logging.getLogger("deepseek_local_server.service")


@dataclass(frozen=True, slots=True)
class CompletionResult:
    completion_id: str
    model: str
    parsed: ParsedOutput
    reasoning: str
    usage: Usage
    duration_ms: int
    backend: str


class CompletionService:
    def __init__(self, settings: Settings, direct: DeepSeekDirectClient | None = None) -> None:
        self.settings = settings
        self.direct = direct or DeepSeekDirectClient(settings)
        self.sessions = SessionStore()
        self.request_count = 0
        self.last_request_id: str | None = None
        self.last_request_at: str | None = None
        self.last_backend: str | None = None
        self.last_error: str | None = None

    def runtime_status(self) -> dict[str, object]:
        return {
            "request_count": self.request_count,
            "last_request_id": self.last_request_id,
            "last_request_at": self.last_request_at,
            "last_backend": self.last_backend,
            "last_error": self.last_error,
        }

    def _begin(self) -> str:
        request_id = uuid4().hex
        self.request_count += 1
        self.last_request_id = request_id
        self.last_request_at = datetime.now(UTC).isoformat()
        self.last_error = None
        return request_id

    def _prompt_for_locked_session(
        self, request: ChatCompletionRequest, session: AgentSession
    ) -> tuple[str, list[ChatMessage]]:
        delta = session.continuation_delta(request.messages)
        if delta is None:
            # A new/replaced history must not continue an unrelated remote chain.
            session.remote.reset()
            messages_used = list(request.messages)
        else:
            messages_used = delta
        prompt = build_prompt(request, messages=messages_used)
        if len(prompt) > self.settings.max_prompt_chars:
            raise ValueError(f"Serialized prompt has {len(prompt)} chars; limit is {self.settings.max_prompt_chars}")
        return prompt, messages_used

    async def _upload_images(self, messages: list[ChatMessage], spec: ModelSpec) -> list[str]:
        file_ids = []
        for image in extract_images(messages):
            file_ids.append(await self.direct.upload_image(image.data, image.filename, image.content_type, spec))
        return file_ids

    async def complete(self, request: ChatCompletionRequest, session_key: str) -> CompletionResult:
        request_id = self._begin()
        started = time.perf_counter()
        spec = resolve_model(request.model)
        session = await self.sessions.get(session_key)
        allowed = {tool.function.name for tool in request.tools or []}
        content = ""
        reasoning = ""
        backend = "direct"

        async with session.lock:
            prompt, messages_used = self._prompt_for_locked_session(request, session)
            fresh_prompt = build_prompt(request)
            try:
                file_ids = await self._upload_images(messages_used, spec)
                async for piece in self.direct.stream(
                    prompt, spec, session.remote, fresh_prompt=fresh_prompt, ref_file_ids=file_ids
                ):
                    if piece.kind == "reasoning":
                        reasoning += piece.text
                    else:
                        content += piece.text
                session.last_messages = list(request.messages)
                self.last_backend = "direct"
            except Exception as exc:
                LOGGER.warning("[%s] direct backend failed: %s", request_id, exc, exc_info=True)
                self.last_error = f"{type(exc).__name__}: {exc}"
                raise

        parsed = parse_output(content, allowed)
        usage = Usage(
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(parsed.content or content),
            reasoning_tokens=estimate_tokens(reasoning),
        )
        return CompletionResult(
            completion_id=f"chatcmpl-{request_id}",
            model=request.model,
            parsed=parsed,
            reasoning=reasoning,
            usage=usage,
            duration_ms=int((time.perf_counter() - started) * 1000),
            backend=backend,
        )

    async def stream(self, request: ChatCompletionRequest, session_key: str) -> AsyncIterator[dict]:
        """Yield OpenAI delta payloads as DeepSeek Web produces them.

        Requests with tools are intentionally buffered so textual tool markup cannot leak
        before it is converted into OpenAI tool_calls.
        """
        if request.tools:
            result = await self.complete(request, session_key)
            if result.reasoning:
                yield {"delta": {"reasoning_content": result.reasoning}, "finish_reason": None, "backend": result.backend}
            if result.parsed.tool_calls:
                yield {
                    "delta": {"role": "assistant", "content": None, "tool_calls": result.parsed.tool_calls},
                    "finish_reason": None,
                    "backend": result.backend,
                }
                yield {"delta": {}, "finish_reason": "tool_calls", "backend": result.backend, "usage": result.usage}
            else:
                if result.parsed.content:
                    yield {"delta": {"content": result.parsed.content}, "finish_reason": None, "backend": result.backend}
                yield {"delta": {}, "finish_reason": "stop", "backend": result.backend, "usage": result.usage}
            return

        request_id = self._begin()
        spec = resolve_model(request.model)
        session = await self.sessions.get(session_key)
        content = ""
        reasoning = ""
        backend = "direct"

        async with session.lock:
            prompt, messages_used = self._prompt_for_locked_session(request, session)
            fresh_prompt = build_prompt(request)
            try:
                file_ids = await self._upload_images(messages_used, spec)
                async for piece in self.direct.stream(
                    prompt, spec, session.remote, fresh_prompt=fresh_prompt, ref_file_ids=file_ids
                ):
                    if piece.kind == "reasoning":
                        reasoning += piece.text
                        yield {"delta": {"reasoning_content": piece.text}, "finish_reason": None, "backend": "direct"}
                    else:
                        content += piece.text
                        yield {"delta": {"content": piece.text}, "finish_reason": None, "backend": "direct"}
                session.last_messages = list(request.messages)
                self.last_backend = "direct"
            except Exception as exc:
                LOGGER.warning("[%s] direct stream failed: %s", request_id, exc, exc_info=True)
                self.last_error = f"{type(exc).__name__}: {exc}"
                raise

        usage = Usage(
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(content),
            reasoning_tokens=estimate_tokens(reasoning),
        )
        yield {"delta": {}, "finish_reason": "stop", "backend": backend, "usage": usage}
