from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from playwright.async_api import Page

from deepseek_local_server.browser.manager import BrowserManager
from deepseek_local_server.browser.worker import DeepSeekBrowserWorker
from deepseek_local_server.config import Settings
from deepseek_local_server.errors import ToolProtocolError
from deepseek_local_server.openai.prompt import build_prompt
from deepseek_local_server.openai.responses import Usage, estimate_tokens
from deepseek_local_server.openai.schemas import ChatCompletionRequest, ChatMessage
from deepseek_local_server.openai.tool_protocol import ParsedAssistantOutput, parse_assistant_output

LOGGER = logging.getLogger("deepseek_local_server.service")


@dataclass(frozen=True, slots=True)
class CompletionResult:
    completion_id: str
    model: str
    parsed: ParsedAssistantOutput
    usage: Usage
    duration_ms: int
    browser_url: str
    partial: bool


class CompletionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.manager = BrowserManager(settings)
        self.worker = DeepSeekBrowserWorker(settings, self.manager)
        self._lock = asyncio.Lock()
        self.request_count = 0
        self.last_request_id: str | None = None
        self.last_request_at: str | None = None
        self.last_stage = "idle"
        self.last_error: str | None = None
        self._session_page: Page | None = None
        self._session_messages: list[ChatMessage] | None = None

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def _set_stage(self, request_id: str, stage: str) -> None:
        self.last_request_id = request_id
        self.last_stage = stage
        LOGGER.info("[%s] stage=%s", request_id, stage)

    def runtime_status(self) -> dict[str, object]:
        return {
            "request_count": self.request_count,
            "last_request_id": self.last_request_id,
            "last_request_at": self.last_request_at,
            "last_stage": self.last_stage,
            "last_error": self.last_error,
            "session_active": self._session_page is not None and not self._session_page.is_closed(),
        }

    def _continuation_delta(self, messages: list[ChatMessage]) -> list[ChatMessage] | None:
        """Return the new trailing messages if `messages` extends the last session, else None."""
        prev = self._session_messages
        if prev is None or self._session_page is None or self._session_page.is_closed():
            return None
        if len(messages) <= len(prev) or messages[: len(prev)] != prev:
            return None
        return messages[len(prev):]

    async def _close_session(self) -> None:
        page, self._session_page = self._session_page, None
        self._session_messages = None
        if page is not None:
            try:
                await page.close()
            except Exception:
                pass

    async def complete(self, request: ChatCompletionRequest) -> CompletionResult:
        request_id = uuid4().hex
        self.request_count += 1
        self.last_request_id = request_id
        self.last_request_at = datetime.now(UTC).isoformat()
        self.last_error = None
        self._set_stage(request_id, "received")
        LOGGER.info(
            "[%s] completion received: model=%s stream=%s messages=%d tools=%d",
            request_id,
            request.model,
            request.stream,
            len(request.messages),
            len(request.tools or []),
        )

        try:
            if request.model != self.settings.model_id:
                raise ValueError(f"Unknown model {request.model!r}; expected {self.settings.model_id!r}")

            allowed_tools = {tool.function.name for tool in request.tools or []}
            started = time.perf_counter()
            self._set_stage(request_id, "queued")
            async with self._lock:
                self._set_stage(request_id, "browser")

                delta = self._continuation_delta(request.messages)
                if delta is None and self._session_page is not None:
                    await self._close_session()

                if delta is not None:
                    prompt = build_prompt(
                        request,
                        messages=delta,
                        start_index=len(request.messages) - len(delta),
                        continuation=True,
                    )
                    page_arg = self._session_page
                else:
                    prompt = build_prompt(request)
                    page_arg = None

                LOGGER.info(
                    "[%s] serialized prompt characters=%d continuation=%s",
                    request_id, len(prompt), delta is not None,
                )
                if len(prompt) > self.settings.max_prompt_chars:
                    raise ValueError(
                        f"Serialized prompt has {len(prompt)} characters; limit is {self.settings.max_prompt_chars}"
                    )

                browser_result, session_page = await self.worker.query(
                    request_id=request_id,
                    prompt=prompt,
                    timeout_seconds=self.settings.request_timeout_seconds,
                    progress=lambda stage: self._set_stage(request_id, stage),
                    page=page_arg,
                    keep_open=True,
                    mode=request.mode,
                )
                self._session_page = session_page
                self._session_messages = list(request.messages)

            self._set_stage(request_id, "parsing_response")
            try:
                parsed = parse_assistant_output(browser_result.answer, allowed_tools)
            except ToolProtocolError:
                raise

            duration_ms = int((time.perf_counter() - started) * 1000)
            usage = Usage(
                prompt_tokens=estimate_tokens(prompt),
                completion_tokens=estimate_tokens(browser_result.answer),
            )
            self._set_stage(request_id, "completed")
            LOGGER.info("[%s] completion finished in %d ms", request_id, duration_ms)
            return CompletionResult(
                completion_id=f"chatcmpl-{request_id}",
                model=request.model,
                parsed=parsed,
                usage=usage,
                duration_ms=duration_ms,
                browser_url=browser_result.browser_url,
                partial=browser_result.partial,
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._set_stage(request_id, "failed")
            LOGGER.exception("[%s] completion failed", request_id)
            raise

    async def close(self) -> None:
        await self.manager.close()
