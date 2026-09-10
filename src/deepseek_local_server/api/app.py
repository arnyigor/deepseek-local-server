from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from deepseek_local_server import __version__
from deepseek_local_server.api.dependencies import require_bearer_token
from deepseek_local_server.auth import ensure_api_token
from deepseek_local_server.config import Settings
from deepseek_local_server.errors import AuthenticationRequiredError, DeepSeekLocalError
from deepseek_local_server.models import MODEL_SPECS
from deepseek_local_server.openai.responses import Usage, completion_payload, sse
from deepseek_local_server.openai.schemas import ChatCompletionRequest, ChatMessage
from deepseek_local_server.service import CompletionService


def _usage_payload(usage: Usage) -> dict[str, Any]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens + usage.reasoning_tokens,
        "total_tokens": usage.total_tokens,
        "completion_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
    }


def _openai_error(message: str, error_type: str = "server_error", code: str = "deepseek_web_error", status: int = 502) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": error_type, "param": None, "code": code}})


def _session_key(request: Request, payload: ChatCompletionRequest) -> str:
    return request.headers.get("x-agent-session") or payload.user or (request.client.host if request.client else "default")


def create_app(settings: Settings | None = None, service: CompletionService | Any | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()
    resolved.ensure_directories()
    token = ensure_api_token(resolved)
    completion_service = service or CompletionService(resolved)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        close = getattr(completion_service, "close", None)
        if close:
            await close()

    app = FastAPI(title="DeepSeek Local Server v2", version=__version__, lifespan=lifespan, redoc_url=None)
    app.state.settings = resolved
    app.state.api_token = token
    app.state.service = completion_service

    @app.get("/health")
    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "transport": "hybrid", **completion_service.runtime_status()}

    @app.get("/v1/models", dependencies=[Depends(require_bearer_token)])
    async def models() -> dict[str, Any]:
        now = int(time.time())
        return {"object": "list", "data": [{"id": key, "object": "model", "created": now, "owned_by": "deepseek-web"} for key in MODEL_SPECS]}

    @app.get("/v1/model-capabilities", dependencies=[Depends(require_bearer_token)])
    async def capabilities() -> dict[str, Any]:
        return {
            key: {"model_type": spec.model_type, "reasoning": spec.thinking, "web_search": spec.search, "label": spec.label}
            for key, spec in MODEL_SPECS.items()
        }

    @app.get("/v1/sessions", dependencies=[Depends(require_bearer_token)])
    async def sessions() -> list[dict[str, object]]:
        return await completion_service.sessions.status()

    @app.post("/v1/sessions/reset", dependencies=[Depends(require_bearer_token)])
    async def reset_session(agent: str = Query("all")) -> dict[str, Any]:
        if agent == "all":
            return {"reset": await completion_service.sessions.reset_all()}
        return {"reset": await completion_service.sessions.reset(agent), "agent": agent}

    @app.post("/v1/chat/completions", dependencies=[Depends(require_bearer_token)])
    async def chat(request: Request, payload: ChatCompletionRequest):
        session_key = _session_key(request, payload)
        if payload.stream:
            completion_id = f"chatcmpl-{uuid4().hex}"
            created = int(time.time())

            async def generate():
                last_backend = "direct"
                try:
                    async for event in completion_service.stream(payload, session_key):
                        last_backend = event.get("backend", last_backend)
                        body = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": payload.model,
                            "choices": [{"index": 0, "delta": event["delta"], "finish_reason": event["finish_reason"]}],
                        }
                        if "usage" in event and payload.stream_options and payload.stream_options.include_usage:
                            body["usage"] = _usage_payload(event["usage"])
                        yield sse(body)
                    yield sse("[DONE]")
                except Exception as exc:
                    yield sse({"error": {"message": str(exc), "type": type(exc).__name__, "code": "stream_error"}})
                    yield sse("[DONE]")

            mode = "buffered-tools" if payload.tools else "live-direct"
            return StreamingResponse(generate(), media_type="text/event-stream", headers={"X-DeepSeek-Local-Stream-Mode": mode})

        try:
            result = await completion_service.complete(payload, session_key)
        except ValueError as exc:
            return _openai_error(str(exc), "invalid_request_error", "invalid_request", 400)
        except AuthenticationRequiredError as exc:
            return _openai_error(str(exc), "authentication_error", "deepseek_login_required", 401)
        except DeepSeekLocalError as exc:
            return _openai_error(str(exc))
        except Exception as exc:
            return _openai_error(f"{type(exc).__name__}: {exc}")
        return JSONResponse(
            completion_payload(result.completion_id, result.model, result.parsed, result.reasoning, result.usage),
            headers={"X-DeepSeek-Local-Backend": result.backend, "X-DeepSeek-Local-Duration-Ms": str(result.duration_ms)},
        )

    # Minimal Anthropic-compatible shim. It deliberately routes through the same core.
    @app.post("/v1/messages", dependencies=[Depends(require_bearer_token)])
    async def anthropic_messages(request: Request):
        body = await request.json()
        messages = []
        system = body.get("system")
        if system:
            messages.append(ChatMessage(role="system", content=system))
        for item in body.get("messages", []):
            messages.append(ChatMessage.model_validate(item))
        payload = ChatCompletionRequest(model=body.get("model", "deepseek-reasoner"), messages=messages, stream=False, user=body.get("user"))
        result = await completion_service.complete(payload, _session_key(request, payload))
        return {
            "id": "msg_" + result.completion_id,
            "type": "message",
            "role": "assistant",
            "model": payload.model,
            "content": [{"type": "text", "text": result.parsed.content}],
            "stop_reason": "tool_use" if result.parsed.tool_calls else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": result.usage.prompt_tokens, "output_tokens": result.usage.completion_tokens + result.usage.reasoning_tokens},
        }

    @app.post("/v1/responses", dependencies=[Depends(require_bearer_token)])
    async def responses_api(request: Request):
        body = await request.json()
        inp = body.get("input", "")
        if isinstance(inp, str):
            messages = [ChatMessage(role="user", content=inp)]
        elif isinstance(inp, list):
            messages = [ChatMessage.model_validate(x) for x in inp if isinstance(x, dict) and x.get("role")]
        else:
            raise HTTPException(400, "Unsupported Responses API input")
        payload = ChatCompletionRequest(model=body.get("model", "deepseek-reasoner"), messages=messages, stream=False, user=body.get("user"))
        result = await completion_service.complete(payload, _session_key(request, payload))
        return {
            "id": result.completion_id.replace("chatcmpl-", "resp_"),
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": payload.model,
            "output": [{"type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": result.parsed.content, "annotations": []}]}],
            "output_text": result.parsed.content,
            "usage": {"input_tokens": result.usage.prompt_tokens, "output_tokens": result.usage.completion_tokens + result.usage.reasoning_tokens, "total_tokens": result.usage.total_tokens},
        }

    return app
