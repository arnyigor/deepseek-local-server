from __future__ import annotations

import time
from contextlib import asynccontextmanager
import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from deepseek_local_server import __version__
from deepseek_local_server.api.dependencies import require_bearer_token
from deepseek_local_server.auth import ensure_api_token
from deepseek_local_server.config import Settings
from deepseek_local_server.errors import (
    AuthenticationRequiredError,
    BrowserProtocolError,
    DeepSeekPageError,
    ToolProtocolError,
    UnsupportedContentError,
)
from deepseek_local_server.openai.responses import buffered_stream, completion_payload
from deepseek_local_server.openai.schemas import ChatCompletionRequest, ModelCard, ModelList
from deepseek_local_server.service import CompletionService

LOGGER = logging.getLogger("deepseek_local_server.api")


def _openai_error(message: str, error_type: str, code: str, http_status: int) -> JSONResponse:
    return JSONResponse(
        status_code=http_status,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )


def create_app(settings: Settings | None = None, service: CompletionService | Any | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()
    resolved.ensure_directories()
    token = ensure_api_token(resolved)
    completion_service = service or CompletionService(resolved)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        close = getattr(completion_service, "close", None)
        if close is not None:
            await close()

    app = FastAPI(
        title="DeepSeek Local Server",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.api_token = token
    app.state.service = completion_service

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _openai_error(str(exc), "invalid_request_error", "invalid_request", 400)

    @app.get("/health")
    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "model": resolved.model_id,
            "browser_started": completion_service.manager.started if hasattr(completion_service, "manager") else False,
            "queue_busy": bool(getattr(completion_service, "busy", False)),
            "streaming": "buffered",
            **(completion_service.runtime_status() if hasattr(completion_service, "runtime_status") else {}),
        }

    @app.get(
        "/v1/models",
        response_model=ModelList,
        dependencies=[Depends(require_bearer_token)],
    )
    async def models() -> ModelList:
        return ModelList(
            data=[
                ModelCard(
                    id=resolved.model_id,
                    created=int(time.time()),
                )
            ]
        )

    @app.post("/v1/chat/completions", dependencies=[Depends(require_bearer_token)])
    async def chat_completions(request: ChatCompletionRequest):  # type: ignore[no-untyped-def]
        LOGGER.info(
            "POST /v1/chat/completions model=%s stream=%s messages=%d tools=%d",
            request.model, request.stream, len(request.messages), len(request.tools or []),
        )
        try:
            result = await completion_service.complete(request)
        except ValueError as exc:
            return _openai_error(str(exc), "invalid_request_error", "invalid_request", 400)
        except UnsupportedContentError as exc:
            return _openai_error(str(exc), "invalid_request_error", "unsupported_content", 400)
        except AuthenticationRequiredError as exc:
            return _openai_error(str(exc), "authentication_error", "deepseek_login_required", 401)
        except ToolProtocolError as exc:
            return _openai_error(str(exc), "server_error", "invalid_tool_protocol", 502)
        except (BrowserProtocolError, DeepSeekPageError) as exc:
            return _openai_error(str(exc), "server_error", "deepseek_web_error", 502)

        headers = {
            "X-DeepSeek-Local-Stream-Mode": "buffered",
            "X-DeepSeek-Local-Duration-Ms": str(result.duration_ms),
            "X-DeepSeek-Local-Partial": "true" if result.partial else "false",
        }
        if request.stream:
            include_usage = bool(request.stream_options and request.stream_options.include_usage)
            return StreamingResponse(
                buffered_stream(
                    completion_id=result.completion_id,
                    model=result.model,
                    parsed=result.parsed,
                    usage=result.usage,
                    include_usage=include_usage,
                ),
                media_type="text/event-stream",
                headers=headers,
            )

        return JSONResponse(
            content=completion_payload(
                completion_id=result.completion_id,
                model=result.model,
                parsed=result.parsed,
                usage=result.usage,
            ),
            headers=headers,
        )

    return app
