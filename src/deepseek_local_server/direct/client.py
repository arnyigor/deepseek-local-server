from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from deepseek_local_server.config import Settings
from deepseek_local_server.direct.auth_config import DeepSeekAuth, load_auth
from deepseek_local_server.direct.pow import PowSolver
from deepseek_local_server.direct.stream import DeepSeekPatchParser, StreamPiece
from deepseek_local_server.errors import AuthenticationRequiredError, DirectBackendError, DirectProtocolError, DirectRateLimitError
from deepseek_local_server.models import ModelSpec

LOGGER = logging.getLogger("deepseek_local_server.direct")


@dataclass(slots=True)
class RemoteSession:
    id: str | None = None
    parent_message_id: str | None = None
    created_at: float = 0.0
    message_count: int = 0

    def reset(self) -> None:
        self.id = None
        self.parent_message_id = None
        self.created_at = 0.0
        self.message_count = 0


class DeepSeekDirectClient:
    POW_PATH = "/api/v0/chat/create_pow_challenge"
    SESSION_PATH = "/api/v0/chat_session/create"
    COMPLETION_PATH = "/api/v0/chat/completion"
    FILE_UPLOAD_PATH = "/api/v0/file/upload_file"
    FILE_STATUS_PATH = "/api/v0/file/fetch_files"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._auth: DeepSeekAuth | None = None
        self._pow = PowSolver(settings.wasm_cache_dir)
        self._sem = asyncio.Semaphore(settings.max_concurrent_direct)

    def reload_auth(self) -> DeepSeekAuth:
        self._auth = load_auth(self.settings.auth_file)
        return self._auth

    @property
    def auth(self) -> DeepSeekAuth:
        return self._auth or self.reload_auth()

    def _headers(self) -> dict[str, str]:
        auth = self.auth
        return {
            "User-Agent": auth.user_agent,
            "x-client-platform": "web",
            "x-client-version": auth.client_version,
            "x-client-locale": auth.locale,
            "x-client-timezone-offset": auth.timezone_offset,
            "x-app-version": auth.app_version,
            "Authorization": f"Bearer {auth.token}",
            "x-hif-dliq": auth.hif_dliq,
            "x-hif-leim": auth.hif_leim,
            "Origin": "https://chat.deepseek.com",
            "Referer": "https://chat.deepseek.com/",
            "Cookie": auth.cookie,
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
        }

    async def _post_json(self, client: httpx.AsyncClient, path: str, body: dict) -> dict:
        response = await client.post(path, headers=self._headers(), json=body)
        if response.status_code in {401, 403}:
            raise AuthenticationRequiredError(f"DeepSeek Web authentication rejected the request (HTTP {response.status_code})")
        if response.status_code == 429:
            raise DirectRateLimitError("DeepSeek Web rate limit reached (HTTP 429)")
        if response.is_error:
            raise DirectBackendError(f"DeepSeek Web returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            return response.json()
        except Exception as exc:
            raise DirectProtocolError(f"DeepSeek Web returned non-JSON data for {path}: {response.text[:160]}") from exc

    async def _challenge(self, client: httpx.AsyncClient, target_path: str) -> tuple[dict, int]:
        payload = await self._post_json(client, self.POW_PATH, {"target_path": target_path})
        challenge = (((payload.get("data") or {}).get("biz_data") or {}).get("challenge"))
        if not isinstance(challenge, dict):
            raise DirectProtocolError("DeepSeek PoW response has no data.biz_data.challenge")
        answer = await self._pow.solve(challenge, self.auth.wasm_url)
        return challenge, answer

    async def _ensure_session(self, client: httpx.AsyncClient, session: RemoteSession) -> bool:
        """Ensure a usable remote chat session and report whether a fresh one was created."""
        now = asyncio.get_running_loop().time()
        if session.id and session.message_count < self.settings.max_session_messages and now - session.created_at < self.settings.session_ttl_seconds:
            return False
        session.reset()
        payload = await self._post_json(client, self.SESSION_PATH, {})
        biz = ((payload.get("data") or {}).get("biz_data") or {})
        chat = biz.get("chat_session") or {}
        session_id = chat.get("id") if isinstance(chat, dict) else None
        session_id = session_id or biz.get("id")
        if not session_id:
            raise DirectProtocolError("DeepSeek did not return a chat_session id")
        session.id = str(session_id)
        session.created_at = now
        return True

    @staticmethod
    def _pow_header(challenge: dict, answer: int, target_path: str) -> str:
        payload = {
            "algorithm": challenge.get("algorithm"),
            "challenge": challenge.get("challenge"),
            "salt": challenge.get("salt"),
            "answer": answer,
            "signature": challenge.get("signature"),
            "target_path": target_path,
        }
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return base64.b64encode(raw).decode("ascii")

    async def upload_image(self, content: bytes, filename: str, content_type: str, spec: ModelSpec) -> str:
        """Upload one image and wait until DeepSeek finishes analyzing it, returning its file id."""
        timeout = httpx.Timeout(self.settings.fetch_timeout_seconds)
        async with httpx.AsyncClient(base_url="https://chat.deepseek.com", timeout=timeout) as client:
            challenge, answer = await self._challenge(client, self.FILE_UPLOAD_PATH)
            headers = self._headers()
            headers.pop("Content-Type", None)  # let httpx set the multipart boundary
            headers["X-DS-PoW-Response"] = self._pow_header(challenge, answer, self.FILE_UPLOAD_PATH)
            headers["x-file-size"] = str(len(content))
            headers["x-model-type"] = spec.model_type
            headers["x-thinking-enabled"] = "1" if spec.thinking else "0"
            response = await client.post(
                self.FILE_UPLOAD_PATH, headers=headers, files={"file": (filename, content, content_type)}
            )
            if response.status_code in {401, 403}:
                raise AuthenticationRequiredError(f"DeepSeek Web authentication rejected the upload (HTTP {response.status_code})")
            if response.is_error:
                raise DirectBackendError(f"DeepSeek file upload HTTP {response.status_code}: {response.text[:300]}")
            file_id = (((response.json().get("data") or {}).get("biz_data") or {}).get("id"))
            if not file_id:
                raise DirectProtocolError("DeepSeek file upload response has no data.biz_data.id")

            deadline = asyncio.get_running_loop().time() + min(self.settings.fetch_timeout_seconds, 60.0)
            while True:
                status_response = await client.get(
                    self.FILE_STATUS_PATH, headers=self._headers(), params={"file_ids": file_id}
                )
                if status_response.is_error:
                    raise DirectBackendError(f"DeepSeek file status HTTP {status_response.status_code}")
                files = (((status_response.json().get("data") or {}).get("biz_data") or {}).get("files")) or []
                status = files[0].get("status") if files else None
                if status == "SUCCESS":
                    return str(file_id)
                if status in {"FAILED", "ERROR"}:
                    raise DirectBackendError(f"DeepSeek could not process the uploaded image (status={status})")
                if asyncio.get_running_loop().time() > deadline:
                    raise DirectProtocolError("Timed out waiting for DeepSeek to finish analyzing the uploaded image")
                await asyncio.sleep(0.5)

    async def stream(
        self,
        prompt: str,
        spec: ModelSpec,
        session: RemoteSession,
        *,
        fresh_prompt: str | None = None,
        ref_file_ids: list[str] | None = None,
    ) -> AsyncIterator[StreamPiece]:
        """Stream one DeepSeek completion, recovering one stale remote session.

        ``prompt`` may be a compact continuation delta. ``fresh_prompt`` is the full
        serialized conversation and is mandatory for safe session rollover/recovery.
        """
        recovery_prompt = fresh_prompt or prompt
        if len(prompt) > self.settings.max_prompt_chars or len(recovery_prompt) > self.settings.max_prompt_chars:
            longest = max(len(prompt), len(recovery_prompt))
            raise ValueError(f"Prompt has {longest} chars; limit is {self.settings.max_prompt_chars}")

        async with self._sem:
            timeout = httpx.Timeout(self.settings.fetch_timeout_seconds, read=self.settings.request_timeout_seconds)
            async with httpx.AsyncClient(base_url="https://chat.deepseek.com", timeout=timeout, follow_redirects=True) as client:
                challenge, answer = await self._challenge(client, self.COMPLETION_PATH)
                fresh_session = await self._ensure_session(client, session)
                effective_prompt = recovery_prompt if fresh_session else prompt

                for attempt in range(2):
                    headers = self._headers()
                    headers["X-DS-PoW-Response"] = self._pow_header(challenge, answer, self.COMPLETION_PATH)
                    body = {
                        "chat_session_id": session.id,
                        "parent_message_id": session.parent_message_id,
                        "model_type": spec.model_type,
                        "prompt": effective_prompt,
                        "ref_file_ids": ref_file_ids or [],
                        "thinking_enabled": spec.thinking,
                        "search_enabled": spec.search,
                        "action": None,
                        "preempt": False,
                    }
                    async with client.stream("POST", self.COMPLETION_PATH, headers=headers, json=body) as response:
                        if response.status_code in {401, 403}:
                            raise AuthenticationRequiredError(
                                f"DeepSeek Web authentication rejected completion (HTTP {response.status_code})"
                            )
                        if response.status_code == 429:
                            raise DirectRateLimitError("DeepSeek Web rate limit reached (HTTP 429)")
                        if response.is_error:
                            text = (await response.aread()).decode("utf-8", errors="replace")
                            if response.status_code in {400, 404, 500} and attempt == 0:
                                # The DeepSeek remote chain can expire independently of our local
                                # TTL. Recreate it once and replay the complete conversation.
                                session.reset()
                                await self._ensure_session(client, session)
                                effective_prompt = recovery_prompt
                                continue
                            raise DirectBackendError(
                                f"DeepSeek completion HTTP {response.status_code}: {text[:300]}"
                            )

                        parser = DeepSeekPatchParser()
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line or line.startswith(":"):
                                continue
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                event = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            for piece in parser.feed(event):
                                yield piece
                        if parser.model_error:
                            raise DirectBackendError(parser.model_error)
                        if parser.message_id:
                            session.parent_message_id = parser.message_id
                            session.message_count += 1
                        if not parser.content.strip():
                            raise DirectProtocolError("DeepSeek direct backend returned an empty response")
                        return

                raise DirectBackendError("DeepSeek direct backend failed after session recovery")

    async def probe(self) -> dict[str, object]:
        timeout = httpx.Timeout(min(self.settings.fetch_timeout_seconds, 10.0))
        async with httpx.AsyncClient(base_url="https://chat.deepseek.com", timeout=timeout) as client:
            try:
                payload = await self._post_json(client, self.POW_PATH, {"target_path": self.COMPLETION_PATH})
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        challenge = (((payload.get("data") or {}).get("biz_data") or {}).get("challenge"))
        return {"ok": isinstance(challenge, dict), "challenge": isinstance(challenge, dict)}
