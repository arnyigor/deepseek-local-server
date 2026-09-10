from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from deepseek_local_server.direct.client import RemoteSession
from deepseek_local_server.openai.schemas import ChatMessage


@dataclass(slots=True)
class AgentSession:
    remote: RemoteSession = field(default_factory=RemoteSession)
    last_messages: list[ChatMessage] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def reset(self) -> None:
        self.remote.reset()
        self.last_messages = None

    def continuation_delta(self, messages: list[ChatMessage]) -> list[ChatMessage] | None:
        prev = self.last_messages
        if prev is None:
            return None
        if len(messages) <= len(prev) or messages[: len(prev)] != prev:
            return None
        return messages[len(prev):]


class SessionStore:
    def __init__(self) -> None:
        self._items: dict[str, AgentSession] = {}
        self._guard = asyncio.Lock()

    async def get(self, key: str) -> AgentSession:
        async with self._guard:
            return self._items.setdefault(key, AgentSession())

    async def reset(self, key: str) -> bool:
        async with self._guard:
            session = self._items.get(key)
            if session is None:
                return False
            session.reset()
            return True

    async def reset_all(self) -> int:
        async with self._guard:
            n = len(self._items)
            for session in self._items.values():
                session.reset()
            self._items.clear()
            return n

    async def status(self) -> list[dict[str, object]]:
        async with self._guard:
            return [
                {
                    "id": key,
                    "remote_session": value.remote.id,
                    "message_count": value.remote.message_count,
                    "busy": value.lock.locked(),
                }
                for key, value in self._items.items()
            ]
