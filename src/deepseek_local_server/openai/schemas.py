from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "tool"]
    content: Any = None
    tool_call_id: str | None = None
    name: str | None = None


class FunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDef(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "deepseek-reasoner"
    messages: list[ChatMessage]
    stream: bool = False
    stream_options: StreamOptions | None = None
    tools: list[ToolDef] | None = None
    tool_choice: Any = None
    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
