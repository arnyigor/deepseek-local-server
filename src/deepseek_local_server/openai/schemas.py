from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: Any = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=256)
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    function: FunctionDefinition


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    tools: list[ToolDefinition] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stop: str | list[str] | None = None
    user: str | None = None

    @model_validator(mode="after")
    def validate_supported_shape(self) -> "ChatCompletionRequest":
        if self.n != 1:
            raise ValueError("Only n=1 is supported")
        return self

    @property
    def requested_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "deepseek-local-server"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
