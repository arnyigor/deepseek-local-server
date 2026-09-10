from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    model_type: str
    thinking: bool
    search: bool
    label: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "deepseek-chat": ModelSpec("deepseek-chat", "default", False, False, "Fast"),
    "deepseek-reasoner": ModelSpec("deepseek-reasoner", "default", True, False, "Fast + reasoning"),
    "deepseek-chat-search": ModelSpec("deepseek-chat-search", "default", False, True, "Fast + web search"),
    "deepseek-reasoner-search": ModelSpec("deepseek-reasoner-search", "default", True, True, "Fast + reasoning + web search"),
}

ALIASES = {
    "deepseek-default": "deepseek-chat",
    "deepseek-v3": "deepseek-chat",
    "deepseek-r1": "deepseek-reasoner",
    "deepseek-r1-search": "deepseek-reasoner-search",
    # DeepSeek merged Instant/Expert/Vision into one "default" model (2026-09):
    # these are kept only so old model names in configs/scripts keep working.
    "deepseek-web": "deepseek-reasoner",
    "deepseek-expert": "deepseek-reasoner",
    "deepseek-v4-pro": "deepseek-reasoner",
}


def resolve_model(model: str) -> ModelSpec:
    key = ALIASES.get(model.lower(), model.lower())
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model!r}. Available: {', '.join(sorted(MODEL_SPECS))}") from exc
