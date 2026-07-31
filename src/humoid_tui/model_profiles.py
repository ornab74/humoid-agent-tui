from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True, slots=True)
class ModelProfile:
    key: str
    match: tuple[str, ...]
    protocol: str
    prompt_style: str
    supports_parallel: bool = True
    supports_reasoning: bool = False
    supports_programmatic_tools: bool = False
    preferred_api: str = "chat_completions"
    request_overrides: dict[str, Any] = field(default_factory=dict)

PROFILES = (
    ModelProfile("gpt56", ("gpt-5.6",), "openai", "gpt56", True, True, True, "responses"),
    ModelProfile("gemma4", ("gemma-4", "gemma4"), "gemma4", "gemma4", False, True),
    ModelProfile("glm52", ("glm-5.2", "glm5.2", "glm_5_2"), "tagged-json", "glm52", True, True),
    ModelProfile("muse", ("muse-spark", "muse spark"), "openai", "muse", True, True),
    ModelProfile("llamacpp", ("local-model",), "auto", "local", True, False),
    ModelProfile("litert", ("litert",), "auto", "local", False, False),
)

DEFAULT = ModelProfile("generic", (), "auto", "generic", True, False)

def resolve_profile(provider: str, model: str) -> ModelProfile:
    identity = f"{provider} {model}".lower()
    for profile in PROFILES:
        if any(token in identity for token in profile.match):
            return profile
    if provider == "llamacpp":
        return next(p for p in PROFILES if p.key == "llamacpp")
    if provider == "litert":
        return next(p for p in PROFILES if p.key == "litert")
    return DEFAULT
