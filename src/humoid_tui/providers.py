from __future__ import annotations
from collections.abc import AsyncIterator
from typing import Any
from openai import AsyncOpenAI
from .config import ProviderConfig, Settings

class ProviderError(RuntimeError):
    pass

class UnifiedProvider:
    """Transport adapter. Tool syntax is normalized later by tool_protocols.py."""

    def __init__(self, cfg: ProviderConfig, settings: Settings):
        if not cfg.base_url:
            raise ProviderError(f"{cfg.name} base URL is not configured")
        if not cfg.model:
            raise ProviderError(f"{cfg.name} model is not configured")
        self.cfg = cfg
        self.settings = settings
        self.client = AsyncOpenAI(api_key=cfg.api_key or "local", base_url=cfg.base_url)

    async def health(self) -> tuple[bool, str]:
        try:
            models = await self.client.models.list()
            return True, f"{self.cfg.name}: {len(models.data)} model(s) visible"
        except Exception as exc:
            # Some compatible endpoints do not expose /models, but may still work.
            return False, f"{self.cfg.name}: model-list probe failed: {exc}"

    async def stream(self, messages: list[dict[str, Any]],
                     tools: list[dict[str, Any]] | None = None) -> AsyncIterator[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.settings.humoid_temperature,
            "max_tokens": self.settings.humoid_max_output_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = self.settings.humoid_parallel_tool_calls

        if self.cfg.name == "meta" and self.settings.meta_enable_search_grounding:
            kwargs["extra_body"] = {"search_grounding": {"enabled": True}}

        try:
            stream = await self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Compatibility fallback for servers rejecting parallel_tool_calls.
            if "parallel_tool_calls" in kwargs:
                kwargs.pop("parallel_tool_calls", None)
                try:
                    stream = await self.client.chat.completions.create(**kwargs)
                except Exception:
                    raise ProviderError(str(exc)) from exc
            else:
                raise ProviderError(str(exc)) from exc

        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            yield {
                "text": delta.content or "",
                "reasoning": getattr(delta, "reasoning_content", None) or "",
                "tool_calls": [tc.model_dump() for tc in (delta.tool_calls or [])],
                "finish_reason": choice.finish_reason,
                "usage": getattr(chunk, "usage", None),
            }
