from __future__ import annotations

"""Gemma 4 agentic prompt and history policy.

The OpenAI-compatible transport remains the preferred path. This module adds a
Gemma-native fallback and the conversation policy required by Gemma 4:
- dedicated tool declaration/call/response control tokens
- <|"|> string delimiters
- optional thinking mode
- thoughts retained during a tool-call chain, then stripped after completion
"""

from dataclasses import dataclass
from typing import Any
import json
import re

STRING_DELIM = '<|"|>'


def _scalar(value: Any) -> str:
    if isinstance(value, str):
        return f"{STRING_DELIM}{value}{STRING_DELIM}"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    # Nested values are encoded as a delimited JSON string for maximum parser safety.
    return f"{STRING_DELIM}{json.dumps(value, ensure_ascii=False, separators=(',', ':'))}{STRING_DELIM}"


def render_mapping(mapping: dict[str, Any]) -> str:
    return ",".join(f"{key}:{_scalar(value)}" for key, value in mapping.items())


def render_tool_declaration(schema: dict[str, Any]) -> str:
    fn = schema.get("function", schema)
    name = fn.get("name", "unknown")
    payload = {
        "description": fn.get("description", ""),
        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
    }
    return f"<|tool>declaration:{name}{{{render_mapping(payload)}}}<tool|>"


def render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    return f"<|tool_call>call:{name}{{{render_mapping(arguments)}}}<tool_call|>"


def render_tool_response(name: str, output: str, ok: bool = True) -> str:
    return (
        f"<|tool_response>response:{name}{{ok:{str(ok).lower()},"
        f"output:{_scalar(output)}}}<tool_response|>"
    )


def thinking_instruction(level: str) -> str:
    if level == "off":
        return "Do not emit a thought channel. Use tools directly when needed."
    if level == "on":
        return "<|think|> Use deliberate reasoning before tool calls. Keep hidden thought concise and task-focused."
    return (
        "<|think|> Use adaptive LOW thinking: reason only as much as necessary, "
        "prefer direct verified tool calls, and avoid repetitive analysis."
    )


def gemma_system_suffix(tool_schemas: list[dict[str, Any]], thinking: str) -> str:
    declarations = "".join(render_tool_declaration(x) for x in tool_schemas)
    return (
        "\n\n[GEMMA 4 NATIVE AGENT MODE]\n"
        f"{thinking_instruction(thinking)}\n"
        "Use the native Gemma 4 tool lifecycle exactly. String values inside native "
        f"blocks must use {STRING_DELIM} delimiters. Stop after emitting a tool call.\n"
        f"{declarations}"
    )


_THOUGHT = re.compile(r"<\|channel>thought\s*.*?<channel\|>", re.DOTALL)


def strip_thought_channels(text: str) -> str:
    return _THOUGHT.sub("", text).strip()


@dataclass(slots=True)
class Gemma4RuntimePolicy:
    mode: str = "auto"
    thinking: str = "low"
    native_fallback: bool = True
    strip_completed_thoughts: bool = True

    def active(self, model: str) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "native":
            return True
        identity = model.lower()
        return "gemma-4" in identity or "gemma4" in identity
