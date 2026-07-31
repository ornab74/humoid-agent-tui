from __future__ import annotations

"""Model-aware tool-call protocol normalization.

Every provider/model dialect is translated into a single internal ToolCall IR.  The
IR is also rendered as the human-readable TOOL.CALL envelope used by the TUI and
trace logs.  This keeps model syntax at the edge and policy/execution in the core.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Iterable
from uuid import uuid4
import ast
import json
import re
from .gemma4 import render_tool_response


@dataclass(slots=True)
class CanonicalToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    agent: str = "root"
    provider: str = "unknown"
    model: str = "unknown"
    protocol: str = "unknown"
    source: str = "model"
    confidence: float = 1.0
    raw: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        blob = json.dumps(
            {"name": self.name, "arguments": self.arguments},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        return sha256(blob).hexdigest()[:12]

    def openai_tool_call(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    def envelope(self, phase: str = "REQUEST") -> str:
        header = (
            f"⟪TOOL.CALL/v1 {phase} id={self.call_id} agent={self.agent} "
            f"provider={self.provider} model={self.model} protocol={self.protocol} "
            f"sha={self.digest} confidence={self.confidence:.2f}⟫"
        )
        body = json.dumps(
            {"tool": self.name, "arguments": self.arguments, "metadata": self.metadata},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        return f"{header}\n{body}\n⟪/TOOL.CALL⟫"


@dataclass(slots=True)
class ToolResult:
    call: CanonicalToolCall
    output: str
    ok: bool
    elapsed_ms: float = 0.0

    def envelope(self) -> str:
        header = (
            f"⟪TOOL.RESULT/v1 id={self.call.call_id} tool={self.call.name} "
            f"ok={str(self.ok).lower()} elapsed_ms={self.elapsed_ms:.1f} "
            f"sha={self.call.digest}⟫"
        )
        return f"{header}\n{self.output}\n⟪/TOOL.RESULT⟫"


class ProtocolError(ValueError):
    pass


_FENCE = re.compile(
    r"^\s*```(?:json|json5|javascript|tool_call)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _strip_json_wrappers(value: str) -> str:
    """Remove harmless wrappers commonly emitted by compatible model servers."""

    value = value.strip().lstrip("\ufeff")
    fenced = _FENCE.match(value)
    if fenced:
        value = fenced.group(1).strip()

    for opening, closing in (
        ("<tool_call>", "</tool_call>"),
        ("<|tool_call|>", "<|/tool_call|>"),
    ):
        if value.startswith(opening) and value.endswith(closing):
            value = value[len(opening):-len(closing)].strip()

    prefix = re.match(
        r"^(?:arguments?|parameters?|params?)\s*[:=]\s*(.+)$",
        value,
        re.IGNORECASE | re.DOTALL,
    )
    if prefix:
        value = prefix.group(1).strip()

    return value


def _json_loads(value: str) -> Any:
    """Decode JSON while tolerating literal control characters in strings."""

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return json.loads(value, strict=False)


def _unwrap_json_strings(value: Any, *, depth: int = 4) -> Any:
    """Unwrap providers that double-encode function argument objects."""

    current = value
    for _ in range(depth):
        if not isinstance(current, str):
            break
        candidate = _strip_json_wrappers(current)
        if not candidate:
            return {}
        try:
            decoded = _json_loads(candidate)
        except json.JSONDecodeError:
            break
        if decoded == current:
            break
        current = decoded
    return current


def _last_embedded_json_object(value: str) -> dict[str, Any] | None:
    """Recover the last complete object from cumulative or duplicated deltas.

    A few OpenAI-compatible servers stream cumulative ``arguments`` snapshots.
    Concatenating those snapshots can produce text such as ``{...}{...}`` or a
    partial prefix followed by a complete object.  Scanning every object start
    lets us select the newest complete mapping without evaluating model text.
    """

    decoder = json.JSONDecoder(strict=False)
    objects: list[dict[str, Any]] = []
    for index, character in enumerate(value):
        if character != "{":
            continue
        try:
            decoded, _ = decoder.raw_decode(value, index)
        except json.JSONDecodeError:
            continue
        decoded = _unwrap_json_strings(decoded)
        if isinstance(decoded, dict):
            objects.append(decoded)
    return objects[-1] if objects else None


def _jsonish(value: str) -> Any:
    value = _strip_json_wrappers(value)
    if not value:
        return {}

    first_error: json.JSONDecodeError | None = None
    candidates = [value]

    # Conservative repair for common local-model output. No eval.
    repaired = re.sub(r",\s*([}\]])", r"\1", value)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    if repaired != value:
        candidates.append(repaired)

    for candidate in candidates:
        try:
            decoded = _unwrap_json_strings(_json_loads(candidate))
            return decoded
        except json.JSONDecodeError as exc:
            if first_error is None:
                first_error = exc

    for candidate in candidates:
        embedded = _last_embedded_json_object(candidate)
        if embedded is not None:
            return embedded

    try:
        literal = ast.literal_eval(value)
        literal = _unwrap_json_strings(literal)
        if isinstance(literal, (dict, list, str, int, float, bool, type(None))):
            return literal
    except (ValueError, SyntaxError):
        pass

    preview = value[:180].replace("\n", "\\n").replace("\r", "\\r")
    location = ""
    if first_error is not None:
        location = f" at line {first_error.lineno} column {first_error.colno}"
    raise ProtocolError(
        "Tool arguments are not valid JSON or a safe Python literal"
        f"{location}; preview={preview!r}"
    )


def _new_call(name: str, arguments: Any, *, provider: str, model: str,
              protocol: str, raw: str = "", call_id: str | None = None,
              confidence: float = 1.0, metadata: dict[str, Any] | None = None) -> CanonicalToolCall:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]{0,127}", name or ""):
        raise ProtocolError(f"Invalid tool name: {name!r}")
    if isinstance(arguments, str):
        arguments = _jsonish(arguments)
    arguments = _unwrap_json_strings(arguments)
    if not isinstance(arguments, dict):
        raise ProtocolError("Tool arguments must normalize to an object")
    return CanonicalToolCall(
        call_id=call_id or f"call_{uuid4().hex[:20]}",
        name=name,
        arguments=arguments,
        provider=provider,
        model=model,
        protocol=protocol,
        raw=raw,
        confidence=confidence,
        metadata=metadata or {},
    )


class ToolProtocol:
    name = "base"

    def parse(self, *, text: str, structured: list[dict[str, Any]],
              provider: str, model: str) -> list[CanonicalToolCall]:
        raise NotImplementedError

    def append_result(self, messages: list[dict[str, Any]], result: ToolResult) -> None:
        messages.append({
            "role": "tool",
            "tool_call_id": result.call.call_id,
            "content": result.output,
        })


class OpenAIChatProtocol(ToolProtocol):
    name = "openai-chat-tools"

    def parse(self, *, text: str, structured: list[dict[str, Any]],
              provider: str, model: str) -> list[CanonicalToolCall]:
        calls: list[CanonicalToolCall] = []
        for i, tc in enumerate(structured):
            fn = tc.get("function") or {}
            calls.append(_new_call(
                fn.get("name", ""), fn.get("arguments", "{}"),
                provider=provider, model=model, protocol=self.name,
                raw=json.dumps(tc, ensure_ascii=False, default=str),
                call_id=tc.get("id") or f"call_{i}_{uuid4().hex[:12]}",
            ))
        return calls


class Gemma4Protocol(ToolProtocol):
    name = "gemma4-control-tokens"
    _pattern = re.compile(
        r"<\|tool_call>\s*call:([A-Za-z_][\w.:-]*)\{(.*?)\}\s*<tool_call\|>",
        re.DOTALL,
    )

    @staticmethod
    def _parse_gemma_args(body: str) -> dict[str, Any]:
        # Gemma 4 strings use <|"|> delimiters. Parse key/value pairs without
        # treating braces or commas inside delimited strings as syntax.
        result: dict[str, Any] = {}
        pair = re.compile(
            r"([A-Za-z_]\w*)\s*:\s*(?:<\|\"\|>(.*?)<\|\"\|>|([^,}]*)?)",
            re.DOTALL,
        )
        for key, quoted, bare in pair.findall(body):
            raw = quoted if quoted != "" else (bare or "").strip()
            if quoted != "":
                value: Any = quoted
            else:
                try:
                    value = json.loads(raw)
                except Exception:
                    value = raw.strip("'\"")
            result[key] = value
        return result

    def parse(self, *, text: str, structured: list[dict[str, Any]],
              provider: str, model: str) -> list[CanonicalToolCall]:
        if structured:
            # Some Gemma servers already convert the native syntax to OAI deltas.
            # If that payload is malformed, continue to the native text fallback
            # instead of failing the whole adaptive protocol immediately.
            try:
                calls = OpenAIChatProtocol().parse(
                    text=text, structured=structured, provider=provider, model=model
                )
            except ProtocolError:
                calls = []
            if calls:
                for call in calls:
                    call.protocol = f"{self.name}+server-normalized"
                return calls
        return [
            _new_call(name, self._parse_gemma_args(body), provider=provider,
                      model=model, protocol=self.name, raw=whole, confidence=.98)
            for whole, name, body in (
                (m.group(0), m.group(1), m.group(2)) for m in self._pattern.finditer(text)
            )
        ]

    def append_result(self, messages: list[dict[str, Any]], result: ToolResult) -> None:
        # Keep an OpenAI-compatible tool message while adding the exact native
        # response block for servers that pass content directly to Gemma's template.
        messages.append({
            "role": "tool",
            "tool_call_id": result.call.call_id,
            "name": result.call.name,
            "content": render_tool_response(result.call.name, result.output, result.ok),
        })


class TaggedJSONProtocol(ToolProtocol):
    """Hermes/Qwen/GLM/generic local tagged JSON parser."""
    name = "tagged-json-tools"
    patterns = [
        re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
        re.compile(r"<\|tool_call\|>\s*(\{.*?\})\s*<\|/tool_call\|>", re.DOTALL),
        re.compile(r"```tool_call\s*(\{.*?\})\s*```", re.DOTALL),
    ]

    def parse(self, *, text: str, structured: list[dict[str, Any]],
              provider: str, model: str) -> list[CanonicalToolCall]:
        if structured:
            # Do not let malformed server-normalized arguments block a valid
            # tagged fallback carried in the assistant text.
            try:
                calls = OpenAIChatProtocol().parse(
                    text=text, structured=structured, provider=provider, model=model
                )
            except ProtocolError:
                calls = []
            if calls:
                for call in calls:
                    call.protocol = f"{self.name}+server-normalized"
                return calls
        out: list[CanonicalToolCall] = []
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                obj = _jsonish(match.group(1))
                if not isinstance(obj, dict):
                    continue
                fn = obj.get("function", obj)
                name = fn.get("name") or obj.get("tool")
                args = fn.get("arguments", obj.get("arguments", obj.get("parameters", {})))
                out.append(_new_call(name, args, provider=provider, model=model,
                                     protocol=self.name, raw=match.group(0), confidence=.92))
        return out


class HumoidEnvelopeProtocol(ToolProtocol):
    """Parser for the invented portable TOOL.CALL/v1 wire format."""
    name = "humoid-tool-call-v1"
    pattern = re.compile(
        r"⟪TOOL\.CALL/v1(?P<header>[^⟫]*)⟫\s*(?P<body>\{.*?\})\s*⟪/TOOL\.CALL⟫",
        re.DOTALL,
    )

    def parse(self, *, text: str, structured: list[dict[str, Any]],
              provider: str, model: str) -> list[CanonicalToolCall]:
        out: list[CanonicalToolCall] = []
        for match in self.pattern.finditer(text):
            body = _jsonish(match.group("body"))
            attrs = dict(re.findall(r"([\w.-]+)=([^\s]+)", match.group("header")))
            out.append(_new_call(
                body.get("tool") or body.get("name"),
                body.get("arguments", {}), provider=provider, model=model,
                protocol=self.name, raw=match.group(0), call_id=attrs.get("id"),
                confidence=.99, metadata=body.get("metadata", {}),
            ))
        return out


class CompositeProtocol(ToolProtocol):
    def __init__(self, name: str, protocols: Iterable[ToolProtocol]):
        self.name = name
        self.protocols = list(protocols)

    def parse(self, *, text: str, structured: list[dict[str, Any]],
              provider: str, model: str) -> list[CanonicalToolCall]:
        errors: list[str] = []
        for protocol in self.protocols:
            try:
                calls = protocol.parse(text=text, structured=structured,
                                       provider=provider, model=model)
                if calls:
                    return calls
            except ProtocolError as exc:
                errors.append(f"{protocol.name}: {exc}")
        if errors:
            raise ProtocolError("; ".join(errors))
        return []


class ProtocolRegistry:
    """Selects a parser/result codec from provider + model identity.

    Selection is deterministic and inspectable. It can be overridden with a
    protocol name from configuration for unusual serving templates.
    """
    def __init__(self):
        self.openai = OpenAIChatProtocol()
        self.gemma = Gemma4Protocol()
        self.tagged = TaggedJSONProtocol()
        self.humoid = HumoidEnvelopeProtocol()

    def resolve(self, provider: str, model: str, override: str = "auto") -> ToolProtocol:
        if override != "auto":
            mapping = {
                "openai": self.openai,
                "gemma4": self.gemma,
                "tagged-json": self.tagged,
                "humoid-v1": self.humoid,
            }
            if override not in mapping:
                raise ProtocolError(f"Unknown tool protocol override: {override}")
            return mapping[override]

        identity = f"{provider} {model}".lower()
        if "gemma-4" in identity or "gemma4" in identity:
            return CompositeProtocol("gemma4-adaptive", [self.openai, self.gemma, self.humoid, self.tagged])
        if any(x in identity for x in ("glm", "qwen", "hermes", "functionary", "deepseek")):
            return CompositeProtocol("glm-qwen-adaptive", [self.openai, self.tagged, self.humoid])
        if provider in {"openai", "digitalocean", "meta"}:
            return CompositeProtocol("hosted-openai-adaptive", [self.openai, self.humoid, self.tagged])
        if provider in {"llamacpp", "litert"}:
            return CompositeProtocol("local-adaptive", [self.openai, self.gemma, self.tagged, self.humoid])
        return CompositeProtocol("universal-adaptive", [self.openai, self.humoid, self.tagged, self.gemma])
