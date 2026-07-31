from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from .config import ProviderName, Settings
from .context_accordion import ContextAccordion
from .gemma4 import (
    Gemma4RuntimePolicy,
    gemma_system_suffix,
    strip_thought_channels,
)
from .memory import MemoryRouter
from .model_profiles import resolve_profile
from .models import AgentEvent
from .prompts import build_system_prompt
from .providers import UnifiedProvider
from .tool_protocols import (
    ProtocolError,
    ProtocolRegistry,
    ToolResult,
)
from .tools import ToolRegistry


SYSTEM = "Humoid dynamic system prompt"


class AgentHarness:
    """
    Main autonomous-agent execution harness.

    Responsibilities:

    - Provider selection and switching
    - Model-specific prompt construction
    - Tool-protocol selection
    - Streaming response aggregation
    - Tool-call parsing and execution
    - Tool-result continuation
    - Memory retrieval and persistence
    - Runtime telemetry events
    """

    def __init__(
        self,
        settings: Settings,
        emit: Callable[[AgentEvent], Awaitable[None]],
    ) -> None:
        self.s = settings
        self.emit = emit

        self.memory = MemoryRouter(settings)
        self.tools = ToolRegistry(settings)

        self.provider_name: ProviderName = (
            settings.humoid_provider
        )

        self.provider = UnifiedProvider(
            settings.provider(),
            settings,
        )

        self.protocols = ProtocolRegistry()

        self.gemma4 = Gemma4RuntimePolicy(
            mode=settings.humoid_gemma4_mode,
            thinking=settings.humoid_gemma4_thinking,
            native_fallback=(
                settings.humoid_gemma4_native_fallback
            ),
            strip_completed_thoughts=(
                settings.humoid_gemma4_strip_completed_thoughts
            ),
        )

        self.profile = resolve_profile(
            self.provider.cfg.name,
            self.provider.cfg.model,
        )

        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(
                    self.profile,
                    autonomy=settings.humoid_autonomy_mode,
                ),
            }
        ]
        self.context_accordion = ContextAccordion(settings.context_limit())

    @property
    def protocol(self):
        """
        Resolve the active tool-call protocol.

        Native Gemma mode overrides automatic protocol detection.
        """

        cfg = self.s.provider(self.provider_name)

        override = (
            "gemma4"
            if self.gemma4.mode == "native"
            else self.s.humoid_tool_protocol
        )

        return self.protocols.resolve(
            cfg.name,
            cfg.model,
            override,
        )

    def gemma4_active(self) -> bool:
        return self.gemma4.active(
            self.provider.cfg.model
        )

    def set_gemma4(
        self,
        mode: str | None = None,
        thinking: str | None = None,
    ) -> None:
        if mode is not None:
            if mode not in {
                "auto",
                "off",
                "native",
            }:
                raise ValueError(
                    "Gemma 4 mode must be auto, off, or native"
                )

            self.gemma4.mode = mode
            self.s.humoid_gemma4_mode = mode

        if thinking is not None:
            if thinking not in {
                "off",
                "low",
                "on",
            }:
                raise ValueError(
                    "Gemma 4 thinking must be off, low, or on"
                )

            self.gemma4.thinking = thinking
            self.s.humoid_gemma4_thinking = thinking

    def _base_system_prompt(self) -> str:
        return build_system_prompt(
            self.profile,
            autonomy=self.s.humoid_autonomy_mode,
        )

    def _request_messages(self) -> list[dict[str, Any]]:
        """
        Build request messages without mutating persisted conversation
        history.

        Gemma-native tool declarations are added only to the outgoing
        request copy.
        """

        folded_messages = self.context_accordion.fold(self.messages)
        if folded_messages is not self.messages:
            self.messages = folded_messages
        messages = [
            dict(message)
            for message in folded_messages
        ]

        # llama.cpp's OpenAI-compatible request model requires assistant
        # content to be a string.  Tool-only or otherwise empty model turns
        # may be represented as None by other providers, so normalize both
        # current and previously persisted session history at the transport
        # boundary.
        for message in messages:
            if (
                message.get("role") == "assistant"
                and message.get("content") is None
            ):
                message["content"] = ""

            # llama.cpp's bundled Gemma chat template expects historical
            # function arguments as mappings, even though the OpenAI wire
            # format normally represents them as JSON strings.  Convert only
            # the outgoing copy so persisted canonical tool calls remain
            # portable to other providers.
            if (
                self.provider.cfg.name == "llamacpp"
                and message.get("role") == "assistant"
                and isinstance(message.get("tool_calls"), list)
            ):
                normalized_calls = []
                for tool_call in message["tool_calls"]:
                    normalized_call = dict(tool_call)
                    function = dict(normalized_call.get("function") or {})
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            decoded = json.loads(arguments)
                        except json.JSONDecodeError:
                            decoded = None
                        if isinstance(decoded, dict):
                            function["arguments"] = decoded
                    normalized_call["function"] = function
                    normalized_calls.append(normalized_call)
                message["tool_calls"] = normalized_calls

        if not self.gemma4_active():
            return messages

        suffix = gemma_system_suffix(
            self.tools.schemas(),
            self.gemma4.thinking,
        )

        if (
            messages
            and messages[0].get("role") == "system"
        ):
            existing = str(
                messages[0].get("content") or ""
            )

            messages[0]["content"] = (
                existing + suffix
            )
        else:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        self._base_system_prompt()
                        + suffix
                    ),
                },
            )

        return messages

    async def initialize(self) -> None:
        """
        Initialize memory and verify the selected model provider.
        """

        await self.memory.initialize()

        await self.emit(
            AgentEvent(
                "memory",
                self.memory.status,
            )
        )

        provider_ok, provider_status = (
            await self.provider.health()
        )

        await self.emit(
            AgentEvent(
                "provider",
                provider_status,
                {
                    "ok": provider_ok,
                },
            )
        )

        await self.emit(
            AgentEvent(
                "protocol",
                self.protocol.name,
            )
        )

        await self.emit(
            AgentEvent(
                "profile",
                (
                    f"{self.profile.key} "
                    f"api={self.profile.preferred_api} "
                    f"parallel="
                    f"{self.profile.supports_parallel}"
                ),
            )
        )

        await self.emit(
            AgentEvent(
                "gemma4",
                (
                    f"mode={self.gemma4.mode} "
                    f"active={self.gemma4_active()} "
                    f"thinking={self.gemma4.thinking}"
                ),
            )
        )

    async def close(self) -> None:
        """
        Close memory connections and embedded services cleanly.
        """

        await self.memory.close()

    async def switch_provider(
        self,
        name: ProviderName,
    ) -> None:
        self.provider_name = name

        self.provider = UnifiedProvider(
            self.s.provider(name),
            self.s,
        )

        self.profile = resolve_profile(
            self.provider.cfg.name,
            self.provider.cfg.model,
        )

        if self.messages:
            self.messages[0]["content"] = (
                self._base_system_prompt()
            )

        provider_ok, provider_status = (
            await self.provider.health()
        )

        await self.emit(
            AgentEvent(
                "provider",
                provider_status,
                {
                    "ok": provider_ok,
                },
            )
        )

        await self.emit(
            AgentEvent(
                "protocol",
                self.protocol.name,
            )
        )

        await self.emit(
            AgentEvent(
                "profile",
                (
                    f"{self.profile.key} "
                    f"api={self.profile.preferred_api} "
                    f"parallel="
                    f"{self.profile.supports_parallel}"
                ),
            )
        )

        await self.emit(
            AgentEvent(
                "gemma4",
                (
                    f"mode={self.gemma4.mode} "
                    f"active={self.gemma4_active()} "
                    f"thinking={self.gemma4.thinking}"
                ),
            )
        )

    @staticmethod
    def _merge_deltas(
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Merge streamed OpenAI-compatible tool-call deltas.

        Tool names and JSON argument strings may arrive over multiple
        stream chunks.
        """

        merged: dict[int, dict[str, Any]] = {}

        for tool_call in chunks:
            try:
                index = int(
                    tool_call.get("index", 0)
                )
            except (TypeError, ValueError):
                index = 0

            slot = merged.setdefault(
                index,
                {
                    "id": "",
                    "type": "function",
                    "function": {
                        "name": "",
                        "arguments": "",
                    },
                },
            )

            call_id = tool_call.get("id")

            if call_id:
                slot["id"] = call_id

            function_delta = (
                tool_call.get("function")
                or {}
            )

            function_name = function_delta.get(
                "name"
            )

            if function_name:
                slot["function"]["name"] = (
                    function_name
                )

            argument_delta = function_delta.get(
                "arguments"
            )

            if argument_delta:
                slot["function"]["arguments"] += (
                    argument_delta
                )

        return [
            merged[index]
            for index in sorted(merged)
        ]

    @staticmethod
    def _event_value(
        event: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Read provider stream events defensively.

        Providers normally return dictionaries, but this also tolerates
        objects with matching attributes.
        """

        if isinstance(event, dict):
            return event.get(key, default)

        return getattr(event, key, default)

    def _clean_final_text(
        self,
        content: str,
    ) -> str:
        if (
            self.gemma4_active()
            and self.gemma4.strip_completed_thoughts
        ):
            return strip_thought_channels(
                content
            ).strip()

        return content.strip()

    async def _emit_complete_reasoning(
        self,
        reasoning: str,
    ) -> None:
        """
        Emit one complete reasoning block per model round.

        This avoids RichLog treating every streamed word or token as a
        separate terminal line.
        """

        cleaned = reasoning.strip()

        if not cleaned:
            return

        await self.emit(
            AgentEvent(
                "reasoning",
                cleaned,
            )
        )

    async def _emit_complete_answer(
        self,
        answer: str,
    ) -> None:
        """
        Emit the completed answer as one event.

        The existing TUI listens for the ``token`` event, so this keeps
        compatibility while preventing one-word-per-line rendering.
        """

        cleaned = answer.strip()

        if not cleaned:
            return

        await self.emit(
            AgentEvent(
                "token",
                cleaned,
            )
        )

    async def run(
        self,
        prompt: str,
    ) -> str:
        """
        Run an agent request through memory retrieval, model reasoning,
        tool execution and final memory persistence.
        """

        prompt = prompt.strip()

        if not prompt:
            return ""

        context = ""

        try:
            context = await self.memory.context_packet(
                prompt,
                limit=self.s.humoid_memory_context_limit,
            )
        except Exception as exc:
            await self.emit(
                AgentEvent(
                    "memory_error",
                    (
                        "Memory retrieval failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    {
                        "phase": "search",
                    },
                )
            )

        if context:
            user_content = (
                f"{prompt}\n\n"
                "Relevant memory from earlier activity:\n"
                f"{context}"
            )
        else:
            user_content = prompt

        self.messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        final_text = ""
        empty_response_retries = 0

        for round_no in range(
            self.s.humoid_max_tool_rounds
        ):
            await self.emit(
                AgentEvent(
                    "round",
                    (
                        "reasoning round "
                        f"{round_no + 1}"
                    ),
                    {
                        "round": round_no + 1,
                    },
                )
            )

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            structured_deltas: list[
                dict[str, Any]
            ] = []

            try:
                async for event in self.provider.stream(
                    self._request_messages(),
                    self.tools.schemas(),
                ):
                    text_chunk = (
                        self._event_value(
                            event,
                            "text",
                            "",
                        )
                        or ""
                    )

                    reasoning_chunk = (
                        self._event_value(
                            event,
                            "reasoning",
                            "",
                        )
                        or ""
                    )

                    tool_call_chunks = (
                        self._event_value(
                            event,
                            "tool_calls",
                            [],
                        )
                        or []
                    )

                    if text_chunk:
                        text_parts.append(
                            str(text_chunk)
                        )

                    if reasoning_chunk:
                        reasoning_parts.append(
                            str(reasoning_chunk)
                        )

                    if isinstance(
                        tool_call_chunks,
                        list,
                    ):
                        structured_deltas.extend(
                            tool_call_chunks
                        )

            except Exception as exc:
                error_text = (
                    "Provider stream failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                await self.emit(
                    AgentEvent(
                        "provider_error",
                        error_text,
                        {
                            "round": round_no + 1,
                        },
                    )
                )

                final_text = error_text
                break

            content = "".join(text_parts)

            reasoning_content = "".join(
                reasoning_parts
            )

            # Emit one reasoning block instead of one streamed chunk per
            # terminal line.
            await self._emit_complete_reasoning(
                reasoning_content
            )

            structured = self._merge_deltas(
                structured_deltas
            )

            try:
                calls = self.protocol.parse(
                    text=content,
                    structured=structured,
                    provider=self.provider.cfg.name,
                    model=self.provider.cfg.model,
                )

            except ProtocolError as exc:
                await self.emit(
                    AgentEvent(
                        "protocol_error",
                        str(exc),
                        {
                            "round": round_no + 1,
                            "protocol": self.protocol.name,
                        },
                    )
                )

                self.messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous tool call could not "
                            "be parsed. Retry once using the "
                            "model's native structured function-"
                            "call interface. Use an existing tool "
                            "name and a valid JSON object for its "
                            "arguments. Do not place explanatory "
                            "prose inside the tool-call payload."
                        ),
                    }
                )

                continue

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                # Keep empty assistant turns valid for stricter OpenAI-
                # compatible servers such as llama.cpp.
                "content": content,
            }

            if calls:
                assistant_message["tool_calls"] = [
                    call.openai_tool_call()
                    for call in calls
                ]

            self.messages.append(
                assistant_message
            )

            if not calls:
                final_text = self._clean_final_text(
                    content
                )

                if not final_text:
                    empty_response_retries += 1
                    if (
                        empty_response_retries
                        <= self.s.humoid_tool_retry_limit
                    ):
                        self.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "The previous turn was empty. Continue the "
                                    "requested task now: either call the next "
                                    "required tool or provide a concise final "
                                    "answer describing the completed work."
                                ),
                            }
                        )
                        await self.emit(
                            AgentEvent(
                                "protocol_error",
                                "Empty model response; requesting continuation",
                                {"round": round_no + 1},
                            )
                        )
                        continue

                    final_text = (
                        "The model returned an empty response after "
                        f"{empty_response_retries} attempts. Check the local "
                        "server log and retry the request."
                    )
                    self.messages[-1]["content"] = final_text
                    await self.emit(
                        AgentEvent(
                            "agent_error",
                            final_text,
                            {"reason": "empty_model_response"},
                        )
                    )
                    await self._emit_complete_answer(final_text)
                    break

                if final_text != content:
                    self.messages[-1]["content"] = (
                        final_text
                    )

                # Important fix:
                # send the complete final response to the TUI once.
                await self._emit_complete_answer(
                    final_text
                )

                break

            for call in calls:
                try:
                    serialized_arguments = json.dumps(
                        call.arguments,
                        ensure_ascii=False,
                    ).encode("utf-8")

                    argument_bytes = len(
                        serialized_arguments
                    )

                except Exception as exc:
                    argument_bytes = 0

                    result = ToolResult(
                        call,
                        (
                            "ERROR: tool arguments could not "
                            f"be serialized: {exc}"
                        ),
                        False,
                        0.0,
                    )

                    await self.emit(
                        AgentEvent(
                            "tool_result",
                            result.envelope(),
                            {
                                "tool": call.name,
                                "ok": False,
                            },
                        )
                    )

                    self.protocol.append_result(
                        self.messages,
                        result,
                    )

                    continue

                if (
                    argument_bytes
                    > self.s.humoid_tool_argument_max_bytes
                ):
                    output = (
                        "ERROR: argument payload "
                        f"{argument_bytes} bytes exceeds "
                        "configured limit "
                        f"{self.s.humoid_tool_argument_max_bytes}"
                    )

                    result = ToolResult(
                        call,
                        output,
                        False,
                        0.0,
                    )

                else:
                    await self.emit(
                        AgentEvent(
                            "tool_call",
                            call.envelope(),
                            {
                                "tool": call.name,
                                "protocol": (
                                    call.protocol
                                ),
                                "digest": call.digest,
                            },
                        )
                    )

                    started = time.perf_counter()

                    try:
                        output = await self.tools.execute(
                            call.name,
                            call.arguments,
                        )

                        output = str(output)

                        ok = not output.startswith(
                            "ERROR:"
                        )

                    except Exception as exc:
                        output = (
                            "ERROR: "
                            f"{type(exc).__name__}: {exc}"
                        )

                        ok = False

                    elapsed_ms = (
                        time.perf_counter() - started
                    ) * 1000.0

                    encoded_result = output.encode(
                        "utf-8",
                        errors="replace",
                    )

                    if (
                        len(encoded_result)
                        > self.s.humoid_tool_result_max_bytes
                    ):
                        allowed = (
                            self.s.humoid_tool_result_max_bytes
                        )

                        output = (
                            encoded_result[:allowed]
                            .decode(
                                "utf-8",
                                errors="ignore",
                            )
                            + "\n\n"
                            "[tool result truncated by "
                            "configured byte limit]"
                        )

                    result = ToolResult(
                        call,
                        output,
                        ok,
                        elapsed_ms,
                    )

                await self.emit(
                    AgentEvent(
                        "tool_result",
                        result.envelope(),
                        {
                            "tool": call.name,
                            "ok": result.ok,
                        },
                    )
                )

                self.protocol.append_result(
                    self.messages,
                    result,
                )

        else:
            final_text = (
                "The agent reached the configured maximum "
                f"of {self.s.humoid_max_tool_rounds} "
                "tool-reasoning rounds before producing a "
                "final response."
            )

            await self.emit(
                AgentEvent(
                    "agent_error",
                    final_text,
                    {
                        "reason": "max_tool_rounds",
                    },
                )
            )

            await self._emit_complete_answer(
                final_text
            )

        memory_text = (
            f"USER REQUEST:\n{prompt}\n\n"
            f"ASSISTANT RESPONSE:\n{final_text}"
        )

        try:
            memory_id = await self.memory.add(
                memory_text,
                memory_tier="episodic",
                channel="conversation",
                validation_status="observed",
                metadata={
                    "provider": (
                        self.provider.cfg.name
                    ),
                    "model": (
                        self.provider.cfg.model
                    ),
                    "protocol": (
                        self.protocol.name
                    ),
                },
            )

            await self.emit(
                AgentEvent(
                    "memory_write",
                    f"stored memory {memory_id}",
                    {
                        "memory_id": memory_id,
                    },
                )
            )

        except Exception as exc:
            await self.emit(
                AgentEvent(
                    "memory_error",
                    (
                        "Memory write failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    {
                        "phase": "write",
                    },
                )
            )

        return final_text
