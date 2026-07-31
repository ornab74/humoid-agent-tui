from __future__ import annotations

"""Runtime guard against planning-only and malformed-tool edit loops.

The normal agent loop remains provider-agnostic. This guard wraps it only for
explicit file-mutation requests and adds two bounded recovery paths:

1. Abort a run after repeated protocol parsing errors instead of consuming all
   configured reasoning rounds.
2. Retry once when the model ends with deferred-action prose ("let me write…")
   without producing a successful file mutation.

The wrapper preserves the normal tool executor, diff approval, undo stack,
Project Perspective synchronization, and final memory write.
"""

import os
import re
from functools import wraps
from typing import Any, Awaitable, Callable

from .agent import AgentHarness
from .models import AgentEvent


_MUTATION_TOOLS = {
    "write_file",
    "append_file",
    "insert_text",
    "replace_text",
    "apply_file_edits",
    "undo_file_change",
}

_MUTATION_WORDS = re.compile(
    r"\b(?:write|edit|append|insert|replace|modify|update|patch|inject|remove|delete|"
    r"add|continue adding|rewrite|refactor)\b",
    re.IGNORECASE,
)

_FILE_CLUES = re.compile(
    r"(?:\bfile\b|\bcode\b|\bproject\b|\brepo(?:sitory)?\b|\bREADME\b|"
    r"(?:^|[\s`'\"])(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]{1,10}(?:$|[\s`'\",:;)]))",
    re.IGNORECASE,
)

_DEFERRED_ACTION = re.compile(
    r"\b(?:let me|i need to|i have to|i(?:'|’)ll|i will|i am going to|i'm going to|"
    r"next i(?:'|’)ll|now i(?:'|’)ll|time to)\b.{0,220}\b(?:write|edit|append|insert|"
    r"replace|modify|update|patch|inject|rewrite|construct|apply)\b|"
    r"\b(?:write|edit|append|insert|replace|modify|update|patch|inject|rewrite)\b.{0,120}"
    r"\b(?:now|next|shortly|after this)\b",
    re.IGNORECASE | re.DOTALL,
)

_MAX_ROUNDS_TEXT = re.compile(
    r"reached the configured maximum.*tool-reasoning rounds",
    re.IGNORECASE | re.DOTALL,
)


class _RetryAction(RuntimeError):
    """Internal control-flow signal raised from the guarded event callback."""


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 8) -> int:
    try:
        return min(maximum, max(minimum, int(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


def mutation_requested(prompt: str) -> bool:
    """Conservatively identify explicit file/code mutation requests."""

    return bool(_MUTATION_WORDS.search(prompt) and _FILE_CLUES.search(prompt))


def deferred_action(text: str) -> bool:
    """Identify prose that promises a mutation instead of executing one."""

    return bool(_DEFERRED_ACTION.search(text or ""))


def _force_action_prompt(original_prompt: str, prior_result: str) -> str:
    prior = " ".join((prior_result or "").split())[:700]
    return (
        "ACTION GUARD: the previous attempt described a file edit but did not complete one. "
        "Do not explain, plan, reconstruct the full existing file, or say what you will do. "
        "Call exactly one smallest valid mutation tool now. Use append_file for an end append, "
        "insert_text for a single stable anchor, replace_text for one exact replacement, or "
        "apply_file_edits for coordinated insertions and summary/table updates. If exact anchor "
        "text is missing, call read_file_range once, then perform the mutation on the following "
        "turn. Keep tool arguments valid JSON and copy anchors exactly from observed evidence.\n\n"
        f"ORIGINAL OBJECTIVE:\n{original_prompt}\n\n"
        f"PRIOR STALLED OUTPUT:\n{prior or '(protocol parsing failed before a final answer)'}"
    )


async def _store_final_memory(
    harness: AgentHarness,
    add_memory: Callable[..., Awaitable[str]],
    prompt: str,
    final_text: str,
) -> None:
    memory_text = f"USER REQUEST:\n{prompt}\n\nASSISTANT RESPONSE:\n{final_text}"
    try:
        memory_id = await add_memory(
            memory_text,
            memory_tier="episodic",
            channel="conversation",
            validation_status="observed",
            metadata={
                "provider": harness.provider.cfg.name,
                "model": harness.provider.cfg.model,
                "protocol": harness.protocol.name,
                "action_guard": True,
            },
        )
        await harness.emit(
            AgentEvent(
                "memory_write",
                f"stored memory {memory_id}",
                {"memory_id": memory_id, "action_guard": True},
            )
        )
    except Exception as exc:
        await harness.emit(
            AgentEvent(
                "memory_error",
                f"Memory write failed: {type(exc).__name__}: {exc}",
                {"phase": "write", "action_guard": True},
            )
        )


async def _guarded_run(
    harness: AgentHarness,
    original_run: Callable[[AgentHarness, str], Awaitable[str]],
    prompt: str,
) -> str:
    if not mutation_requested(prompt):
        return await original_run(harness, prompt)

    attempts = _env_int("HUMOID_ACTION_GUARD_ATTEMPTS", 2, maximum=4)
    protocol_limit = _env_int(
        "HUMOID_ACTION_GUARD_PROTOCOL_ERRORS",
        max(1, int(getattr(harness.s, "humoid_tool_retry_limit", 2))),
        maximum=4,
    )

    original_emit = harness.emit
    original_memory_add = harness.memory.add
    initial_undo_depth = len(harness.tools.undo_stack)
    current_prompt = prompt
    final_text = ""

    async def deferred_memory_add(*args: Any, **kwargs: Any) -> str:
        return "action-guard-deferred-memory"

    try:
        # Prevent intermediate planning-only attempts from polluting durable
        # memory. One final memory is written after the guard resolves.
        harness.memory.add = deferred_memory_add  # type: ignore[method-assign]

        for attempt in range(1, attempts + 1):
            protocol_errors = 0
            buffered_answers: list[str] = []

            async def guarded_emit(event: AgentEvent) -> None:
                nonlocal protocol_errors

                if event.kind == "token":
                    buffered_answers.append(event.message)
                    return

                if event.kind == "memory_write":
                    return

                if event.kind == "agent_error" and event.detail.get("reason") == "max_tool_rounds":
                    # Do not flash a terminal failure when the guard is about
                    # to issue a bounded recovery attempt.
                    return

                if event.kind == "protocol_error" and "Empty model response" not in event.message:
                    protocol_errors += 1
                    await original_emit(event)
                    if protocol_errors >= protocol_limit:
                        raise _RetryAction(
                            f"stopped after {protocol_errors} consecutive malformed tool calls"
                        )
                    return

                await original_emit(event)

            harness.emit = guarded_emit

            retry_reason = ""
            try:
                final_text = await original_run(harness, current_prompt)
            except _RetryAction as exc:
                retry_reason = str(exc)
                final_text = retry_reason
            finally:
                harness.emit = original_emit

            mutated = len(harness.tools.undo_stack) > initial_undo_depth
            planning_stall = not mutated and deferred_action(final_text)
            round_stall = not mutated and bool(_MAX_ROUNDS_TEXT.search(final_text or ""))
            protocol_stall = not mutated and bool(retry_reason)

            if not (planning_stall or round_stall or protocol_stall):
                if buffered_answers:
                    await original_emit(AgentEvent("token", buffered_answers[-1]))
                await _store_final_memory(harness, original_memory_add, prompt, final_text)
                return final_text

            if attempt < attempts:
                reason = (
                    retry_reason
                    or "planning-only mutation response"
                    if planning_stall
                    else "maximum rounds reached without a mutation"
                )
                await original_emit(
                    AgentEvent(
                        "action_guard",
                        f"{reason}; forcing a surgical tool action (attempt {attempt + 1}/{attempts})",
                        {
                            "attempt": attempt + 1,
                            "attempts": attempts,
                            "protocol_errors": protocol_errors,
                            "mutation_observed": mutated,
                        },
                    )
                )
                current_prompt = _force_action_prompt(prompt, final_text)
                continue

            final_text = (
                "Action guard stopped the task after repeated malformed or planning-only edit turns. "
                "No successful file mutation was observed. Inspect the latest protocol error or use a "
                "smaller append_file, insert_text, replace_text, or apply_file_edits call."
            )
            if not harness.messages or harness.messages[-1].get("content") != final_text:
                harness.messages.append({"role": "assistant", "content": final_text})
            await original_emit(
                AgentEvent(
                    "agent_error",
                    final_text,
                    {
                        "reason": "action_stall",
                        "attempts": attempts,
                        "protocol_errors": protocol_errors,
                    },
                )
            )
            await original_emit(AgentEvent("token", final_text))
            await _store_final_memory(harness, original_memory_add, prompt, final_text)
            return final_text

        return final_text
    finally:
        harness.emit = original_emit
        harness.memory.add = original_memory_add  # type: ignore[method-assign]


def install_action_guard() -> None:
    """Install the bounded action guard exactly once on AgentHarness.run."""

    current = AgentHarness.run
    if getattr(current, "_humoid_action_guard", False):
        return

    @wraps(current)
    async def guarded(self: AgentHarness, prompt: str) -> str:
        return await _guarded_run(self, current, prompt)

    setattr(guarded, "_humoid_action_guard", True)
    AgentHarness.run = guarded  # type: ignore[method-assign]
