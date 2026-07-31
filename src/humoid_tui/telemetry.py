from __future__ import annotations
from dataclasses import dataclass, field
from time import perf_counter

@dataclass(slots=True)
class SessionTelemetry:
    started: float = field(default_factory=perf_counter)
    messages: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    tokens_estimate: int = 0
    last_latency_ms: float = 0.0
    context_limit: int = 20000

    @property
    def elapsed(self) -> float:
        return perf_counter() - self.started

    @property
    def context_percent(self) -> int:
        return min(100, int(self.tokens_estimate / max(1, self.context_limit) * 100))
