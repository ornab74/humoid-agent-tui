from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass(slots=True)
class MemoryHit:
    memory_id: str
    text: str
    score: float = 0.0
    memory_tier: str = "episodic"
    channel: str = "agent"
    task_id: str = ""
    validation_status: str = "unverified"
    created_at: str = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class AgentEvent:
    kind: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow)
    event_id: str = field(default_factory=lambda: str(uuid4()))
