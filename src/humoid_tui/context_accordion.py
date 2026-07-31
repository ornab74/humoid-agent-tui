from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ContextArchive:
    summary: str
    decisions: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    tool_evidence: list[str] = field(default_factory=list)
    original_messages: list[dict[str, Any]] = field(default_factory=list, repr=False)


class ContextAccordion:
    """Reversible, structured context folding rather than lossy flat summaries."""

    def __init__(self, context_limit: int, threshold: float = 0.72) -> None:
        self.context_limit = context_limit
        self.threshold = threshold
        self.archives: list[ContextArchive] = []

    @staticmethod
    def estimate(messages: list[dict[str, Any]]) -> int:
        return sum(len(str(message.get("content") or "")) for message in messages) // 4

    def fold(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.estimate(messages) < self.context_limit * self.threshold or len(messages) < 8:
            return messages
        system, body = messages[0], messages[1:]
        fold_count = max(2, len(body) // 2)
        selected = body[:fold_count]
        archive = self._archive(selected)
        self.archives.append(archive)
        folded = {
            "role": "system",
            "content": self.render_archive(archive, len(self.archives) - 1),
        }
        return [system, folded, *body[fold_count:]]

    def _archive(self, messages: list[dict[str, Any]]) -> ContextArchive:
        decisions: list[str] = []
        unresolved: list[str] = []
        files: set[str] = set()
        evidence: list[str] = []
        summaries: list[str] = []
        for message in messages:
            role = str(message.get("role", "unknown"))
            content = str(message.get("content") or "")
            compact = " ".join(content.split())
            summaries.append(f"{role}: {compact[:400]}")
            lowered = content.lower()
            if any(word in lowered for word in ("decided", "will use", "selected", "fixed")):
                decisions.append(compact[:300])
            if any(word in lowered for word in ("todo", "unresolved", "remaining", "failed", "error")):
                unresolved.append(compact[:300])
            for token in content.replace("`", " ").split():
                if "/" in token or token.endswith((".py", ".md", ".toml", ".json", ".env")):
                    files.add(token.strip(".,:;()[]{}"))
            if role == "tool" or "TOOL.RESULT" in content:
                evidence.append(compact[:300])
        return ContextArchive(
            summary="\n".join(summaries), decisions=decisions[:12],
            unresolved=unresolved[:12], files=sorted(files)[:30],
            tool_evidence=evidence[:12], original_messages=[dict(item) for item in messages],
        )

    @staticmethod
    def render_archive(archive: ContextArchive, index: int) -> str:
        def block(title: str, values: list[str]) -> str:
            return f"{title}:\n" + ("\n".join(f"- {value}" for value in values) or "- none")
        return "\n\n".join([
            f"[CONTEXT ACCORDION #{index}: folded history; expandable from retained source]",
            f"SUMMARY:\n{archive.summary}", block("DECISIONS", archive.decisions),
            block("UNRESOLVED", archive.unresolved), block("FILE REFERENCES", archive.files),
            block("TOOL EVIDENCE", archive.tool_evidence),
        ])

    def expanded(self, index: int) -> list[dict[str, Any]]:
        return [dict(item) for item in self.archives[index].original_messages]

