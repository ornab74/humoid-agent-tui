from __future__ import annotations

import re
from pathlib import Path


class EnvStore:
    """Small, comment-preserving writer for the project's .env file."""

    _KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(self, path: Path = Path(".env")) -> None:
        self.path = path

    def values(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        result: dict[str, str] = {}
        for raw in self.path.read_text(errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip().strip('"').strip("'")
        return result

    def set(self, key: str, value: object) -> None:
        key = key.strip().upper()
        if not self._KEY.fullmatch(key):
            raise ValueError("Invalid environment setting name")
        text = str(value).replace("\n", "\\n")
        if any(character.isspace() for character in text) or "#" in text:
            text = '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
        lines = self.path.read_text(errors="replace").splitlines() if self.path.exists() else []
        replacement = f"{key}={text}"
        for index, line in enumerate(lines):
            if line.lstrip().startswith(f"{key}="):
                lines[index] = replacement
                break
        else:
            if lines and lines[-1]:
                lines.append("")
            lines.append(replacement)
        self.path.write_text("\n".join(lines) + "\n")

