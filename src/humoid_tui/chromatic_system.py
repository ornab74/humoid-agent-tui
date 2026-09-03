from __future__ import annotations

"""Advanced chromatic code surfaces used by Rhomgaic retrieval.

The primitives in this module are deterministic acceleration structures.  They
never claim that a chromatic bucket, hierarchy key, or change address is truth.
They are compact hints that let later lexical/semantic/structural reranking do
less work.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

_MAINTENANCE_HINTS: dict[str, tuple[str, ...]] = {
    "api": ("api", "signature", "compatibility", "public", "interface", "contract"),
    "dependency": ("dependency", "dependencies", "import", "imports", "caller", "callers", "fanout"),
    "control": ("branch", "control flow", "condition", "loop", "match", "exception"),
    "regression": ("regression", "bug", "failure", "traceback", "broken", "repair"),
    "validation": ("test", "pytest", "validate", "validator", "lint", "typecheck", "benchmark"),
    "refactor": ("refactor", "cleanup", "maintain", "maintenance", "debt", "simplify"),
}


def _norm_path(value: str) -> str:
    return value.strip().replace("\\", "/").lstrip("./").lower()


def _digest_key(level: str, values: Iterable[str]) -> str:
    material = "\x1f".join([level, *(value for value in values if value)])
    return hashlib.blake2s(material.encode("utf-8", "ignore"), digest_size=8).hexdigest()


def _signed_bucket(value: int | float, *, scale: float = 1.0) -> int:
    if not math.isfinite(float(value)):
        return 128
    normalized = math.tanh(float(value) / max(1e-9, scale))
    return int(round((normalized + 1.0) * 127.5))


def path_scopes(paths: Sequence[str]) -> tuple[str, ...]:
    """Return bounded directory/module scopes from broad to specific."""

    scopes: list[str] = []
    for raw in paths[:16]:
        normalized = _norm_path(raw)
        if not normalized:
            continue
        path = Path(normalized)
        parts = list(path.parts)
        if len(parts) > 1:
            directories = parts[:-1]
            for depth in range(1, min(len(directories), 6) + 1):
                scopes.append("/".join(directories[:depth]))
        suffix = path.suffix
        stemmed = normalized[: -len(suffix)] if suffix else normalized
        if stemmed.endswith("/__init__"):
            stemmed = stemmed[: -len("/__init__")]
        dotted = stemmed.replace("/", ".").strip(".")
        if dotted:
            scopes.append(dotted)
            if dotted.startswith("src."):
                scopes.append(dotted[4:])
    return tuple(dict.fromkeys(value for value in scopes if value))[:48]


def behavior_terms(text: str, ast_terms: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    """Compress AST and maintenance intent into behavior-class coordinates."""

    values: list[str] = []
    kinds = {str(kind) for kind, _ in ast_terms}
    mapping = {
        "ast_def": "defines",
        "ast_class": "defines-type",
        "ast_call": "calls",
        "ast_import": "imports",
        "ast_raise": "raises",
        "ast_decorator": "decorated",
    }
    for kind, label in mapping.items():
        if kind in kinds:
            values.append(label)

    lowered = " ".join(text.lower().split())
    words = set(re.findall(r"[a-z][a-z0-9_-]+", lowered))
    for label, hints in _MAINTENANCE_HINTS.items():
        if any((hint in words) if " " not in hint else (hint in lowered) for hint in hints):
            values.append(label)
    return tuple(dict.fromkeys(values))[:24]


@dataclass(frozen=True, slots=True)
class ChromaticHierarchySurface:
    """Multi-resolution deterministic address pyramid for coding retrieval."""

    action: str
    language: str
    path_scopes: tuple[str, ...]
    symbols: tuple[str, ...]
    behaviors: tuple[str, ...]
    resolution_keys: tuple[tuple[str, str], ...]

    def terms(self) -> tuple[tuple[str, str], ...]:
        terms: list[tuple[str, str]] = []
        terms.extend(("path_scope", value) for value in self.path_scopes)
        terms.extend(("behavior", value) for value in self.behaviors)
        terms.extend((f"chromatic_{level}", key) for level, key in self.resolution_keys)
        return tuple(dict.fromkeys(terms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "language": self.language,
            "path_scopes": list(self.path_scopes),
            "symbols": list(self.symbols),
            "behaviors": list(self.behaviors),
            "resolution_keys": {level: key for level, key in self.resolution_keys},
        }


def build_hierarchy_surface(
    *,
    text: str,
    action: str,
    language: str,
    files: Sequence[str],
    symbols: Sequence[str],
    ast_terms: Sequence[tuple[str, str]],
) -> ChromaticHierarchySurface:
    scopes = path_scopes(files)
    behaviors = behavior_terms(text, ast_terms)
    file_values = tuple(_norm_path(value) for value in files if _norm_path(value))[:8]
    symbol_values = tuple(str(value).lower() for value in symbols if str(value).strip())[:12]

    # The levels intentionally become more specific.  Missing evidence simply
    # produces a coarser key; final ranking still uses exact evidence.
    l0 = _digest_key("l0", (language, action))
    l1 = _digest_key("l1", (language, action, *(scopes[:3] or ("global",))))
    l2 = _digest_key("l2", (language, action, *(file_values[:2] or scopes[-2:] or ("nofile",))))
    l3 = _digest_key(
        "l3",
        (
            language,
            action,
            *(file_values[:1] or ("nofile",)),
            *(symbol_values[:3] or ("nosymbol",)),
            *(behaviors[:3] or ("generic",)),
        ),
    )
    return ChromaticHierarchySurface(
        action=action,
        language=language,
        path_scopes=scopes,
        symbols=symbol_values,
        behaviors=behaviors,
        resolution_keys=(("l0", l0), ("l1", l1), ("l2", l2), ("l3", l3)),
    )


@dataclass(frozen=True, slots=True)
class ChromaticChangeDelta:
    """A compact 16-byte address for the *shape* of a code change."""

    address: bytes
    path: str
    risk_bucket: int
    public_api_change: bool
    parse_ok_after: bool
    fingerprint: str

    def distance(self, other: "ChromaticChangeDelta | bytes") -> float:
        address = other.address if isinstance(other, ChromaticChangeDelta) else bytes(other)
        if len(address) != 16:
            return 1.0
        # Structural-risk channels matter more than the fingerprint tail. File
        # and language are useful but should not prevent transfer of a repair
        # strategy to a neighboring module with the same change shape.
        weights = (0.35, 0.25, 1.4, 1.8, 2.0, 1.3, 1.0, 0.9, 0.9, 0.8, 0.8, 1.1, 0.55, 0.15, 0.15, 0.15)
        total = sum(weights)
        return sum(
            weight * abs(self.address[index] - address[index]) / 255.0
            for index, weight in enumerate(weights)
        ) / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address.hex(),
            "path": self.path,
            "risk_bucket": self.risk_bucket,
            "public_api_change": self.public_api_change,
            "parse_ok_after": self.parse_ok_after,
            "fingerprint": self.fingerprint,
            "channels": {
                "file": self.address[0],
                "language": self.address[1],
                "risk": self.address[2],
                "public_api": self.address[3],
                "parse_after": self.address[4],
                "signature_changes": self.address[5],
                "definitions_delta": self.address[6],
                "imports_delta": self.address[7],
                "calls_delta": self.address[8],
                "raises_delta": self.address[9],
                "branch_delta": self.address[10],
                "broad_except_delta": self.address[11],
                "line_delta": self.address[12],
                "fingerprint_0": self.address[13],
                "fingerprint_1": self.address[14],
                "fingerprint_2": self.address[15],
            },
        }


def encode_change_delta(fingerprint: dict[str, Any]) -> ChromaticChangeDelta:
    """Encode a SemanticChangeFingerprint-like mapping without source code."""

    path = _norm_path(str(fingerprint.get("path") or ""))[:512]
    language = str(fingerprint.get("language") or "text").lower()[:32]
    risk = max(0.0, min(10.0, float(fingerprint.get("risk_score", 0.0) or 0.0)))
    public_api = bool(fingerprint.get("public_api_change"))
    parse_ok_after = fingerprint.get("parse_ok_after") is not False

    added_defs = len(fingerprint.get("added_definitions") or ())
    removed_defs = len(fingerprint.get("removed_definitions") or ())
    added_imports = len(fingerprint.get("added_imports") or ())
    removed_imports = len(fingerprint.get("removed_imports") or ())
    added_calls = len(fingerprint.get("added_calls") or ())
    removed_calls = len(fingerprint.get("removed_calls") or ())
    added_raises = len(fingerprint.get("added_raises") or ())
    removed_raises = len(fingerprint.get("removed_raises") or ())
    signature_changes = len(fingerprint.get("signature_changes") or ())

    stable_fp = str(fingerprint.get("fingerprint") or "")
    if not stable_fp:
        stable_fp = hashlib.blake2s(
            json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8", "ignore"),
            digest_size=12,
        ).hexdigest()
    fp_bytes = hashlib.blake2s(stable_fp.encode("ascii", "ignore"), digest_size=3).digest()

    channels = bytes(
        [
            hashlib.blake2s(path.encode("utf-8", "ignore"), digest_size=1).digest()[0] if path else 0,
            hashlib.blake2s(language.encode("utf-8", "ignore"), digest_size=1).digest()[0],
            int(round(risk / 10.0 * 255.0)),
            255 if public_api else 0,
            255 if parse_ok_after else 0,
            min(255, signature_changes * 32),
            _signed_bucket(added_defs - removed_defs, scale=3.0),
            _signed_bucket(added_imports - removed_imports, scale=3.0),
            _signed_bucket(added_calls - removed_calls, scale=5.0),
            _signed_bucket(added_raises - removed_raises, scale=2.0),
            _signed_bucket(int(fingerprint.get("branch_delta", 0) or 0), scale=4.0),
            _signed_bucket(int(fingerprint.get("broad_except_delta", 0) or 0), scale=2.0),
            _signed_bucket(int(fingerprint.get("line_delta", 0) or 0), scale=40.0),
            fp_bytes[0],
            fp_bytes[1],
            fp_bytes[2],
        ]
    )
    return ChromaticChangeDelta(
        address=channels,
        path=path,
        risk_bucket=channels[2],
        public_api_change=public_api,
        parse_ok_after=parse_ok_after,
        fingerprint=stable_fp,
    )
