from __future__ import annotations

"""Predictive chromatic maintenance for software changes.

This module does *not* predict defects as calibrated probabilities. It compiles
bounded, inspectable risk-priority surfaces from current repository topology,
chromatic invariant obligations, linked tests, and historical coding outcomes.
The output is a planning aid: current code, explicit invariants, and observed
validator results remain authoritative.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .chromatic_invariants import InvariantContract
from .code_intelligence import CodeIntelligenceIndex, ImpactNode, ImpactReport

PREDICTIVE_CHROMATIC_VERSION = 8


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _norm_path(value: str) -> str:
    return value.strip().replace("\\", "/").lstrip("./")


def _stable_byte(value: str, salt: bytes = b"") -> int:
    return hashlib.blake2s(value.encode("utf-8", "ignore"), key=salt, digest_size=1).digest()[0]


def _bucket01(value: float) -> int:
    return int(round(_clip01(value) * 255.0))


def _digest(value: str, size: int = 12) -> str:
    return hashlib.blake2s(value.encode("utf-8", "ignore"), digest_size=size).hexdigest()


def _language_from_paths(paths: Sequence[str]) -> str:
    aliases = {".pyi": "python", ".py": "python", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript", ".rs": "rust", ".go": "go", ".dart": "dart", ".java": "java", ".kt": "kotlin"}
    counts: Counter[str] = Counter()
    for raw in paths:
        suffix = Path(raw).suffix.lower()
        counts[aliases.get(suffix, suffix.lstrip(".") or "text")] += 1
    return counts.most_common(1)[0][0] if counts else "text"


@dataclass(frozen=True, slots=True)
class ChromaticRiskAddress:
    """Compact 16-byte address for the *planning shape* of change risk."""

    address: bytes

    def __post_init__(self) -> None:
        if len(self.address) != 16:
            raise ValueError("ChromaticRiskAddress requires exactly 16 bytes")

    def distance(self, other: "ChromaticRiskAddress | bytes") -> float:
        right = other.address if isinstance(other, ChromaticRiskAddress) else bytes(other)
        if len(right) != 16:
            return 1.0
        weights = (
            0.25, 0.20, 1.55, 1.15, 1.05, 1.10, 1.25, 1.20,
            1.25, 1.05, 1.35, 1.10, 1.10, 0.95, 0.35, 0.35,
        )
        total = sum(weights)
        return sum(
            weight * abs(self.address[index] - right[index]) / 255.0
            for index, weight in enumerate(weights)
        ) / total

    def to_dict(self) -> dict[str, Any]:
        labels = (
            "action", "language", "static_risk", "blast_radius", "graph_depth",
            "test_gap", "critical_invariants", "validation_obligations",
            "historical_failure_pressure", "historical_support", "api_surface",
            "dependency_surface", "exception_surface", "uncertainty",
            "shape_0", "shape_1",
        )
        return {
            "address": self.address.hex(),
            "channels": {label: self.address[index] for index, label in enumerate(labels)},
        }


@dataclass(frozen=True, slots=True)
class PredictiveRiskPath:
    path: str
    symbol: str
    role: str
    distance: int
    priority: float
    confidence: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidationBudget:
    tier: str
    inspection_paths: tuple[str, ...]
    validators: tuple[str, ...]
    linked_tests: tuple[str, ...]
    reasons: tuple[str, ...]
    max_graph_depth: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PredictiveMaintenanceForecast:
    forecast_id: str
    version: int
    query: str
    operation: str
    overall_risk: float
    confidence: float
    risk_address_hex: str
    estimated_change_radius: int
    current_static_risk: float
    historical_failure_pressure: float
    historical_support_strength: float
    uncertainty_score: float
    hot_paths: tuple[PredictiveRiskPath, ...]
    validation_budget: ValidationBudget
    risk_reasons: tuple[str, ...]
    historical_evidence: tuple[dict[str, Any], ...]
    caveat: str

    @property
    def risk_address(self) -> ChromaticRiskAddress:
        return ChromaticRiskAddress(bytes.fromhex(self.risk_address_hex))

    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast_id": self.forecast_id,
            "version": self.version,
            "query": self.query,
            "operation": self.operation,
            "overall_risk": self.overall_risk,
            "confidence": self.confidence,
            "risk_address_hex": self.risk_address_hex,
            "estimated_change_radius": self.estimated_change_radius,
            "current_static_risk": self.current_static_risk,
            "historical_failure_pressure": self.historical_failure_pressure,
            "historical_support_strength": self.historical_support_strength,
            "uncertainty_score": self.uncertainty_score,
            "hot_paths": [item.to_dict() for item in self.hot_paths],
            "validation_budget": self.validation_budget.to_dict(),
            "risk_reasons": list(self.risk_reasons),
            "historical_evidence": [dict(item) for item in self.historical_evidence],
            "caveat": self.caveat,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PredictiveMaintenanceForecast":
        budget = value.get("validation_budget") if isinstance(value.get("validation_budget"), dict) else {}
        return cls(
            forecast_id=str(value.get("forecast_id") or ""),
            version=int(value.get("version") or PREDICTIVE_CHROMATIC_VERSION),
            query=str(value.get("query") or ""),
            operation=str(value.get("operation") or "edit"),
            overall_risk=float(value.get("overall_risk") or 0.0),
            confidence=float(value.get("confidence") or 0.0),
            risk_address_hex=str(value.get("risk_address_hex") or "00" * 16),
            estimated_change_radius=int(value.get("estimated_change_radius") or 0),
            current_static_risk=float(value.get("current_static_risk") or 0.0),
            historical_failure_pressure=float(value.get("historical_failure_pressure") or 0.0),
            historical_support_strength=float(value.get("historical_support_strength") or 0.0),
            uncertainty_score=float(value.get("uncertainty_score") or 0.0),
            hot_paths=tuple(
                PredictiveRiskPath(
                    path=str(item.get("path") or ""),
                    symbol=str(item.get("symbol") or ""),
                    role=str(item.get("role") or "implementation"),
                    distance=int(item.get("distance") or 0),
                    priority=float(item.get("priority") or 0.0),
                    confidence=float(item.get("confidence") or 0.0),
                    reasons=tuple(str(reason) for reason in item.get("reasons", []) if reason),
                )
                for item in value.get("hot_paths", [])
                if isinstance(item, dict)
            ),
            validation_budget=ValidationBudget(
                tier=str(budget.get("tier") or "focused"),
                inspection_paths=tuple(str(item) for item in budget.get("inspection_paths", []) if item),
                validators=tuple(str(item) for item in budget.get("validators", []) if item),
                linked_tests=tuple(str(item) for item in budget.get("linked_tests", []) if item),
                reasons=tuple(str(item) for item in budget.get("reasons", []) if item),
                max_graph_depth=int(budget.get("max_graph_depth") or 2),
            ),
            risk_reasons=tuple(str(item) for item in value.get("risk_reasons", []) if item),
            historical_evidence=tuple(dict(item) for item in value.get("historical_evidence", []) if isinstance(item, dict)),
            caveat=str(value.get("caveat") or ""),
        )


@dataclass(frozen=True, slots=True)
class PredictiveOutcome:
    forecast_id: str
    changed_paths: tuple[str, ...]
    predicted_paths: tuple[str, ...]
    surprise_paths: tuple[str, ...]
    path_coverage: float
    failures: int
    validated: bool
    observation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _historical_pressure(history: Sequence[dict[str, Any]]) -> tuple[float, float, tuple[dict[str, Any], ...]]:
    weighted_failure = 0.0
    total_weight = 0.0
    support_mass = 0.0
    evidence: list[dict[str, Any]] = []
    for item in list(history)[:24]:
        score = _clip01(float(item.get("score", 0.0) or 0.0))
        confidence = _clip01(float(item.get("confidence", 0.0) or 0.0))
        success_rate = _clip01(float(item.get("success_rate", 0.5) or 0.5))
        support_count = max(1, int(item.get("support_count", 1) or 1))
        support = min(1.0, math.log1p(support_count) / math.log(17.0))
        weight = max(0.01, score * (0.55 + 0.45 * confidence) * (0.35 + 0.65 * support))
        weighted_failure += weight * (1.0 - success_rate)
        total_weight += weight
        support_mass += weight * support
        evidence.append({
            "procedure_id": str(item.get("procedure_id") or ""),
            "score": round(score, 6),
            "confidence": round(confidence, 6),
            "support_count": support_count,
            "success_rate": round(success_rate, 6),
            "files": list(item.get("files") or [])[:8],
            "symbols": list(item.get("symbols") or [])[:12],
            "validators": list(item.get("validators") or [])[:8],
            "source": str(item.get("predictive_source") or item.get("source") or "procedural"),
        })
    if total_weight <= 1e-9:
        return 0.0, 0.0, ()
    return (
        _clip01(weighted_failure / total_weight),
        _clip01(support_mass / total_weight),
        tuple(evidence[:12]),
    )


def _path_role(path: str) -> str:
    parts = {part.lower() for part in Path(path).parts}
    name = Path(path).name.lower()
    if "tests" in parts or name.startswith("test_") or name.endswith(("_test.py", ".spec.ts", ".test.ts", ".spec.js", ".test.js")):
        return "test"
    if Path(path).suffix.lower() in {".json", ".yaml", ".yml", ".toml", ".ini"}:
        return "configuration"
    if "migration" in name or "migrations" in parts:
        return "migration"
    return "implementation"


def encode_risk_address(
    *,
    query: str,
    operation: str,
    impact: ImpactReport,
    contract: InvariantContract,
    historical_failure_pressure: float,
    historical_support_strength: float,
) -> ChromaticRiskAddress:
    files = impact.files
    language = _language_from_paths(files)
    max_depth = max((node.distance for node in impact.frontier), default=0)
    total_nodes = len(impact.anchors) + len(impact.frontier)
    test_gap = 1.0 if not impact.tests else max(0.0, 1.0 - min(1.0, len(impact.tests) / max(1.0, len(files) * 0.4)))
    critical = sum(1 for item in contract.invariants if item.severity == "critical")
    validation_obligations = sum(1 for item in contract.invariants if item.requires_validation or item.kind == "validation-obligation")
    kinds = Counter(item.kind for item in contract.invariants)
    uncertainty = min(1.0, len(impact.uncertainty) / 4.0)
    shape = hashlib.blake2s(
        json.dumps(
            {
                "q": query.lower()[:512],
                "files": sorted(files)[:32],
                "kinds": sorted(kinds.items()),
                "reasons": sorted(impact.risk_reasons),
            },
            sort_keys=True,
        ).encode("utf-8", "ignore"),
        digest_size=2,
    ).digest()
    return ChromaticRiskAddress(bytes([
        _stable_byte(operation or "edit", b"action"),
        _stable_byte(language, b"lang"),
        _bucket01(min(1.0, impact.risk_score / 10.0)),
        _bucket01(min(1.0, math.log2(len(files) + 1) / 5.0)),
        _bucket01(min(1.0, max_depth / 5.0)),
        _bucket01(test_gap),
        _bucket01(min(1.0, critical / 6.0)),
        _bucket01(min(1.0, validation_obligations / 4.0)),
        _bucket01(historical_failure_pressure),
        _bucket01(historical_support_strength),
        _bucket01(min(1.0, kinds.get("public-signature", 0) / 3.0)),
        _bucket01(min(1.0, kinds.get("dependency-continuity", 0) / 4.0)),
        _bucket01(min(1.0, kinds.get("exception-continuity", 0) / 4.0)),
        _bucket01(uncertainty if total_nodes else 1.0),
        shape[0],
        shape[1],
    ]))


class ChromaticPredictiveMaintainer:
    """Compile an inspectable pre-edit risk-priority surface."""

    def __init__(self, root: Path, index: CodeIntelligenceIndex | None = None) -> None:
        self.root = root.resolve()
        self.index = index or CodeIntelligenceIndex(self.root)

    def _hot_paths(self, impact: ImpactReport, *, limit: int = 16) -> tuple[PredictiveRiskPath, ...]:
        incoming = Counter(edge.target for edge in self.index.edges)
        outgoing = Counter(edge.source for edge in self.index.edges)
        edge_kinds: dict[str, Counter[str]] = defaultdict(Counter)
        for edge in impact.edges:
            edge_kinds[edge.source][edge.kind] += 1
            edge_kinds[edge.target][edge.kind] += 1

        nodes: list[ImpactNode] = [*impact.anchors, *impact.frontier]
        candidates: list[PredictiveRiskPath] = []
        seen: set[tuple[str, str]] = set()
        for node in nodes:
            key = (node.path, node.symbol)
            if key in seen:
                continue
            seen.add(key)
            centrality = min(1.0, (incoming[node.symbol_id] + outgoing[node.symbol_id]) / 16.0)
            anchor_strength = 1.0 - math.exp(-max(0.0, node.score) / 2.5)
            distance_factor = 1.0 / (1.0 + 0.55 * max(0, node.distance))
            kinds = edge_kinds.get(node.symbol_id, Counter())
            structural = min(
                1.0,
                0.12 * kinds.get("calls", 0)
                + 0.16 * kinds.get("imports", 0)
                + 0.25 * kinds.get("inherits", 0)
                + 0.12 * kinds.get("tests", 0),
            )
            role = _path_role(node.path)
            role_weight = 0.72 if role == "test" else 1.0
            priority = _clip01(role_weight * (0.52 * anchor_strength * distance_factor + 0.30 * centrality + 0.18 * structural))
            confidence = _clip01(0.55 * anchor_strength + 0.25 * min(1.0, len(node.reasons) / 3.0) + 0.20 * distance_factor)
            reasons = list(node.reasons)
            if centrality >= 0.5:
                reasons.append(f"graph-centrality={centrality:.2f}")
            if structural >= 0.35:
                reasons.append(f"impact-edge-density={structural:.2f}")
            if role == "test":
                reasons.append("linked-validation-surface")
            candidates.append(PredictiveRiskPath(
                path=node.path,
                symbol=node.symbol,
                role=role,
                distance=node.distance,
                priority=priority,
                confidence=confidence,
                reasons=tuple(dict.fromkeys(reasons))[:8],
            ))
        candidates.sort(key=lambda item: (-item.priority, -item.confidence, item.distance, item.path, item.symbol))
        return tuple(candidates[: max(1, min(64, int(limit)))])

    @staticmethod
    def _validation_budget(
        *,
        overall_risk: float,
        impact: ImpactReport,
        hot_paths: Sequence[PredictiveRiskPath],
        validators: Sequence[str],
        critical_invariants: int,
        historical_failure_pressure: float,
    ) -> ValidationBudget:
        if overall_risk >= 7.0 or critical_invariants >= 4 or historical_failure_pressure >= 0.55:
            tier, graph_depth, inspect_n, validator_n = "defensive", 5, 14, 6
        elif overall_risk >= 5.0 or critical_invariants >= 2:
            tier, graph_depth, inspect_n, validator_n = "expanded", 4, 10, 4
        elif overall_risk >= 2.8:
            tier, graph_depth, inspect_n, validator_n = "focused", 3, 7, 3
        else:
            tier, graph_depth, inspect_n, validator_n = "minimal", 2, 4, 2
        reasons: list[str] = [f"risk-priority={overall_risk:.2f}"]
        if not impact.tests:
            reasons.append("no statically linked tests")
        if critical_invariants:
            reasons.append(f"critical invariants={critical_invariants}")
        if historical_failure_pressure >= 0.25:
            reasons.append(f"similar-history failure pressure={historical_failure_pressure:.2f}")
        if len(impact.files) > 6:
            reasons.append(f"cross-file frontier={len(impact.files)}")
        inspection_paths = tuple(dict.fromkeys(item.path for item in hot_paths))[:inspect_n]
        return ValidationBudget(
            tier=tier,
            inspection_paths=inspection_paths,
            validators=tuple(dict.fromkeys(str(item) for item in validators if str(item).strip()))[:validator_n],
            linked_tests=tuple(impact.tests[: min(12, max(2, validator_n * 2))]),
            reasons=tuple(reasons[:8]),
            max_graph_depth=graph_depth,
        )

    def forecast(
        self,
        query: str,
        *,
        operation: str = "edit",
        contract: InvariantContract,
        historical: Sequence[dict[str, Any]] = (),
        validators: Sequence[str] = (),
        max_nodes: int = 72,
    ) -> PredictiveMaintenanceForecast:
        self.index.build()
        impact = self.index.impact(query, max_depth=5, max_nodes=max(16, min(160, int(max_nodes))))
        failure_pressure, support_strength, historical_evidence = _historical_pressure(historical)
        hot_paths = self._hot_paths(impact)
        critical = sum(1 for item in contract.invariants if item.severity == "critical")
        required_validation = sum(1 for item in contract.invariants if item.requires_validation or item.kind == "validation-obligation")
        centrality = max((item.priority for item in hot_paths if item.role != "test"), default=0.0)
        blast = min(1.0, math.log2(len(impact.files) + 1) / 5.0)
        uncertainty = min(1.0, (len(impact.uncertainty) + (1 if not impact.anchors else 0)) / 4.0)
        invariant_pressure = min(1.0, (critical + 0.5 * required_validation) / 6.0)
        test_gap = 1.0 if not impact.tests else max(0.0, 0.45 - min(0.45, len(impact.tests) * 0.12))
        static = min(1.0, impact.risk_score / 10.0)
        risk_norm = _clip01(
            0.29 * static
            + 0.16 * blast
            + 0.13 * centrality
            + 0.13 * invariant_pressure
            + 0.10 * test_gap
            + 0.09 * uncertainty
            + 0.10 * failure_pressure
        )
        overall = round(10.0 * risk_norm, 4)
        evidence_current = min(1.0, (len(impact.anchors) + len(impact.edges) / 5.0 + len(contract.invariants) / 6.0) / 12.0)
        confidence = _clip01(0.78 * evidence_current + 0.22 * support_strength)
        # High uncertainty must lower confidence even when the risk priority is high.
        confidence = round(_clip01(confidence * (1.0 - 0.40 * uncertainty)), 4)
        address = encode_risk_address(
            query=query,
            operation=operation,
            impact=impact,
            contract=contract,
            historical_failure_pressure=failure_pressure,
            historical_support_strength=support_strength,
        )
        budget = self._validation_budget(
            overall_risk=overall,
            impact=impact,
            hot_paths=hot_paths,
            validators=validators,
            critical_invariants=critical,
            historical_failure_pressure=failure_pressure,
        )
        reasons = list(impact.risk_reasons)
        if critical:
            reasons.append(f"critical invariant obligations={critical}")
        if failure_pressure >= 0.20:
            reasons.append(f"similar procedural failure pressure={failure_pressure:.2f}")
        if not impact.tests:
            reasons.append("no linked test surface")
        if uncertainty:
            reasons.append(f"uncertainty={uncertainty:.2f}")
        estimated_radius = max(1, len({item.path for item in hot_paths if item.priority >= 0.20}) or len(impact.files) or 1)
        forecast_id = _digest(json.dumps({
            "query": query,
            "operation": operation,
            "contract": contract.surface_key,
            "address": address.address.hex(),
            "hot": [(item.path, item.symbol, round(item.priority, 4)) for item in hot_paths[:12]],
        }, sort_keys=True), 12)
        return PredictiveMaintenanceForecast(
            forecast_id=forecast_id,
            version=PREDICTIVE_CHROMATIC_VERSION,
            query=query,
            operation=operation,
            overall_risk=overall,
            confidence=confidence,
            risk_address_hex=address.address.hex(),
            estimated_change_radius=estimated_radius,
            current_static_risk=round(impact.risk_score, 4),
            historical_failure_pressure=round(failure_pressure, 4),
            historical_support_strength=round(support_strength, 4),
            uncertainty_score=round(uncertainty, 4),
            hot_paths=hot_paths,
            validation_budget=budget,
            risk_reasons=tuple(dict.fromkeys(reasons))[:16],
            historical_evidence=historical_evidence,
            caveat=(
                "Risk priority is an uncalibrated planning score, not a defect probability. "
                "Current code, invariant proof, and observed validators remain authoritative."
            ),
        )

    @staticmethod
    def render(forecast: PredictiveMaintenanceForecast | dict[str, Any], max_chars: int = 14000) -> str:
        if isinstance(forecast, dict):
            forecast = PredictiveMaintenanceForecast.from_dict(forecast)
        lines = [
            "[CHROMATIC PREDICTIVE MAINTENANCE: pre-edit risk-priority surface, not a defect probability]",
            f"forecast_id={forecast.forecast_id} version={forecast.version}",
            f"risk_priority={forecast.overall_risk:.3f}/10 confidence={forecast.confidence:.3f}",
            f"estimated_change_radius={forecast.estimated_change_radius}",
            f"static_risk={forecast.current_static_risk:.3f} historical_failure_pressure={forecast.historical_failure_pressure:.3f} historical_support={forecast.historical_support_strength:.3f}",
            f"risk_address={forecast.risk_address_hex}",
            f"validation_tier={forecast.validation_budget.tier} graph_depth={forecast.validation_budget.max_graph_depth}",
            f"risk_reasons={'; '.join(forecast.risk_reasons) or 'none'}",
            "",
            "HOT CHANGE / REGRESSION PATHS",
        ]
        if forecast.hot_paths:
            for item in forecast.hot_paths[:20]:
                lines.append(
                    f"- priority={item.priority:.3f} confidence={item.confidence:.3f} d={item.distance} "
                    f"role={item.role} {item.path}:{item.symbol} reasons={'; '.join(item.reasons) or 'none'}"
                )
        else:
            lines.append("- none")
        lines.extend(["", "VALIDATION / INSPECTION BUDGET"])
        lines.append(f"- inspect={','.join(forecast.validation_budget.inspection_paths) or 'none'}")
        lines.append(f"- linked_tests={','.join(forecast.validation_budget.linked_tests) or 'none'}")
        lines.append(f"- validators={'; '.join(forecast.validation_budget.validators) or 'none'}")
        lines.append(f"- reasons={'; '.join(forecast.validation_budget.reasons) or 'none'}")
        if forecast.historical_evidence:
            lines.extend(["", "SIMILAR HISTORICAL OUTCOME SIGNALS (non-authoritative)"])
            for item in forecast.historical_evidence[:8]:
                lines.append(
                    f"- id={item.get('procedure_id')} score={float(item.get('score', 0)):.3f} "
                    f"support={item.get('support_count', 0)} success_rate={float(item.get('success_rate', 0)):.3f} "
                    f"source={item.get('source')} validators={'; '.join(item.get('validators') or []) or 'none'}"
                )
        lines.extend(["", f"CAVEAT: {forecast.caveat}"])
        return "\n".join(lines)[: max(1000, int(max_chars))]


def evaluate_predictive_outcome(
    forecast: PredictiveMaintenanceForecast | dict[str, Any],
    *,
    changed_paths: Sequence[str],
    failures: int,
    validated: bool,
) -> PredictiveOutcome:
    if isinstance(forecast, dict):
        forecast = PredictiveMaintenanceForecast.from_dict(forecast)
    predicted = tuple(dict.fromkeys(_norm_path(item.path) for item in forecast.hot_paths if item.path))
    actual = tuple(dict.fromkeys(_norm_path(value) for value in changed_paths if _norm_path(value)))
    predicted_set = set(predicted)
    surprise = tuple(value for value in actual if value not in predicted_set)
    coverage = 1.0 if not actual else (len(set(actual) & predicted_set) / len(set(actual)))
    if failures > 0 and forecast.overall_risk < 3.0:
        observation = "underestimated-observed-cycle-difficulty"
    elif surprise and coverage < 0.6:
        observation = "change-radius-surprise"
    elif validated and failures == 0:
        observation = "forecast-compatible-success"
    elif failures > 0:
        observation = "forecast-compatible-repair-needed"
    else:
        observation = "incomplete-observation"
    return PredictiveOutcome(
        forecast_id=forecast.forecast_id,
        changed_paths=actual,
        predicted_paths=predicted[:32],
        surprise_paths=surprise,
        path_coverage=round(coverage, 4),
        failures=max(0, int(failures)),
        validated=bool(validated),
        observation=observation,
    )
