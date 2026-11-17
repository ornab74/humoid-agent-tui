from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from humoid_tui.chromatic_invariants import ChromaticInvariantCompiler
from humoid_tui.chromatic_predictive import ChromaticPredictiveMaintainer
from humoid_tui.maintenance import validator_candidates
from humoid_tui.rhomgaic_memory import RhomgaicMemory


def build_repo(root: Path, modules: int) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(exist_ok=True)
    for i in range(modules):
        source = [f"def worker_{i}(value):", f"    return value + {i}", ""]
        if i > 0:
            source = [
                f"from src.mod_{i-1} import worker_{i-1}",
                "",
                f"def worker_{i}(value):",
                f"    return worker_{i-1}(value) + {i}",
                "",
            ]
        (root / "src" / f"mod_{i}.py").write_text("\n".join(source), encoding="utf-8")
        if i % 8 == 0:
            (root / "tests" / f"test_mod_{i}.py").write_text(
                f"from src.mod_{i} import worker_{i}\n\ndef test_worker_{i}():\n    assert isinstance(worker_{i}(1), int)\n",
                encoding="utf-8",
            )
    (root / "pyproject.toml").write_text("[project]\nname='chromatic-v8-bench'\n", encoding="utf-8")


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[index]


async def run(modules: int, queries: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="chromatic-v8-") as tmp:
        root = Path(tmp) / "repo"
        build_repo(root, modules)
        predictor = ChromaticPredictiveMaintainer(root)
        t0 = time.perf_counter()
        predictor.index.build()
        build_ms = (time.perf_counter() - t0) * 1000.0

        memory = RhomgaicMemory(Path(tmp) / "rhomgaic.sqlite3")
        await memory.initialize()

        # Seed a few durable procedure shapes, including failures, so the
        # two-pass risk-address retrieval path is exercised.
        for i in range(min(40, modules)):
            query = f"fix src/mod_{i}.py worker_{i} regression without breaking callers and verify tests"
            impact = predictor.index.impact(query, max_depth=5, max_nodes=72)
            contract = ChromaticInvariantCompiler(root, predictor.index).compile(
                query, operation="debug", files=[f"src/mod_{i}.py"], symbols=[f"worker_{i}"], acceptance=["verify tests"]
            )
            forecast = predictor.forecast(
                query, operation="debug", contract=contract,
                validators=validator_candidates(root, impact),
            )
            await memory.add(
                f"procedure-{i}",
                f"CODING PROCEDURE MEMORY worker_{i} repair",
                memory_tier="procedural",
                validation_status="verified" if i % 5 else "observed",
                metadata={"coding_trajectory": {
                    "request": query,
                    "operation": "debug",
                    "changed_paths": [f"src/mod_{i}.py"],
                    "symbols": [f"worker_{i}"],
                    "validators": ["python -m pytest"],
                    "validated": i % 5 != 0,
                    "failures": 1 if i % 5 == 0 else 0,
                    "predictive_forecast": forecast.to_dict(),
                    "steps": [{"kind": "apply_patch"}, {"kind": "run_command"}],
                }},
            )

        latencies: list[float] = []
        history_latencies: list[float] = []
        hot_path_hits = 0
        risk_values: list[float] = []
        for n in range(queries):
            i = (n * 7) % modules
            query = f"fix src/mod_{i}.py worker_{i} regression without breaking callers and verify tests issue_{n}"
            impact = predictor.index.impact(query, max_depth=5, max_nodes=72)
            contract = ChromaticInvariantCompiler(root, predictor.index).compile(
                query, operation="debug", files=[f"src/mod_{i}.py"], symbols=[f"worker_{i}"], acceptance=["verify tests"]
            )
            start = time.perf_counter()
            provisional = predictor.forecast(
                query, operation="debug", contract=contract,
                validators=validator_candidates(root, impact),
            )
            hist_start = time.perf_counter()
            history = await memory.predictive_maintenance_history(
                query, contract.to_dict(), risk_address_hex=provisional.risk_address_hex, limit=10
            )
            history_latencies.append((time.perf_counter() - hist_start) * 1000.0)
            forecast = predictor.forecast(
                query, operation="debug", contract=contract, historical=history,
                validators=validator_candidates(root, impact),
            )
            latencies.append((time.perf_counter() - start) * 1000.0)
            risk_values.append(forecast.overall_risk)
            if any(item.path == f"src/mod_{i}.py" for item in forecast.hot_paths[:6]):
                hot_path_hits += 1

        return {
            "modules": modules,
            "queries": queries,
            "index_build_ms": round(build_ms, 3),
            "mean_forecast_ms": round(statistics.fmean(latencies), 3),
            "p50_forecast_ms": round(statistics.median(latencies), 3),
            "p95_forecast_ms": round(percentile(latencies, 0.95), 3),
            "mean_risk_history_ms": round(statistics.fmean(history_latencies), 3),
            "hot_path_hit_rate_top6": round(hot_path_hits / max(1, queries), 4),
            "mean_risk_priority": round(statistics.fmean(risk_values), 4),
            "note": "synthetic sandbox benchmark; risk score is an uncalibrated planning priority",
        }


def main() -> None:
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--modules", type=int, default=200)
    parser.add_argument("--queries", type=int, default=60)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.modules, args.queries)), indent=2))


if __name__ == "__main__":
    main()
