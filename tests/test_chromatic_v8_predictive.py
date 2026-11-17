from __future__ import annotations

import pytest

from humoid_tui.chromatic_invariants import ChromaticInvariantCompiler
from humoid_tui.chromatic_predictive import (
    ChromaticPredictiveMaintainer,
    evaluate_predictive_outcome,
)
from humoid_tui.maintenance import MaintenancePlanner, validator_candidates
from humoid_tui.rhomgaic_memory import RhomgaicMemory


def _repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "core.py").write_text(
        "def compute_total(items):\n    if not items:\n        raise ValueError('items')\n    return sum(items)\n",
        encoding="utf-8",
    )
    (root / "src" / "service.py").write_text(
        "from src.core import compute_total\n\ndef checkout(items):\n    return compute_total(items)\n",
        encoding="utf-8",
    )
    (root / "src" / "api.py").write_text(
        "from src.service import checkout\n\ndef submit(items):\n    return checkout(items)\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_core.py").write_text(
        "from src.core import compute_total\n\ndef test_total():\n    assert compute_total([1, 2]) == 3\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    return root


def _forecast(root, history=()):
    query = "Fix compute_total in src/core.py without breaking callers and verify tests"
    predictor = ChromaticPredictiveMaintainer(root)
    predictor.index.build()
    impact = predictor.index.impact(query, max_depth=5, max_nodes=72)
    contract = ChromaticInvariantCompiler(root, predictor.index).compile(
        query,
        operation="debug",
        files=["src/core.py"],
        symbols=["compute_total"],
        acceptance=["tests should pass"],
    )
    return predictor.forecast(
        query,
        operation="debug",
        contract=contract,
        historical=list(history),
        validators=validator_candidates(root, impact),
    )


def test_predictive_forecast_is_deterministic_and_current_code_grounded(tmp_path):
    root = _repo(tmp_path)
    first = _forecast(root)
    second = _forecast(root)

    assert first.risk_address_hex == second.risk_address_hex
    assert first.forecast_id == second.forecast_id
    assert len(first.risk_address_hex) == 32
    assert first.hot_paths
    assert any(item.path == "src/core.py" for item in first.hot_paths)
    assert first.validation_budget.linked_tests
    assert first.caveat.lower().startswith("risk priority is an uncalibrated")


def test_historical_failures_increase_planning_pressure_without_becoming_probability(tmp_path):
    root = _repo(tmp_path)
    baseline = _forecast(root)
    history = [
        {
            "procedure_id": "old-failure",
            "score": 0.95,
            "confidence": 0.9,
            "support_count": 12,
            "success_rate": 0.1,
            "files": ["src/core.py"],
            "symbols": ["compute_total"],
            "validators": ["python -m pytest tests/test_core.py"],
        }
    ]
    pressured = _forecast(root, history)

    assert pressured.historical_failure_pressure > 0.7
    assert pressured.overall_risk > baseline.overall_risk
    assert "not a defect probability" in ChromaticPredictiveMaintainer.render(pressured).lower()


def test_predictive_outcome_detects_change_radius_surprise(tmp_path):
    root = _repo(tmp_path)
    forecast = _forecast(root)
    outcome = evaluate_predictive_outcome(
        forecast,
        changed_paths=["src/core.py", "src/unrelated_new_surface.py"],
        failures=0,
        validated=True,
    )

    assert "src/unrelated_new_surface.py" in outcome.surprise_paths
    assert outcome.path_coverage < 1.0
    assert outcome.observation in {"change-radius-surprise", "forecast-compatible-success"}


@pytest.mark.asyncio
async def test_rhomgaic_recalls_prior_procedure_by_predictive_risk_shape(tmp_path):
    root = _repo(tmp_path)
    forecast = _forecast(root)
    memory = RhomgaicMemory(tmp_path / "rhomgaic.sqlite3")
    await memory.initialize()
    await memory.add(
        "predictive-procedure-memory",
        "CODING PROCEDURE MEMORY risky cross-file repair",
        memory_tier="procedural",
        validation_status="verified",
        metadata={"coding_trajectory": {
            "request": forecast.query,
            "operation": "debug",
            "changed_paths": ["src/core.py", "tests/test_core.py"],
            "symbols": ["compute_total"],
            "validators": ["python -m pytest tests/test_core.py"],
            "validated": True,
            "predictive_forecast": forecast.to_dict(),
            "predictive_outcome": {
                "forecast_id": forecast.forecast_id,
                "changed_paths": ["src/core.py"],
                "predicted_paths": ["src/core.py"],
                "surprise_paths": [],
                "path_coverage": 1.0,
                "failures": 0,
                "validated": True,
                "observation": "forecast-compatible-success",
            },
            "steps": [{"kind": "apply_patch"}, {"kind": "run_command"}],
        }},
    )

    hits = await memory.search_predictive_procedures(forecast.risk_address_hex, limit=4)

    assert hits
    assert hits[0]["risk_similarity"] > 0.99
    assert hits[0]["success_rate"] == 1.0
    assert hits[0]["predictive_source"] == "risk-address"


@pytest.mark.asyncio
async def test_maintenance_plan_contains_predictive_preflight(tmp_path):
    root = _repo(tmp_path)
    plan = await MaintenancePlanner(root).plan(
        "Fix compute_total in src/core.py without breaking callers and verify tests",
        memory=None,
        max_chars=24000,
    )

    assert "CHROMATIC PREDICTIVE PREFLIGHT" in plan
    assert "CHROMATIC PREDICTIVE MAINTENANCE" in plan
    assert "risk_priority=" in plan
    assert "VALIDATION / INSPECTION BUDGET" in plan


def test_runtime_exposes_v8_predictive_tools():
    from humoid_tui.maintenance_runtime import _schemas

    names = {item["function"]["name"] for item in _schemas()}
    assert "forecast_chromatic_maintenance" in names
    assert "inspect_chromatic_predictive_forecast" in names
    assert "recall_chromatic_regression_history" in names
