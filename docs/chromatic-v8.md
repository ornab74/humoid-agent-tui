# Chromatic v8 — Predictive Maintenance Surfaces

Chromatic v8 adds a **pre-edit predictive maintenance surface** for software engineering. It is intentionally conservative about the word *predictive*: the system produces an **uncalibrated risk-priority score**, not a probability that a defect will occur.

Current repository structure, explicit invariant checks, semantic change fingerprints, and observed validator outcomes remain authoritative.

## Three chromatic geometries

The coding stack now separates:

1. **location geometry** — project → package → file → symbol/behavior,
2. **change geometry** — API/signature/import/call/exception/control-flow deltas,
3. **risk geometry** — blast radius, topology depth, test gaps, invariant density, historical failure pressure, and uncertainty.

A risk surface is represented by a deterministic 16-byte `ChromaticRiskAddress`. Similar historical procedures can therefore be retrieved because they occupied a similar engineering-risk shape even when the bug description and files differ.

## Risk address channels

The compact v8 address contains channels for:

- coding action,
- dominant language,
- current static impact risk,
- blast-radius size,
- graph depth,
- linked-test gap,
- critical invariant density,
- validation obligations,
- similar-history failure pressure,
- historical support strength,
- public API exposure,
- dependency-continuity exposure,
- exception-contract exposure,
- uncertainty,
- two stable shape projections.

The address is an acceleration and transfer-learning hint. It is never evidence by itself.

## Forecast compiler

`ChromaticPredictiveMaintainer` combines:

- `CodeIntelligenceIndex.impact(...)`,
- the v7 `InvariantContract`,
- repository-supported validator candidates,
- historical procedural outcomes from Rhomgaic,
- graph centrality and typed impact edges.

The result contains:

- a 0–10 risk-priority score,
- forecast confidence based on available evidence,
- estimated change radius,
- prioritized implementation/test paths,
- a validation tier (`minimal`, `focused`, `expanded`, `defensive`),
- a bounded inspection budget,
- similar historical outcome evidence,
- the 16-byte risk address.

## Two-pass chromatic history

Prediction uses a bounded two-pass retrieval:

1. semantic + invariant-shaped history produces an initial forecast,
2. the resulting risk address retrieves prior procedures by **risk-shape similarity**,
3. the forecast is recomputed with that additional evidence.

This avoids making the risk coordinate depend exclusively on language similarity while still keeping the loop deterministic and inspectable.

## Edit-cycle integration

At edit-cycle start, Humoid can now compile a predictive surface before any mutation. The edit prompt exposes:

- forecast ID,
- risk priority,
- confidence,
- validation tier,
- hottest predicted paths.

The predictive score does **not** gate an edit. Only grounded repository rules, invariant proof failures, and observed validation can block/conclude the cycle.

At cycle completion Humoid records a `PredictiveOutcome` containing:

- actual changed paths,
- predicted hot paths,
- surprise paths,
- path coverage,
- observed cycle failures,
- validation completion.

This is stored inside durable procedural trajectory metadata so future Rhomgaic indexes can learn from observed maintenance outcomes without persisting source bodies.

## Tools

- `forecast_chromatic_maintenance` — compile a fresh predictive surface.
- `inspect_chromatic_predictive_forecast` — inspect the active cycle forecast plus current forecast-vs-observed path coverage.
- `recall_chromatic_regression_history` — retrieve similar historical outcome surfaces.

The existing `plan_maintenance_change` output now includes a predictive preflight alongside current impact analysis and invariant compilation.

## Safety / correctness invariant

The order of authority remains:

1. current source and repository structure,
2. explicit invariant checks,
3. observed validator results,
4. semantic change fingerprints,
5. predictive and historical risk surfaces.

Historical failure rates and risk-priority scores are planning signals only. They are not calibrated defect probabilities and must not be presented as such.
