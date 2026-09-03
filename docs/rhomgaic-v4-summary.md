# Rhomgaic / Chromatic Memory v4

Phase 4 adds adaptive retrieval telemetry, temporal/importance ranking, deterministic semantic capsule distillation, and cross-file AST dependency edges. All four are rebuildable sidecar state; Weaviate/SQLite remains the durable authority.

Key invariants:

1. Chromatic addresses and capsules narrow/rerank candidates; they are not truth.
2. Importance decay changes rank, never silently deletes durable memory.
3. Adaptive policy changes breadth/graph depth by at most one step and trains only on auto-depth ambiguity/novelty telemetry.
4. Forced recall is logged but cannot bias the adaptive policy.
5. Dependency edges are derived from AST/file surfaces and are safe to rebuild.

## Smoke validation

These figures are local synthetic smoke measurements from the build sandbox, not production guarantees:

- Phase-4 substrate: 400 base memories + 2 dependency memories, 60 queries, exact-target top-1 `1.000`, mean search about `9.0 ms`; 100 semantic capsules distilled in about `9.3 ms`; cross-file dependency recall succeeded.
- General Rhomgaic benchmark: 1,000 memories / 100 queries, exact-target top-1 `1.000`; mean search about `13.5 ms`.
- Compiled working-context cache: 300 memories / 40 repeated recalls, cold build about `47.1 ms`, hot mean about `0.31 ms` in the sandbox run.

The benchmark harnesses are under `benchmarks/` and should be rerun on the target machine before making performance claims.
