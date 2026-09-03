# Rhomgaic / Chromatic Memory Phase 3

Phase 3 extends the Python coding-memory substrate with:

- deterministic Python AST structural terms (`class`, `def`, calls, imports, decorators, raises);
- privacy-preserving verified-code surfaces that store path/digest/structure but not raw source;
- a query-aware retrieval planner that narrows exact-file work and deepens debug/test graph recall;
- generation-safe compiled context caching with bounded TTL/entry controls;
- non-destructive sidecar consolidation through canonical aliases while durable provider memory remains untouched;
- richer retrieval diagnostics, AST structural context surfaces, and cache/consolidation statistics;
- a compiled-context cache benchmark in addition to the synthetic candidate-collapse benchmark.

Validation performed in the build sandbox:

- `tests/test_memory.py`: 15 passed.
- broader non-UI/non-provider selection: 61 passed, with 3 failures reproduced unchanged on the v2 baseline.
- full-suite collection remains blocked in this sandbox by missing optional/runtime dependencies (`openai`, `textual`).

Smoke benchmarks in this sandbox (not production guarantees):

- Rhomgaic synthetic recall, 1,000 memories / 100 queries / 96 dims: exact-target top-1 1.000, mean search about 11.6 ms.
- Compiled working-context cache, 300 memories / 50 hot repeats: cold about 39.7 ms, hot mean about 0.5 ms, approximately 78.7x for that repeated-query workload.
