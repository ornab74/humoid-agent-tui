# Chromatic Systems v6 — Multi-Resolution Code Topology

Phase 6 advances the chromatic layer itself. Rhomgaic still treats Weaviate/SQLite
provider memory as durable authority. Chromatic surfaces are deterministic,
rebuildable acceleration hints that reduce candidate work before exact reranking.

## Address pyramid

A coding query is represented at four simultaneous resolutions:

- **L0 project** — action + language. Useful when the query has no concrete file.
- **L1 package** — project surface plus directory/module hierarchy.
- **L2 file** — concrete file-local coding surface.
- **L3 symbol + behavior** — file + symbols + structural/maintenance behavior.

Each resolution is an independent deterministic 64-bit key. The existing compact
16-byte chromatic address remains available for multiprobe distance ranking.
The hierarchy keys are stored in `rhomgaic_terms`, so no additional authoritative
state is introduced.

A typical query can therefore collapse like:

```text
all memory
  -> Python/debug surface
  -> src/payments package
  -> src/payments/pricing.py
  -> calculate_total + regression/call behavior
  -> dense semantic + exact lexical reranking
```

A hierarchy match is never a hard assertion that a memory is correct. If the
high-resolution buckets miss, bounded lower-resolution and recent fallbacks remain
available.

## Topology and behavior channels

The hierarchy indexes:

- directory prefixes and dotted module scopes,
- file identity,
- symbols,
- AST definitions/classes/calls/imports/raises/decorators,
- maintenance behavior such as API, dependency, control-flow, regression,
  validation, and refactor intent.

Relationship edges now include bounded `chromatic-topology` and `behavior` links,
which supplements the existing exact AST dependency graph.

## Chromatic change delta

Semantic code diffs now receive a second 16-byte address describing the *shape of
the change*, without storing source code. Channels include:

1. file,
2. language,
3. semantic risk bucket,
4. public API change,
5. post-edit parse state,
6. signature-change count,
7. definition delta,
8. import delta,
9. call delta,
10. raised-exception delta,
11. branch delta,
12. broad-exception delta,
13. line delta,
14–16. compact fingerprint projections.

Procedural coding memories preserve these change addresses as compact metadata.
This allows future maintenance work to distinguish, for example, a low-risk
internal refactor from a public-signature migration even when both are described
with similar prose.

## Agent inspection

`inspect_chromatic_code_surface` exposes the current query address, resolution
keys, behaviors, path scopes, candidate-collapse counts, and the final reranked
memory candidates. It is read-only and diagnostic; the agent must not treat the
reported buckets as proof.

`recall_chromatic_change_history` works after semantic edit fingerprints exist.
It compares the current 16-byte change shape against prior validated procedural
change shapes and retrieves the closest validation/repair strategies. The
weighted distance emphasizes API, parse, risk, signature, dependency, exception,
and control-flow channels while allowing the strategy to transfer across files.
It never returns an old patch as an instruction.

Progressive RAG traces additionally expose the chosen chromatic resolution and
candidate-collapse telemetry.

## Benchmark

`benchmarks/chromatic_v6_benchmark.py` measures deterministic exact-target recall,
search latency, hierarchy-posting count, graph size, and the fraction of the
memory universe not passed to final reranking.

Example sandbox smoke run used during development:

```text
1,000 memories / 100 queries
exact-target top-1:      1.000
mean search:             ~5.45 ms
p50:                     ~5.32 ms
p95:                     ~6.34 ms
mean candidate avoided:  ~92%
chromatic hierarchy postings: 10,000
```

These are synthetic measurements on the development sandbox, not production
latency or scale guarantees.
