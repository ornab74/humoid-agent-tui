# Constraint-Graph Code Search

Humoid's project perspective no longer treats repository search as “embed every chunk and hope.” The search path compiles a task into code-specific constraints and returns the smallest useful connected subgraph.

## Query pipeline

1. **Compile constraints** from the request: identifiers, file paths, subsystem roles, test intent, and debugging intent.
2. **Retrieve cheaply** from inverted identifier, symbol-definition, symbol-reference, role, and trigram indexes. Ordinary queries do not scan every vector.
3. **Score candidates** with exact constraint coverage first. Lexical overlap and lightweight local hash vectors are tie-breakers.
4. **Walk typed edges** between chunks. Edges represent symbol references, imports, source/test relationships, and adjacent chunks.
5. **Return a minimal subgraph** by adding only high-value candidates and the bridge chunks needed to connect them.
6. **Fold alternatives** into the Context Accordion so a small model receives a compact change packet and can expand a specific node later.

## Failure memory

For repair, regression, crash, exception, and debugging queries, the perspective reuses Humoid's existing `MemoryRouter`.

When Weaviate is active, historical failures and verified fixes are queried from the durable memory collection. File paths and symbols mentioned in prior failures become bounded ranking boosts. The search packet explains every boost under `FAILURE MEMORY`; history never silently overrides current code evidence.

The same interface works with the SQLite fallback. General architecture searches skip failure-memory retrieval entirely.

## Cost controls

The graph is built with language-neutral extraction and standard-library data structures. There is no graph database and no mandatory compiler pass. Normal queries touch postings lists and a bounded graph neighborhood; full local-vector ranking is reserved for cold-start queries with no identifier, role, path, or memory match.

```env
HUMOID_PERSPECTIVE_GRAPH_MAX_HOPS=3
HUMOID_PERSPECTIVE_GRAPH_FANOUT=20
HUMOID_PERSPECTIVE_FAILURE_MEMORY=true
HUMOID_PERSPECTIVE_FAILURE_MEMORY_LIMIT=16
HUMOID_PERSPECTIVE_VECTOR_DIMS=192
```

## From search to edits

Search packets are now consumed by the evidence-gated edit-cycle runtime documented in `docs/edit-cycle.md`. An edit request is expanded into a query lattice, merged into one context snapshot, and converted into a unique phase prompt for each model round.

The search engine therefore does not merely return candidates. Its graph nodes become write authorization evidence:

- existing files must be present in retrieved evidence;
- the exact target must be read during the active cycle;
- digest-checked patches reject stale source;
- failed validators generate a new failure-focused retrieval query;
- verified fixes and observed failures return to Weaviate/SQLite memory.

This makes repository localization a deterministic infrastructure service rather than a repeated `grep`/`sed` behavior performed inside the language model's context window.

## Further evidence layers

The graph API is intentionally open to more advanced, incrementally computed edges:

- compiler and language-server definitions, references, types, and overrides;
- runtime traces, stack frames, and branch coverage;
- Git co-change, blame, ownership, and review discussion;
- dependency-injection bindings, generated-code provenance, and configuration flow;
- invariant and precondition mining from tests;
- mutation-test and impact-analysis edges;
- patch-risk scores and counterfactual behavior paths.

Those sources can improve the same minimal-subgraph selector without increasing the prompt protocol or requiring a larger editing model.
