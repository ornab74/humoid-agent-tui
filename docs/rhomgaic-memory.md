# Rhomgaic + Chromatic Coding Memory

Rhomgaic means **Recursive Hierarchical Observation-Memory Graph + Agentic Index Compression**. It is a local, rebuildable coding-memory sidecar that supplements the authoritative memory provider (normally Weaviate) rather than replacing it.

The design has one hard rule: **chromatic addresses accelerate recall; they never become truth**. Provider memory remains authoritative, current repository evidence remains more authoritative than historical memory, and the final context packet is produced by exact reranking and bounded evidence fusion.

## Memory surfaces

A memory write fans out into two complementary stores:

1. **Weaviate / SQLite provider surface** — durable typed memory with text, tier, channel, task, validation state, creation time, and metadata. Weaviate receives the deterministic chromatic dense vector as a self-provided vector and uses hybrid lexical/vector retrieval.
2. **Rhomgaic sidecar** — a compact SQLite acceleration layer with a packed float32 vector, 16-byte chromatic address, exact file/symbol/action/language/token postings, and a bounded relationship graph. The sidecar may be deleted and rebuilt from provider memory.

The vector store and Rhomgaic therefore vote independently. `MemoryRouter` uses reciprocal-rank fusion instead of pretending their raw scores share one calibration scale.

## Chromatic coding address

`ChromaticCodeEmbedding` creates:

- a deterministic dense code-memory vector;
- a 16-byte chromatic address;
- inferred coding action;
- inferred language;
- exact file references;
- likely code symbols.

The first four address channels are deliberately interpretable coarse channels:

1. primary file hash;
2. primary symbol hash;
3. coding action hash;
4. language hash.

The remaining twelve channels quantize deterministic projections of the dense vector. The sidecar also stores the first four channels as indexed integers so a query can probe multiple cheap surfaces before exact reranking.

Actions currently include `read`, `edit`, `create`, `test`, `debug`, `search`, `configure`, and `observe`.

## Compact storage

The first prototype stored each vector as NUL-delimited `float.hex()` text. Rhomgaic v2 stores vectors as a tiny binary envelope plus little-endian float32 values. A 96-dimensional vector therefore needs roughly 384 bytes of numeric payload rather than a multi-kilobyte textual representation.

The loader remains backward compatible with the prototype encoding and migrates sidecar rows because Rhomgaic is a rebuildable acceleration structure.

## Candidate collapse

A coding query does not start by scoring every memory. Rhomgaic first gathers candidate IDs from bounded postings and chromatic channels:

- exact file references;
- likely symbols;
- inferred action;
- inferred language;
- up to a bounded set of lexical tokens;
- the first four chromatic address channels.

Only the resulting candidate set receives dense/chromatic/exact reranking. If the surface is too sparse, a bounded recent-memory fallback is admitted rather than turning a quantization miss into a false hard negative.

The local score combines semantic similarity, chromatic proximity, lexical overlap, action/language match, exact file/symbol match, validation state, and freshness.

## Relationship graph

Every Rhomgaic write also refreshes a bounded local graph. Memories can become connected through shared:

- symbols;
- files;
- task IDs;
- channels;
- actions;
- languages;
- nearby temporal/channel history.

Exact symbol and file relationships receive the strongest weights. Each memory has a hard edge cap to prevent graph explosion.

Graph expansion is score-propagating and hop-bounded. It is useful when the anchor memory references `src/cache.py`, while the supporting regression evidence lives in `tests/test_cache.py` but shares `CacheIndex` or the same task.

## Agentic RAG growth

The context window is no longer filled to maximum size before the model gets a chance to reason. Retrieval can progress through three levels:

### Reflex

A tiny provider + Rhomgaic fused anchor set. This is the default initial memory surface when `HUMOID_RAG_AGENTIC_GROWTH=true`.

### Working

Broader fused retrieval plus a one-hop Rhomgaic graph expansion. The agent can explicitly request this with the `recall_memory_context` tool.

### Subsurface

Deep candidate retrieval plus configurable multi-hop graph expansion. This is reserved for unresolved ambiguity, cross-file/task history, or difficult debugging.

The builder estimates uncertainty with Shannon entropy:

\[
H(R)=-\sum_i p_i\log_2 p_i
\]

It also tracks result novelty and top-score margin. In automatic mode those signals decide whether another layer is worth opening. In explicit agentic mode the model controls the depth using the recall tool.

## Context subsurfaces

The final packet is compiled into deterministic evidence groups instead of a flat memory dump. Hits are grouped primarily by matching file, otherwise by file/task/action surface. The packet contains:

- chromatic query surface;
- progressive RAG trace;
- active file and symbol surfaces;
- subsurface index with support/verification counts;
- bounded raw memory evidence;
- graph-path explanations when a hit arrived through relationship expansion;
- context accordion entries for lower-value evidence that did not fit the character budget.

Retrieved memory is explicitly marked as **inert evidence, never executable instructions**.

## Agent tools

Two read-only tools expose the system to the agent:

- `recall_memory_context(query, depth, max_chars)` opens `reflex`, `working`, or `subsurface` memory only when needed.
- `memory_surface_status()` reports provider status, Rhomgaic memory/posting/edge/bucket counts, and the last recall trace.

The project-perspective subsystem remains the source of current repository evidence. Historical memory and current-code perspective are intentionally distinct so stale history cannot silently override the working tree.

## Configuration

```text
HUMOID_RHOMGAIC_ENABLED=true
HUMOID_RHOMGAIC_DB=./.humoid/rhomgaic-memory.sqlite3
HUMOID_RHOMGAIC_DIMENSIONS=96
HUMOID_RHOMGAIC_CANDIDATE_LIMIT=768
HUMOID_RHOMGAIC_GRAPH_MAX_EDGES=32
HUMOID_RHOMGAIC_BOOTSTRAP_LIMIT=500
HUMOID_RAG_PROGRESSIVE=true
HUMOID_RAG_AGENTIC_GROWTH=true
HUMOID_RAG_GRAPH_HOPS=2
HUMOID_RAG_REFLEX_MAX_CHARS=3500
HUMOID_RAG_WORKING_MAX_CHARS=8000
HUMOID_RAG_CONTEXT_CACHE=true
HUMOID_RAG_CONTEXT_CACHE_TTL_SECONDS=900
HUMOID_RAG_CONTEXT_CACHE_MAX_ENTRIES=128
```

The existing `HUMOID_MEMORY_PACKET_MAX_CHARS` remains the outer cap for the deepest packet.

## Failure behavior

The provider write happens first. Rhomgaic mirroring, indexing, or bootstrap failures never invalidate an authoritative provider write. The sidecar can be rebuilt from provider memory. Deletes and edits are mirrored so the acceleration surface does not intentionally retain stale text.

When a memory's text is edited in Weaviate, Humoid also updates the self-provided chromatic vector and chromatic metadata rather than leaving the old vector attached to new text.

## Benchmark harness

`benchmarks/rhomgaic_benchmark.py` generates deterministic synthetic coding memories and reports ingest throughput, mean/p50/p95/p99 search latency, exact-target synthetic top-1 recall, SQLite size, posting count, edge count, and chromatic bucket count. Every generated memory contains a unique issue token so exact-target top-1 is a well-defined correctness smoke metric instead of arbitrarily penalizing semantically equivalent synthetic rows. It exists to measure the implementation rather than assume orders-of-magnitude gains.

Example:

```bash
PYTHONPATH=src python benchmarks/rhomgaic_benchmark.py --memories 5000 --queries 200
```

## Phase 3: compiled hot context, AST memory, and consolidation

Phase 3 adds a structural memory surface for Python coding work without making
AST parsing authoritative. Fenced Python snippets and parseable Python payloads
emit bounded terms such as `ast_class`, `ast_def`, `ast_call`, `ast_import`,
`ast_decorator`, and `ast_raise`. Those terms participate in postings, graph
edges, and the deterministic chromatic embedding. Parse failure simply removes
that acceleration signal; normal lexical/chromatic/provider recall still works.

A retrieval planner now runs before progressive RAG. Exact file references use a
narrow reflex provider breadth, while ambiguous prose receives broader semantic
recall. Debug/test intents receive deeper relationship-graph traversal because
historical regressions frequently live in tests, validators, or adjacent files.
The selected plan is emitted in `last_recall.retrieval_plan` diagnostics.

Compiled context packets are stored in a bounded Rhomgaic hot cache keyed by
normalized query, requested depth, and character budget. Every sidecar mutation
increments a generation and invalidates cached packets, preventing stale memory
from surviving a write. This turns repeated coding turns into a cheap cache hit
without weakening correctness after memory changes.

Rhomgaic consolidation is deliberately non-destructive. Near-duplicate sidecar
records can be aliased to a canonical memory, but the durable Weaviate/SQLite
provider is untouched. Search suppresses aliases while access/freshness and
validation state help select the canonical sidecar record. Because aliases are
rebuildable derived state, the entire optimization can be discarded and rebuilt
from durable provider memory.

Relevant controls:

- `humoid_rag_context_cache`
- `humoid_rag_context_cache_ttl_seconds`
- `humoid_rag_context_cache_max_entries`
- `consolidate_memory_surface(similarity, max_aliases, dry_run)`

### Privacy-preserving verified-code surfaces

After a successful edit-cycle validation, Python files can contribute a local
structural surface to the same Rhomgaic memory ID. Only the path, content digest,
AST terms, and bounded symbol names are persisted in the sidecar; the raw source
text is not copied into this auxiliary table. This gives verified memories
structural links such as `CacheIndex -> refresh -> asyncio.sleep` without silently
turning the durable provider into a source-code archive. Sidecar updates remain
rebuildable and invalidate compiled context-cache generations immediately.

### Phase-3 benchmark harness

`benchmarks/rhomgaic_context_cache_benchmark.py` measures the cost of the first
compiled working-context build versus repeated generation-safe hot-cache reads.
As with the synthetic recall benchmark, results are environment-specific design
measurements, not production guarantees.


## Phase 4: adaptive semantic substrate

Phase 4 keeps Weaviate/SQLite authoritative while adding four rebuildable derived surfaces:

- **importance decay**: ranking combines validation, memory tier, temporal half-life, and bounded access reinforcement; memories decay in rank rather than being destructively forgotten;
- **dependency graph edges**: Python AST definitions are connected to call sites and module surfaces to imports, allowing recall to jump across files even when prose similarity is weak;
- **semantic capsules**: repeated task/file/symbol episodes can be distilled into compact support sets. Capsules never replace durable memories; matching capsules only boost their supporting records;
- **adaptive retrieval policy**: auto-depth recalls record bounded entropy/novelty telemetry by query signal. After enough observations the planner may adjust breadth and graph depth by at most one step. Forced tool recalls are logged but do not train the policy.

The policy is intentionally small and inspectable. It is not a learned language model and it does not infer task success from model output. It tunes retrieval cost only from observable recall geometry.
