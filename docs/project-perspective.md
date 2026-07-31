# Project Perspective System

Humoid now has a task-scoped repository perspective layer for broad project review, repair, redesign, extension, and documentation work.

## Why it exists

Without a repository index, an agent can spend most of its tool budget repeatedly calling `list_files` and `read_file`. That creates long conversations, duplicates source text in model context, and leaves too few rounds for implementation and validation.

The perspective layer changes the default workflow:

1. Build a temporary index once.
2. Search the index with targeted architectural or implementation questions.
3. Expand only the exact chunks needed for the next decision.
4. Read exact files immediately before editing.
5. Make code, tests, configuration, and documentation changes as one coherent change set.
6. Validate and clear the temporary index when the task is complete.

## Tools

### `build_project_perspective`

Indexes text files from the workspace using 5,600-character chunks with 600-character overlap by default. It records paths, line ranges, symbols, entry points, tests, and documentation surfaces.

The index is deliberately separate from durable conversation memory. Each instance receives a unique temporary collection name.

### `search_project_perspective`

Searches by feature, architecture, symbol, failure mode, test, documentation requirement, or change objective. Results combine deterministic hashing-vector similarity with lexical and path-aware scoring. A diversity pass prevents one large file from consuming the entire packet.

The returned packet has a strict character budget. Lower-ranked results are folded into a context accordion that lists file names, chunk IDs, and symbols without injecting all source text into the conversation.

### `expand_project_perspective`

Expands selected chunk IDs and a bounded number of neighboring chunks. This is intended for concrete evidence gaps, not another whole-repository read.

### `project_perspective_status`

Reports the active backend, collection name, objective, repository manifest, chunk count, and files changed since indexing.

### `clear_project_perspective`

Deletes the temporary Weaviate collection and clears the local task cache.

## Backends

`HUMOID_PERSPECTIVE_BACKEND=auto` uses this order:

1. A configured healthy remote Weaviate instance.
2. A healthy existing embedded Weaviate instance.
3. A disposable embedded Weaviate instance on separate perspective ports.
4. A private dependency-free local vector fallback.

The fallback uses a deterministic hashing embedder over identifiers, subtokens, and bigrams. It requires no model download and does not send source code to another service.

## Synchronization after edits

`write_file` and `/undo` automatically update the local perspective delta. Search excludes stale Weaviate chunks for changed files and uses the fresh local vectors instead. A full rebuild is only needed when files are modified outside Humoid or when the indexed scope changes substantially.

## Recommended review sequence

A model should build a multi-perspective implementation map before editing:

- User intent and expected behavior
- Architecture and module boundaries
- Data and control flow
- Failure modes and error handling
- Compatibility and migration constraints
- Security and trust boundaries
- Tests and observability
- Documentation and operator experience

Conflicts between perspectives should be resolved from retrieved repository evidence. Exact file reads remain appropriate before edits, exact quotations, or syntax-sensitive decisions.

## Configuration

```env
HUMOID_PERSPECTIVE_ENABLED=true
HUMOID_PERSPECTIVE_BACKEND=auto
HUMOID_PERSPECTIVE_CHUNK_CHARS=5600
HUMOID_PERSPECTIVE_CHUNK_OVERLAP=600
HUMOID_PERSPECTIVE_MAX_FILES=4000
HUMOID_PERSPECTIVE_MAX_FILE_BYTES=1000000
HUMOID_PERSPECTIVE_SEARCH_LIMIT=12
HUMOID_PERSPECTIVE_PACKET_MAX_CHARS=14000
HUMOID_PERSPECTIVE_VECTOR_DIMS=384
HUMOID_PERSPECTIVE_HTTP_PORT=8089
HUMOID_PERSPECTIVE_GRPC_PORT=50059
```

Use `HUMOID_PERSPECTIVE_BACKEND=local` to force the no-service fallback.
