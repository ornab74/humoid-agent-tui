# Constraint-Graph Code Search

Humoid's project perspective no longer treats repository search as “embed every chunk and hope.” The search path now compiles a task into code-specific constraints and returns the smallest useful connected subgraph.

## Query pipeline

1. **Compile constraints** from the request: identifiers, file paths, subsystem roles, test intent, and debugging intent.
2. **Retrieve cheaply** from inverted identifier, symbol-definition, symbol-reference, role, and trigram indexes. Ordinary queries do not scan every vector.
3. **Score candidates** with exact constraint coverage first. Lexical overlap and lightweight local hash vectors are tie-breakers.
4. **Walk typed edges** between chunks. Edges represent symbol references, imports, source/test relationships, and adjacent chunks.
5. **Return a minimal subgraph** by adding only high-value candidates and the bridge chunks needed to connect them.
6. **Fold alternatives** into the Context Accordion so a small model receives a compact change packet and can expand a specific node later.

## Failure memory

For repair, regression, crash, exception, and debugging queries, the perspective reuses Humoid's existing `MemoryRouter`.

When Weaviate is the active memory backend, historical conversations and verified memories are queried from the normal durable memory collection. File paths and symbols mentioned in prior failures become bounded ranking boosts. The search packet explains every boost under `FAILURE MEMORY`; it never silently overrides current code evidence.

The same interface works with the SQLite fallback. General architecture searches skip failure-memory retrieval entirely, avoiding an unnecessary database query.

## Cost controls

The graph is built with language-neutral regex extraction and standard-library data structures. There is no graph database and no compiler pass. Normal queries touch postings lists and a bounded graph neighborhood; a full local vector ranking is reserved for cold-start queries with no identifier, role, path, or failure-memory match.

```env
HUMOID_PERSPECTIVE_GRAPH_MAX_HOPS=3
HUMOID_PERSPECTIVE_GRAPH_FANOUT=20
HUMOID_PERSPECTIVE_FAILURE_MEMORY=true
HUMOID_PERSPECTIVE_FAILURE_MEMORY_LIMIT=16
HUMOID_PERSPECTIVE_VECTOR_DIMS=192
```

## Packet shape

A search result contains:

- the task objective and repository manifest;
- typed behavior edges;
- historical failure evidence when applicable;
- anchor and bridge nodes with definitions, references, roles, scores, and reasons;
- folded alternatives identified by stable chunk IDs.

This format is designed for small local models: the model receives the code that explains the behavior, not a giant list of semantically similar files.
