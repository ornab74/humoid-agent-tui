# Evidence-Gated Edit Cycles

Humoid now treats a repository edit as a sequence of compiled context states rather than one giant conversational prompt. The harness owns localization, scope control, evidence, and phase transitions; the model receives only the current task contract.

## Automatic entry

Normal TUI startup installs the edit-cycle runtime before `AgentHarness` is imported. When a user prompt contains an edit operation such as fix, implement, refactor, rename, remove, or update, the runtime automatically:

1. compiles the original request into deterministic constraints;
2. builds or refreshes the repository perspective;
3. executes a query lattice over behavior, symbols, tests, risk, and failure history;
4. merges duplicate graph nodes into a bounded evidence packet;
5. injects a unique `[EDIT CYCLE PROMPT v2]` before the first model call.

The model does not need to remember to initialize the process. The explicit `start_edit_cycle` tool remains available for integrations that instantiate `ToolRegistry` outside the standard TUI entry point.

## Intent compiler

`EditIntentCompiler` extracts:

- operation class: debug or edit;
- explicit files and code-like symbols;
- hard constraints such as preserve, only, must, and without;
- acceptance language such as test, verify, ensure, and expect;
- risk surfaces including authentication, concurrency, storage, network, and UI;
- a stable task ID derived from the request.

It emits several retrieval queries rather than one semantic query:

- the original behavior request;
- definitions, references, callers, imports, control flow, and data flow;
- tests, assertions, fixtures, acceptance, and validation;
- compatibility, invariants, and risk surfaces;
- regression, traceback, and historical bug zones for debugging work.

## Phase-specific prompt cycle

Every significant tool result produces a new prompt with a unique cycle ID, current context digest, evidence ledger, and one phase contract.

### Localize

Do not edit. Select the minimum connected behavior subgraph and read the exact primary target.

### Inspect

Read the implementation plus its nearest test or configuration dependency. Identify invariants and patch boundaries.

### Edit

Make one coherent change only in grounded, exact-read files. Prefer digest-checked `apply_patch` for large existing files.

### Verify

Run `run_validation`, which preserves the subprocess exit code. Start with the narrowest meaningful validator.

### Repair

A failed validator stores failure memory and triggers differential graph retrieval using the original request, changed files, command, and error output. The next prompt contains the revised evidence snapshot.

### Conclude

Stop editing. Close the cycle and report changed files, observed validation, and unresolved risks.

## Hard edit gates

`write_file` and `apply_patch` are policy-gated, not merely discouraged in a prompt.

An existing target must:

1. appear in retrieved repository evidence;
2. be read exactly with `read_file` or `read_file_range` during the active cycle;
3. be edited during an allowed phase.

A new file must belong to a grounded parent component. Identical repeated calls are blocked after a configurable threshold.

## Surgical primitives

`read_file_range` returns numbered lines plus the SHA-256 digest of the complete file. `apply_patch` requires that digest and exact replacement counts. A concurrent or out-of-cycle modification therefore fails as a stale patch instead of overwriting newer work.

This allows a small model to change a few lines in a large file without regenerating thousands of untouched lines.

## Validation and memory

`run_validation` executes in the workspace, enforces the configured timeout, and reports non-zero exit codes as errors. A failure becomes structured `code_failure` memory through the existing `MemoryRouter`; a passing validator becomes verified `code_validation` memory. Weaviate is used when configured, with SQLite as the durable fallback.

Memory metadata includes task ID, changed files, validation command, and context digest. Future debugging retrieval can bias toward historical bug zones while still requiring current code evidence.

## Configuration

```env
HUMOID_EDIT_CYCLE_MAX_QUERIES=5
HUMOID_EDIT_CYCLE_SEARCH_LIMIT=10
HUMOID_EDIT_CYCLE_CONTEXT_MAX_CHARS=18000
HUMOID_EDIT_CYCLE_PROMPT_MAX_CHARS=24000
HUMOID_EDIT_CYCLE_REPEAT_LIMIT=3
```

For small local models, reduce the packet rather than disabling the process:

```env
HUMOID_EDIT_CYCLE_MAX_QUERIES=3
HUMOID_EDIT_CYCLE_SEARCH_LIMIT=6
HUMOID_EDIT_CYCLE_CONTEXT_MAX_CHARS=8000
HUMOID_EDIT_CYCLE_PROMPT_MAX_CHARS=12000
```

## Advanced extensions

The cycle boundary can accept stronger evidence providers without changing model behavior:

- language-server and compiler symbol edges;
- runtime traces and failing-stack anchors;
- Git co-change, ownership, and review-history edges;
- mutation-test coverage as a write authorization signal;
- blast-radius and patch-risk estimation;
- counterfactual retrieval: code that must be absent for the behavior not to occur;
- mined invariants from tests and repeated production failures;
- candidate localization from several tiny models followed by deterministic graph intersection;
- snapshot replay for measuring localization precision, context cost, patch correctness, and repair rounds.
