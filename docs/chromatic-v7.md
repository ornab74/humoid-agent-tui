# Chromatic Systems v7 — Invariant Compiler and Proof Surfaces

Phase 7 extends chromatic coding memory from *where should I search?* and *what shape did this edit have?* into a third question:

> **What observable properties is this change supposed to preserve, and what evidence says they still hold?**

This is intentionally a lightweight engineering proof system, not formal verification. A chromatic invariant address is an acceleration/grouping hint. Repository state, AST evidence, semantic fingerprints, and observed validators remain authoritative.

## Invariant contract compilation

At the beginning of an edit cycle, Humoid can compile a bounded contract from:

- the original user request,
- explicit files and symbols,
- preservation constraints (`preserve`, `without breaking`, `must remain`, ...),
- acceptance requirements,
- current AST/public signatures,
- current imports and raised exceptions,
- the code-intelligence impact frontier and linked tests.

Current invariant kinds include:

- `parseability` — touched Python surfaces must remain parseable,
- `public-signature` — public signatures are preserved unless the request explicitly authorizes API/signature change,
- `symbol-presence` — requested symbols remain present unless the request asks to remove/rename them,
- `dependency-continuity` — import drift is surfaced for review,
- `exception-continuity` — exception-contract drift is surfaced for review,
- `blast-radius` — edits outside the grounded impact frontier are flagged,
- `explicit-constraint` — preservation constraints remain proof obligations,
- `validation-obligation` — acceptance/debug work can require observed validation before conclusion.

The compiler stores no source bodies in the contract. Baselines contain compact structural facts such as public signatures, symbol names, imports, raises, parse state, and content digests.

## 16-byte invariant address

Every obligation receives a deterministic 16-byte chromatic address. Channels encode the shape of the obligation rather than the code body:

1. invariant kind,
2. severity,
3. evidence source,
4. whether validation is required,
5. coding operation,
6. package scope,
7. file scope,
8. symbol scope,
9–13. statement-token projections,
14. operation/kind shape,
15. source/severity evidence class,
16. compact statement projection.

Distance is weighted so kind, severity, evidence class, and validation requirements matter more than file identity. That allows a successful public-API preservation strategy in one package to be recalled for a structurally similar obligation in another package without treating the old patch as applicable code.

## Lightweight proof evaluation

After edits, the contract is re-evaluated against:

- the current repository files,
- current AST signatures/symbols,
- semantic change fingerprints,
- changed-path scope,
- observed validator commands and outcomes.

Checks are classified as `pass`, `warn`, `fail`, or `unverified`.

A proof report includes:

- proof score,
- gate state,
- pass/warn/fail/unverified counts,
- concrete evidence for each invariant,
- remaining obligations,
- a deterministic proof key.

High-severity static violations (for example a forbidden public-signature change or a parse failure) block edit-cycle conclusion. Explicit critical validation obligations remain unverified until validation evidence exists.

A validator passing does **not** override a static invariant failure. The cycle stays in repair if the proof gate remains closed.

## Signature preservation is semantic, not name-only

Python public-signature baselines include:

- positional/keyword-only structure,
- default values,
- annotations,
- async status,
- return annotations.

For example, these are different invariant signatures:

```python
def fetch(key): ...

def fetch(key, *, stale_ok=False): ...

async def fetch(key, *, stale_ok=False) -> Result: ...
```

That catches API drift which line counts or symbol-name matching cannot see.

## Invariant-history procedural recall

Successful coding trajectories now persist the chromatic invariant addresses they satisfied alongside their change-shape deltas.

Rhomgaic can retrieve prior procedures by **proof-obligation similarity**:

```text
current invariant contract
        │
        ├── public-signature / critical
        ├── parseability / critical
        ├── dependency-continuity / medium
        └── validation-obligation / critical
        │
        ▼
chromatic invariant nearest procedures
        │
        ▼
prior validator strategy / inspection sequence
```

Historical invariant matches are validation-strategy hints only. They never prove the current change and never authorize replaying a prior patch.

## Edit-cycle integration

The edit cycle now carries:

- `invariant_contract`,
- `invariant_proof`,
- semantic change fingerprints,
- validator evidence.

The prompt exposes proof status, score, gate state, and open obligations. At finish time, a closed invariant gate blocks conclusion and returns the cycle to `repair`.

When the cycle closes successfully, its durable procedural trajectory includes both the contract and proof report. This lets the Rhomgaic sidecar rebuild invariant-history recall from provider memory.

## Agent tools

### `compile_chromatic_invariants`

Compile a read-only invariant contract for an arbitrary coding query before editing.

### `inspect_chromatic_invariant_proof`

Re-evaluate and display the active edit cycle's current proof surface.

### `recall_chromatic_invariant_history`

Retrieve prior validated procedures whose invariant-contract shape resembles the active contract.

These complement the v6 tools:

- `inspect_chromatic_code_surface`
- `recall_chromatic_change_history`

Together they separate three chromatic surfaces:

1. **location/topology** — where relevant code/history likely lives,
2. **change shape** — what structural mutation occurred,
3. **invariant shape** — what the change must preserve and how similar work was validated.

## Smoke benchmark

`benchmarks/chromatic_v7_invariant_benchmark.py` measures contract compilation, proof evaluation, and invariant-history recall.

Example sandbox run during development:

```text
200 synthetic modules / 60 contracts
contract compile mean:       ~34.2 ms
proof evaluate mean:          ~1.18 ms
invariant-history mean:       ~5.49 ms
invariant-history p95:        ~8.97 ms
expected invariant-kind hit:  1.000
mean invariants / contract:   41
```

These are synthetic sandbox measurements, not production latency guarantees.

## Design boundary

The invariant compiler deliberately does **not** claim semantic equivalence or mathematical correctness. Static checks cannot prove arbitrary business behavior. Its job is narrower and useful:

- make preservation assumptions explicit,
- catch structural contradictions early,
- turn acceptance requirements into visible proof obligations,
- prevent a green test from silently overriding a broken API contract,
- and retrieve historical validation strategies by obligation shape.
