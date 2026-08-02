# Incremental Code Intelligence

This layer extends constraint-graph retrieval with a compiler-style change-frontier pass.

## What it adds

- persistent per-file SHA-256 fingerprints under `.humoid/code-intelligence-v2.json`;
- incremental reparsing of changed files only;
- Python AST extraction for classes, functions, methods, signatures, decorators, calls, imports, and inheritance;
- conservative generic-language extraction for symbols, calls, imports, and tests;
- typed `calls`, `inherits`, `contains`, `imports`, and `tests` edges;
- bounded forward and reverse propagation from likely edit anchors;
- blast-radius files, linked tests, risk score, uncertainty, and evidence explanations;
- localization benchmarks with precision, recall, and mean reciprocal rank.

## Runtime composition

The standard entry point installs the layers in this order:

1. `install_edit_cycle_runtime()` establishes evidence-gated phases and surgical editing.
2. `install_code_intelligence_runtime()` enriches every cycle with static impact analysis.
3. `install_action_guard()` remains the outer recovery layer for malformed or planning-only model turns.

This order repairs the post-merge integration regression where the edit-cycle files existed on `main` but only the action guard was activated.

## Edit-cycle integration

Before the first model inference, the edit-cycle context now includes both:

- the minimal constraint-graph behavior packet; and
- a static impact frontier covering callers, subclasses, imports, tests, risk, and uncertainty.

After a validation failure, the impact frontier is rebuilt from changed paths and the failure-focused query before the repair prompt is generated.

## Tools

- `build_code_intelligence`: incrementally refresh the persistent graph.
- `analyze_change_impact`: return anchors, frontier symbols, typed edges, tests, and risk.
- `benchmark_code_localization`: evaluate expected-file precision, recall, and reciprocal rank.

## Deliberate limits

Python uses the standard-library AST and is the highest-confidence adapter. Other languages currently use conservative extraction; future adapters can add tree-sitter, compiler APIs, or LSP call hierarchy without changing the public impact-report contract.
