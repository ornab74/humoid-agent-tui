# Rhomgaic / Chromatic Memory v5 — Coding & Maintenance Intelligence

Phase 5 focuses the memory architecture on software development and long-lived code maintenance. The central design change is to keep **current code facts**, **historical facts**, and **historical procedures** on separate surfaces.

A prior repair can be useful without its old patch being correct today. Therefore procedural recall is explicitly non-authoritative: the current repository AST, dependency frontier, exact file reads, and observed validators always win.

## 1. Development-memory loop

```text
current request
    |
    v
current AST / impact graph -----------+
    |                                 |
    v                                 |
factual Weaviate + Rhomgaic RAG       |
    |                                 |
    +------> procedural playbooks <----+
    |              |
    v              v
maintenance execution contract
    |
exact read -> minimal patch
    |
semantic change fingerprint
    |
focused validator -> broader validator when risk requires it
    |
edit-cycle trajectory
    |
provider procedural memory
    |
rebuildable Rhomgaic procedure index
```

## 2. Procedural coding memory

A successful edit cycle writes a compact provider-backed `procedural` memory. Its metadata contains a bounded `coding_trajectory` with:

- original request and operation;
- intended/current files and symbols;
- constraints, acceptance criteria, and risk categories;
- actual changed files;
- edit-cycle step kinds rather than hidden reasoning;
- observed validator commands;
- success/failure outcome;
- compact semantic change fingerprints.

The sidecar aggregates trajectories with the same **procedure shape** rather than memorizing an exact patch. The signature uses action, language, risk surface, step kinds, and validator family. That permits multiple cache-regression repairs, for example, to reinforce one reusable procedure while keeping their file-specific factual memories separate.

Reliability is Laplace-smoothed:

```text
confidence = (successful_support + 1) / (support_count + 2)
```

This prevents one lucky repair from becoming a supposedly perfect procedure.

Procedure ranking combines deterministic chromatic similarity with lexical/code surfaces and observed reliability:

```text
S_proc =
    0.28 semantic
  + 0.14 chromatic
  + 0.16 lexical
  + 0.09 action
  + 0.04 language
  + 0.08 exact-file
  + 0.05 symbol
  + 0.11 confidence
  + 0.05 support-strength
```

The exact coefficients are engineering defaults, not learned universal constants.

## 3. Semantic change fingerprints

Every successful file mutation can emit a compact structural diff. Python fingerprints record:

- added, removed, and structurally modified definitions;
- function/class signature changes;
- import-edge changes;
- call-edge changes;
- changed raised-exception surfaces;
- branch-count delta;
- broad-exception-handler delta;
- public API change flag;
- bounded heuristic risk score.

Raw source is not stored in the fingerprint. The fingerprint is designed to answer a maintenance question quickly: **what kind of behavior surface did this edit actually disturb?**

A Python parse failure after an edit moves the active edit cycle into `repair` rather than allowing the agent to behave as though the patch is ready for normal verification.

High semantic risk or a public-surface change also strengthens the edit-cycle verification contract: start narrow, then include linked tests or the smallest broader suite before concluding.

## 4. Maintenance pressure report

`maintenance_health_report` scans current code and surfaces deterministic inspection priorities such as:

- parse failures;
- very large symbols;
- dependency fanout;
- high change centrality / fan-in;
- broad exception handlers;
- TODO/FIXME/HACK markers;
- files for which the static graph found no direct test edge.

This is intentionally called **maintenance pressure**, not code quality. A central function can be excellent code and still deserve extra inspection because many behaviors depend on it.

## 5. Maintenance execution planner

`plan_maintenance_change` combines:

1. current AST-backed impact anchors and change frontier;
2. linked test surfaces;
3. risk and uncertainty from current code;
4. historical procedural playbooks;
5. validator candidates discovered from files actually present in the repository.

The resulting contract makes the precedence explicit:

```text
current source evidence
    > observed current validators
    > current factual history
    > historical procedural strategy
```

The tool does not execute a remembered plan. It helps the agent avoid repeatedly rediscovering useful repair structure while still forcing it to localize and verify against the current repository.

## 6. New coding/maintenance tools

- `plan_maintenance_change(query)`
- `maintenance_health_report(paths?)`
- `recall_coding_playbook(query)`

Existing `analyze_change_impact`, Project Perspective, edit-cycle gates, and Rhomgaic factual recall remain separate and complementary.

## 7. Safety / correctness invariants

1. Weaviate/SQLite remains durable authority.
2. Rhomgaic procedure tables are rebuildable from provider procedural metadata.
3. Historical procedures never authorize a write.
4. Existing files still require grounded evidence and exact reads before edits.
5. Semantic fingerprint failure never invalidates a successful filesystem write; it only degrades advisory intelligence.
6. A post-edit Python parse failure is surfaced as a repair condition.
7. Validator evidence is recorded as observed commands/outcomes, not inferred success.
8. Hidden chain-of-thought is never stored as a procedure; only tool/event trajectory metadata is retained.

## 8. Benchmarking

`benchmarks/rhomgaic_phase5_maintenance_benchmark.py` measures two local mechanisms:

- procedural-playbook ingestion/retrieval;
- semantic-fingerprint construction.

The output is a synthetic smoke benchmark only. Re-run it on the target workstation and on real historical repair tasks before using its timings or recall rate as a production claim.
