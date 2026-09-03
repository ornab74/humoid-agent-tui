from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from pathlib import Path

from humoid_tui.models import MemoryHit
from humoid_tui.rhomgaic_memory import RagStage, RhomgaicMemory


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[index]


async def run(memories: int, queries: int, root: Path) -> None:
    store = RhomgaicMemory(
        root / "phase4.sqlite3",
        dimensions=96,
        semantic_capsules=True,
        policy_adaptation=True,
        importance_half_life_days=180,
    )
    await store.initialize()

    started = time.perf_counter()
    for index in range(memories):
        task = f"task-{index // 4}"
        await store.add(
            f"memory-{index}",
            f"Fixed src/module_{index % 24}.py Symbol_{index % 48} regression issue_{index:06d}",
            task_id=task,
            validation_status="verified" if index % 3 == 0 else "observed",
        )
    ingest_seconds = time.perf_counter() - started

    # One explicit definition/caller pair validates cross-file dependency traversal.
    await store.add("definition", "Implemented src/core.py compute_total primitive", validation_status="verified")
    await store.attach_code_surface("definition", "src/core.py", "def compute_total(items):\n    return sum(items)\n")
    await store.add("caller", "Updated src/service.py checkout path", validation_status="verified")
    await store.attach_code_surface(
        "caller",
        "src/service.py",
        "from src.core import compute_total\n\ndef checkout(items):\n    return compute_total(items)\n",
    )
    anchors = await store.search("src/core.py compute_total", limit=1)
    dependency_hits = await store.expand_graph(anchors, limit=8, hops=1)
    dependency_recalled = any(hit.memory_id == "caller" for hit in dependency_hits)

    started = time.perf_counter()
    capsules = await store.distill_semantic_capsules(min_support=3, max_capsules=256)
    distill_ms = (time.perf_counter() - started) * 1000.0

    latencies: list[float] = []
    exact = 0
    for index in range(queries):
        target = (index * 37) % memories
        query = f"fix src/module_{target % 24}.py Symbol_{target % 48} issue_{target:06d}"
        started = time.perf_counter()
        hits = await store.search(query, limit=8)
        latencies.append((time.perf_counter() - started) * 1000.0)
        if hits and hits[0].memory_id == f"memory-{target}":
            exact += 1

    probe = MemoryHit(
        memory_id="probe",
        text="debug src/cache.py CacheIndex traceback",
        score=0.1,
        memory_tier="episodic",
        channel="agent",
        task_id="",
        validation_status="observed",
        created_at="2026-09-03T12:00:00+00:00",
        metadata={},
    )
    stages = [
        RagStage("reflex", [probe], 1.8, novelty=1.0, breadth=1),
        RagStage("working", [probe], 1.6, novelty=0.8, breadth=3),
        RagStage("subsurface", [probe], 1.5, novelty=0.7, breadth=6),
    ]
    policy_query = "debug src/cache.py CacheIndex traceback"
    for _ in range(6):
        await store.record_recall_outcome(policy_query, stages)
    policy = await store.policy_hint(policy_query)

    print("Rhomgaic phase-4 adaptive substrate benchmark")
    print(f"memories={memories} queries={queries}")
    print(f"ingest_seconds={ingest_seconds:.3f}")
    print(f"ingest_per_second={memories / max(1e-9, ingest_seconds):.1f}")
    print(f"capsules={capsules['capsules']} distill_ms={distill_ms:.3f}")
    print(f"dependency_recalled={dependency_recalled}")
    print(f"search_mean_ms={statistics.fmean(latencies):.3f}")
    print(f"search_p50_ms={percentile(latencies, 0.50):.3f}")
    print(f"search_p95_ms={percentile(latencies, 0.95):.3f}")
    print(f"search_p99_ms={percentile(latencies, 0.99):.3f}")
    print(f"exact_target_top1={exact / max(1, queries):.3f}")
    print(f"policy={policy}")
    print(f"stats={await store.stats()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Rhomgaic phase-4 derived memory surfaces.")
    parser.add_argument("--memories", type=int, default=600)
    parser.add_argument("--queries", type=int, default=80)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.memories < 8 or args.queries < 1:
        raise SystemExit("--memories must be >= 8 and --queries positive")
    if args.root:
        args.root.mkdir(parents=True, exist_ok=True)
        asyncio.run(run(args.memories, args.queries, args.root))
        return
    with tempfile.TemporaryDirectory(prefix="rhomgaic-v4-bench-") as temp:
        asyncio.run(run(args.memories, args.queries, Path(temp)))


if __name__ == "__main__":
    main()
