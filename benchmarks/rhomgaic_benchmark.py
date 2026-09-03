from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import tempfile
import time
from pathlib import Path

from humoid_tui.rhomgaic_memory import RhomgaicMemory

FILES = [
    "src/agent.py",
    "src/cache.py",
    "src/memory.py",
    "src/providers.py",
    "src/tools.py",
    "tests/test_memory.py",
    "tests/test_agent.py",
    "docs/architecture.md",
]
ACTIONS = ["fixed", "implemented", "tested", "debugged", "refactored", "inspected"]
SYMBOLS = ["AgentHarness", "CacheIndex", "MemoryRouter", "ProviderStream", "ToolRegistry", "ContextSurface"]
TOPICS = ["invalidation", "streaming", "retrieval", "validation", "routing", "context", "fallback", "graph"]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[index]


def make_memory(index: int, rng: random.Random) -> str:
    path = FILES[index % len(FILES)]
    action = ACTIONS[index % len(ACTIONS)]
    symbol = SYMBOLS[index % len(SYMBOLS)]
    topic = TOPICS[index % len(TOPICS)]
    noise = " ".join(rng.sample(TOPICS, k=3))
    # A unique issue token makes exact-target recall measurable. Without it,
    # many synthetic rows are intentionally equivalent in file/symbol/topic
    # coordinates, so an exact memory-id top-1 metric is not well-defined.
    issue = f"issue_{index:06d}"
    return (
        f"{action} {path} {symbol} {topic} {issue}; "
        f"supporting evidence {noise}; revision={index}"
    )


def make_query(index: int) -> str:
    path = FILES[index % len(FILES)]
    action = ACTIONS[index % len(ACTIONS)]
    symbol = SYMBOLS[index % len(SYMBOLS)]
    topic = TOPICS[index % len(TOPICS)]
    return f"{action} {path} {symbol} {topic} issue_{index:06d}"


async def run(memories: int, queries: int, dimensions: int, path: Path) -> None:
    rng = random.Random(73021)
    store = RhomgaicMemory(path, dimensions=dimensions)
    await store.initialize()

    started = time.perf_counter()
    for index in range(memories):
        await store.add(
            f"memory-{index}",
            make_memory(index, rng),
            task_id=f"task-{index // 5}",
            validation_status="verified" if index % 3 == 0 else "observed",
        )
    build_seconds = time.perf_counter() - started

    latencies_ms: list[float] = []
    top1 = 0
    for query_index in range(queries):
        target = (query_index * 97) % memories
        words = make_query(target)
        started = time.perf_counter()
        hits = await store.search(words, limit=8)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        if hits and hits[0].memory_id == f"memory-{target}":
            top1 += 1

    stats = await store.stats()
    db_bytes = path.stat().st_size if path.exists() else 0
    print("Rhomgaic synthetic benchmark")
    print(f"memories={memories} queries={queries} dimensions={dimensions}")
    print(f"build_seconds={build_seconds:.3f}")
    print(f"ingest_per_second={memories / max(build_seconds, 1e-9):.1f}")
    print(f"search_mean_ms={statistics.fmean(latencies_ms):.3f}")
    print(f"search_p50_ms={percentile(latencies_ms, 0.50):.3f}")
    print(f"search_p95_ms={percentile(latencies_ms, 0.95):.3f}")
    print(f"search_p99_ms={percentile(latencies_ms, 0.99):.3f}")
    print(f"exact_target_top1={top1 / max(1, queries):.3f}")
    print(f"sqlite_bytes={db_bytes}")
    print(f"stats={stats}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Rhomgaic candidate collapse and recall.")
    parser.add_argument("--memories", type=int, default=5000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--dimensions", type=int, default=96)
    parser.add_argument("--db", type=Path)
    args = parser.parse_args()
    if args.memories < 1 or args.queries < 1:
        raise SystemExit("--memories and --queries must be positive")
    if args.db:
        args.db.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(run(args.memories, args.queries, args.dimensions, args.db))
        return
    with tempfile.TemporaryDirectory(prefix="rhomgaic-bench-") as temp:
        asyncio.run(run(args.memories, args.queries, args.dimensions, Path(temp) / "memory.sqlite3"))


if __name__ == "__main__":
    main()
