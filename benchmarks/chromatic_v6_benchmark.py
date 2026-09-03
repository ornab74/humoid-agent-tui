from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from pathlib import Path

from humoid_tui.rhomgaic_memory import RhomgaicMemory


async def run(memories: int, queries: int) -> None:
    with tempfile.TemporaryDirectory(prefix="humoid-chromatic-v6-") as directory:
        store = RhomgaicMemory(Path(directory) / "rhomgaic.sqlite3", candidate_limit=512)
        await store.initialize()
        start = time.perf_counter()
        for index in range(memories):
            package = index % 20
            symbol = f"Resolver{index}"
            path = f"src/domain_{package}/resolver_{index}.py"
            issue = f"issue_{index:06d}"
            await store.add(
                f"m-{index}",
                f"debug {path} {symbol} {issue} regression callers dependency tests validation",
                validation_status="verified" if index % 4 == 0 else "observed",
                task_id=f"task-{package}",
            )
        ingest_seconds = time.perf_counter() - start

        timings = []
        exact = 0
        avoided = []
        for offset in range(queries):
            index = (offset * 7919) % memories
            query = (
                f"debug src/domain_{index % 20}/resolver_{index}.py "
                f"Resolver{index} issue_{index:06d} regression"
            )
            started = time.perf_counter()
            hits = await store.search(query, limit=5)
            timings.append((time.perf_counter() - started) * 1000.0)
            if hits and hits[0].memory_id == f"m-{index}":
                exact += 1
            trace = store.last_chromatic_trace
            universe = max(1, int(trace.get("universe", memories)))
            avoided.append(float(trace.get("candidates_avoided", 0)) / universe)

        stats = await store.stats()
        print(f"memories={memories}")
        print(f"queries={queries}")
        print(f"ingest_seconds={ingest_seconds:.4f}")
        print(f"ingest_per_second={memories / max(ingest_seconds, 1e-9):.1f}")
        print(f"top1={exact / max(1, queries):.3f}")
        print(f"search_mean_ms={statistics.mean(timings):.3f}")
        print(f"search_p50_ms={statistics.median(timings):.3f}")
        ordered = sorted(timings)
        print(f"search_p95_ms={ordered[min(len(ordered)-1, int(len(ordered)*0.95))]:.3f}")
        print(f"candidate_avoidance_mean={statistics.mean(avoided):.3f}")
        print(f"hierarchy_postings={stats['chromatic_hierarchy_postings']}")
        print(f"graph_edges={stats['graph_edges']}")
        print(f"schema_version={stats['schema_version']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memories", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    asyncio.run(run(max(20, args.memories), max(1, args.queries)))


if __name__ == "__main__":
    main()
