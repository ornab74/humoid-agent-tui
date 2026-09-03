from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from pathlib import Path

from humoid_tui.config import Settings
from humoid_tui.memory import MemoryRouter


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[index]


async def run(memories: int, repeats: int, root: Path) -> None:
    router = MemoryRouter(Settings(
        humoid_memory_backend="sqlite",
        humoid_memory_db=root / "provider.sqlite3",
        humoid_rhomgaic_db=root / "rhomgaic.sqlite3",
        humoid_rag_context_cache=True,
        humoid_rag_context_cache_ttl_seconds=3600,
    ))
    await router.initialize()
    for index in range(memories):
        await router.add(
            f"Fixed src/cache.py CacheIndex invalidation issue_{index:06d} and tests pass",
            task_id=f"task-{index // 5}",
            validation_status="verified" if index % 4 == 0 else "observed",
        )

    target = max(0, memories * 2 // 3)
    query = f"fix src/cache.py CacheIndex invalidation issue_{target:06d}"
    started = time.perf_counter()
    cold_packet = await router.context_packet(query, depth="working")
    cold_ms = (time.perf_counter() - started) * 1000.0

    hot_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        hot_packet = await router.context_packet(query, depth="working")
        hot_ms.append((time.perf_counter() - started) * 1000.0)
        if hot_packet != cold_packet:
            raise RuntimeError("hot context packet diverged from cold packet")

    print("Rhomgaic compiled-context cache benchmark")
    print(f"memories={memories} repeats={repeats}")
    print(f"cold_context_ms={cold_ms:.3f}")
    print(f"hot_context_mean_ms={statistics.fmean(hot_ms):.3f}")
    print(f"hot_context_p50_ms={percentile(hot_ms, 0.50):.3f}")
    print(f"hot_context_p95_ms={percentile(hot_ms, 0.95):.3f}")
    print(f"hot_context_p99_ms={percentile(hot_ms, 0.99):.3f}")
    print(f"speedup={cold_ms / max(1e-9, statistics.fmean(hot_ms)):.1f}x")
    print(f"status={await router.surface_status()}")
    await router.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Rhomgaic compiled context cache.")
    parser.add_argument("--memories", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.memories < 1 or args.repeats < 1:
        raise SystemExit("--memories and --repeats must be positive")
    if args.root:
        args.root.mkdir(parents=True, exist_ok=True)
        asyncio.run(run(args.memories, args.repeats, args.root))
        return
    with tempfile.TemporaryDirectory(prefix="rhomgaic-cache-bench-") as temp:
        asyncio.run(run(args.memories, args.repeats, Path(temp)))


if __name__ == "__main__":
    main()
