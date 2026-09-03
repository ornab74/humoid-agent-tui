from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from pathlib import Path

from humoid_tui.maintenance import semantic_change_fingerprint
from humoid_tui.rhomgaic_memory import RhomgaicMemory


async def run(procedures: int, queries: int) -> None:
    with tempfile.TemporaryDirectory(prefix="rhomgaic-v5-") as directory:
        memory = RhomgaicMemory(Path(directory) / "rhomgaic.sqlite3")
        await memory.initialize()
        start = time.perf_counter()
        for index in range(procedures):
            family = index % 20
            trajectory = {
                "request": f"debug src/service_{family}.py Handler{family} regression issue-{index}",
                "operation": "debug",
                "changed_paths": [f"src/service_{family}.py", f"tests/test_service_{family}.py"],
                "symbols": [f"Handler{family}"],
                "risks": ["network" if family % 2 else "storage"],
                "validators": [f"python -m pytest tests/test_service_{family}.py"],
                "validated": index % 11 != 0,
                "steps": [
                    {"kind": "analyze_change_impact"},
                    {"kind": "read_file_range"},
                    {"kind": "apply_patch"},
                    {"kind": "run_command"},
                ],
            }
            await memory.add(
                f"procedure-{index}",
                f"CODING PROCEDURE MEMORY service {family} issue-{index}",
                memory_tier="procedural",
                validation_status="verified" if trajectory["validated"] else "observed",
                metadata={"coding_trajectory": trajectory},
            )
        ingest_seconds = time.perf_counter() - start

        latencies = []
        hits = 0
        for index in range(queries):
            family = index % 20
            query = f"debug src/service_{family}.py Handler{family} regression"
            started = time.perf_counter()
            result = await memory.search_procedures(query, limit=4)
            latencies.append((time.perf_counter() - started) * 1000.0)
            if result and result[0]["action"] == "debug" and f"src/service_{family}.py" in result[0]["files"]:
                hits += 1

        before = "import math\n\ndef calculate(value):\n    return value + 1\n"
        after = "import decimal\n\ndef calculate(value, *, scale=1):\n    if value < 0:\n        raise ValueError('negative')\n    return value * scale\n"
        diff_latencies = []
        for _ in range(max(50, queries)):
            started = time.perf_counter()
            fingerprint = semantic_change_fingerprint("src/math_service.py", before, after)
            diff_latencies.append((time.perf_counter() - started) * 1000.0)
        stats = await memory.stats()

        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else 0.0
        print(f"procedural supports:       {procedures}")
        print(f"aggregated procedures:     {stats['coding_procedures']}")
        print(f"procedure ingest:          {ingest_seconds:.3f} s")
        print(f"procedure recall accuracy: {hits / max(1, queries):.3f}")
        print(f"procedure search mean:     {statistics.mean(latencies):.3f} ms")
        print(f"procedure search p95:      {p95:.3f} ms")
        print(f"semantic diff mean:        {statistics.mean(diff_latencies):.3f} ms")
        print(f"semantic diff risk:        {fingerprint.risk_score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--procedures", type=int, default=400)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    asyncio.run(run(max(1, args.procedures), max(1, args.queries)))


if __name__ == "__main__":
    main()
