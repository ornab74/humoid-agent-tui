from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from pathlib import Path

from humoid_tui.chromatic_invariants import ChromaticInvariantCompiler
from humoid_tui.rhomgaic_memory import RhomgaicMemory


def build_repo(root: Path, modules: int) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='bench'\n", encoding="utf-8")
    for index in range(modules):
        package = root / "src" / f"domain_{index % 10}"
        package.mkdir(parents=True, exist_ok=True)
        path = package / f"service_{index}.py"
        path.write_text(
            "import math\n\n"
            f"def compute_{index}(items):\n"
            "    if not items:\n"
            "        raise ValueError('items')\n"
            "    return sum(items)\n",
            encoding="utf-8",
        )
        (root / "tests" / f"test_service_{index}.py").write_text(
            f"from src.domain_{index % 10}.service_{index} import compute_{index}\n\n"
            f"def test_compute_{index}():\n"
            f"    assert compute_{index}([1, 2]) == 3\n",
            encoding="utf-8",
        )


async def run(modules: int, queries: int) -> None:
    with tempfile.TemporaryDirectory(prefix="humoid-chromatic-v7-") as directory:
        root = Path(directory) / "repo"
        build_repo(root, modules)
        compiler = ChromaticInvariantCompiler(root)
        memory = RhomgaicMemory(Path(directory) / "rhomgaic.sqlite3")
        await memory.initialize()

        contracts = []
        compile_ms = []
        evaluate_ms = []
        for offset in range(queries):
            index = (offset * 7919) % modules
            path = f"src/domain_{index % 10}/service_{index}.py"
            symbol = f"compute_{index}"
            query = f"Fix {symbol} in {path} without breaking callers and verify tests."
            started = time.perf_counter()
            contract = compiler.compile(
                query,
                operation="debug",
                files=[path],
                symbols=[symbol],
                acceptance=["tests should pass"],
            )
            compile_ms.append((time.perf_counter() - started) * 1000.0)
            contracts.append(contract)
            started = time.perf_counter()
            report = compiler.evaluate(
                contract,
                changed_paths=[path],
                validators=[f"python -m pytest tests/test_service_{index}.py"],
                validated=True,
            )
            evaluate_ms.append((time.perf_counter() - started) * 1000.0)
            await memory.add(
                f"procedure-{offset}",
                f"CODING PROCEDURE MEMORY {symbol} compatibility repair",
                memory_tier="procedural",
                validation_status="verified",
                metadata={"coding_trajectory": {
                    "request": query,
                    "operation": "debug",
                    "changed_paths": [path, f"tests/test_service_{index}.py"],
                    "symbols": [symbol],
                    "validators": [f"python -m pytest tests/test_service_{index}.py"],
                    "validated": True,
                    "invariant_contract": contract.to_dict(),
                    "invariant_proof": report.to_dict(),
                    "steps": [{"kind": "apply_patch"}, {"kind": "run_command"}],
                }},
            )

        recall_ms = []
        exact_kind = 0
        for contract in contracts:
            started = time.perf_counter()
            hits = await memory.search_invariant_procedures(contract.to_dict(), limit=4)
            recall_ms.append((time.perf_counter() - started) * 1000.0)
            if hits and "public-signature" in hits[0]["matched_kinds"]:
                exact_kind += 1

        print(f"modules={modules}")
        print(f"queries={queries}")
        print(f"compile_mean_ms={statistics.mean(compile_ms):.3f}")
        print(f"evaluate_mean_ms={statistics.mean(evaluate_ms):.3f}")
        print(f"invariant_history_mean_ms={statistics.mean(recall_ms):.3f}")
        ordered = sorted(recall_ms)
        print(f"invariant_history_p95_ms={ordered[min(len(ordered)-1, int(len(ordered)*0.95))]:.3f}")
        print(f"invariant_kind_match={exact_kind / max(1, queries):.3f}")
        print(f"invariants_per_contract_mean={statistics.mean(len(item.invariants) for item in contracts):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modules", type=int, default=200)
    parser.add_argument("--queries", type=int, default=60)
    args = parser.parse_args()
    asyncio.run(run(max(20, args.modules), max(1, args.queries)))


if __name__ == "__main__":
    main()
