from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(slots=True)
class HardwareProfile:
    backend: str
    name: str
    cuda_version: str = ""
    detail: str = ""


class LocalModelManager:
    """Hardware detection and lifecycle management for local GGUF models."""

    CUDA_WHEELS: ClassVar = {(12, minor): f"cu12{minor}" for minor in range(1, 6)}

    def __init__(self, model_dir: Path = Path(".humoid/models")) -> None:
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.process: asyncio.subprocess.Process | None = None
        self.log_path = Path(".humoid/llama-server.log")
        self._download_cancel = threading.Event()

    async def _command(self, *args: str, timeout: float = 8) -> tuple[int, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await asyncio.wait_for(process.communicate(), timeout)
            return process.returncode or 0, output.decode(errors="replace")
        except (FileNotFoundError, TimeoutError) as exc:
            return 1, str(exc)

    async def detect_hardware(self) -> HardwareProfile:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return HardwareProfile("metal", "Apple Silicon", detail=platform.platform())
        code, output = await self._command(
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        )
        if code == 0 and output.strip():
            _, version_output = await self._command("nvcc", "--version")
            match = re.search(r"release\s+(\d+\.\d+)", version_output)
            cuda = match.group(1) if match else ""
            return HardwareProfile("cuda", output.strip().splitlines()[0], cuda, version_output.strip())
        if shutil.which("rocminfo"):
            return HardwareProfile("rocm", "AMD ROCm GPU", detail="rocminfo detected")
        return HardwareProfile(
            "cpu",
            platform.processor() or platform.machine() or "CPU",
            detail=f"{os.cpu_count() or 1} logical cores; {platform.platform()}",
        )

    def package_version(self) -> str:
        try:
            return importlib.metadata.version("llama-cpp-python")
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    def install_command(self, hardware: HardwareProfile, reinstall: bool = False) -> list[str]:
        command = [sys.executable, "-m", "pip", "install"]
        if reinstall:
            command += ["--force-reinstall", "--no-cache-dir"]
        # The OpenAI-compatible local provider needs the optional FastAPI/
        # Uvicorn server dependencies as well as the core Python bindings.
        command.append("llama-cpp-python[server]")
        python_supported = sys.version_info[:2] in {(3, 10), (3, 11), (3, 12)}
        if hardware.backend == "cuda" and hardware.cuda_version:
            major_minor = tuple(int(part) for part in hardware.cuda_version.split(".")[:2])
            wheel = self.CUDA_WHEELS.get(major_minor)
            if wheel and python_supported:
                command += ["--extra-index-url", f"https://abetlen.github.io/llama-cpp-python/whl/{wheel}"]
        elif hardware.backend == "cpu" and python_supported:
            command += ["--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu"]
        return command

    async def mutate_package(self, action: str, hardware: HardwareProfile) -> tuple[int, str]:
        if action == "uninstall":
            command = [sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"]
        elif action in {"install", "reinstall"}:
            command = self.install_command(hardware, reinstall=action == "reinstall")
        else:
            raise ValueError(f"Unknown package action: {action}")
        return await self._command(*command, timeout=1800)

    def models(self) -> list[Path]:
        return sorted(self.model_dir.glob("*.gguf"))

    def cancel_download(self) -> None:
        self._download_cancel.set()

    async def download(
        self,
        repo_id: str,
        filename: str,
        progress: Callable[[dict[str, float | int | str]], None] | None = None,
    ) -> Path:
        # Xet is the Hub's default transport for many large GGUF repositories, but
        # its native client can stall before constructing a tqdm bar when its
        # cache is unavailable (common in sandboxed/read-only home directories).
        # The regular HTTP transport is slightly less optimized, but gives us
        # dependable progress and cancellation callbacks. Users can explicitly
        # opt back in by setting HF_HUB_DISABLE_XET=0 before starting Humoid.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        xet_cache = self.model_dir / ".xet-cache"
        xet_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_XET_CACHE", str(xet_cache.resolve()))

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face Hub is not installed. Run: "
                "python -m pip install huggingface-hub"
            ) from exc

        from tqdm.auto import tqdm

        self._download_cancel.clear()
        cancel_event = self._download_cancel
        loop = asyncio.get_running_loop()

        if progress is not None:
            progress({
                "downloaded": 0,
                "total": 0,
                "rate": 0.0,
                "filename": filename,
                "phase": "Connecting to Hugging Face",
            })

        def report(bar) -> None:
            if progress is None:
                return
            total = int(bar.total or 0)
            downloaded = int(bar.n or 0)
            rate = float(bar.format_dict.get("rate") or 0.0)
            loop.call_soon_threadsafe(progress, {
                "downloaded": downloaded,
                "total": total,
                "rate": rate,
                "filename": filename,
                "phase": "Downloading",
            })

        class ProgressTqdm(tqdm):
            def update(self, n=1):
                if cancel_event.is_set():
                    raise RuntimeError("Model download cancelled")
                result = super().update(n)
                report(self)
                return result

            def close(self):
                report(self)
                return super().close()

        try:
            downloaded = await asyncio.to_thread(
                hf_hub_download,
                repo_id=repo_id,
                filename=filename,
                local_dir=str(self.model_dir),
                tqdm_class=ProgressTqdm,
            )
        except Exception as exc:
            error_name = type(exc).__name__
            if error_name == "GatedRepoError" or (
                error_name == "HfHubHTTPError"
                and any(status in str(exc) for status in ("401", "403"))
            ):
                raise RuntimeError(
                    "Google Gemma access requires accepting the model license at "
                    f"https://huggingface.co/{repo_id}, then authenticating with: hf auth login"
                ) from exc
            if error_name in {"RemoteEntryNotFoundError", "EntryNotFoundError"}:
                raise RuntimeError(
                    f"File {filename!r} does not exist in {repo_id!r}. "
                    "Refresh the official repository file list before retrying."
                ) from exc
            raise
        return Path(downloaded)

    async def inspect(self, path: Path) -> dict[str, object]:
        stat = path.stat()
        return {"name": path.name, "bytes": stat.st_size, "modified": stat.st_mtime}

    async def launch(self, path: Path, port: int = 8080, context: int = 10000) -> str:
        if self.process and self.process.returncode is None:
            return (
                f"server already running pid={self.process.pid} port={port}; "
                "use STOP before launching a different model or context"
            )
        missing = []
        for module in ("uvicorn", "fastapi", "sse_starlette"):
            try:
                available = importlib.util.find_spec(module)
            except (ImportError, ValueError):
                missing.append(module)
            else:
                if available is None:
                    missing.append(module)
        if missing:
            raise RuntimeError(
                "Local server dependencies are missing: "
                f"{', '.join(missing)}. Click INSTALL (or REINSTALL) to install "
                "llama-cpp-python with server support."
            )
        command = [
            sys.executable, "-m", "llama_cpp.server",
            "--model", str(path.resolve()), "--host", "127.0.0.1",
            "--port", str(port), "--n_ctx", str(context),
            # Keep the CPU-only server within small-host memory limits. The
            # server otherwise locks the entire model and uses 512-token
            # processing batches in addition to its runtime repack buffer.
            "--use_mmap", "true", "--use_mlock", "false",
            "--n_batch", "128", "--n_ubatch", "64",
            # llama-cpp-python's server defaults this to true, allocating
            # n_ctx * vocab float32 scores. Gemma 4's 262,144-token vocab
            # makes that 9.77 GiB at a 10k context. Chat only needs the
            # latest-token logits.
            "--logits_all", "false",
        ]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        try:
            self.process = await asyncio.create_subprocess_exec(
                *command, stdout=log, stderr=log,
            )
        finally:
            log.close()
        return (
            f"server pid={self.process.pid} model={path.name} port={port} "
            f"context={context}"
        )

    async def wait_until_ready(self, host: str = "127.0.0.1", port: int = 8080, timeout: float = 180) -> None:
        """Wait until the managed llama.cpp server accepts connections."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process and self.process.returncode is not None:
                detail = ""
                if self.log_path.exists():
                    detail = self.log_path.read_text(errors="replace")[-1500:]
                if self.process.returncode in {-9, 137}:
                    raise RuntimeError(
                        "Local model server was killed by the OS, most likely due to "
                        "insufficient RAM. This host has no swap and Gemma's CPU-repacked "
                        "weights require about 4.2 GiB before context allocation. Close "
                        "other memory-heavy processes, add swap, or use a smaller GGUF.\n"
                        f"{detail}"
                    )
                if "ArrayMemoryError" in detail and "self.scores" in detail:
                    raise RuntimeError(
                        "Local model server attempted to allocate full-context logits. "
                        "Humoid now launches it with logits_all disabled; restart the TUI "
                        "to use the corrected launch profile.\n"
                        f"{detail}"
                    )
                raise RuntimeError(f"Local model server stopped during startup.\n{detail}")
            try:
                _reader, writer = await asyncio.open_connection(host, port)
                writer.close()
                await writer.wait_closed()
                return
            except OSError:
                await asyncio.sleep(0.25)
        raise TimeoutError(f"Local model server did not become ready within {timeout:.0f} seconds")

    async def stop(self) -> str:
        if not self.process or self.process.returncode is not None:
            return "no managed llama.cpp server is running"
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), 10)
        except TimeoutError:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
            await self.process.wait()
        return "local model server stopped"

    async def benchmark(self, path: Path) -> dict[str, object]:
        if not path.is_file():
            return {"ok": False, "error": f"Model file not found: {path}"}

        def run() -> dict[str, object]:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                return {
                    "ok": False,
                    "error": "llama-cpp-python is not installed; use the Install button first.",
                    "detail": str(exc),
                }

            started = time.perf_counter()
            try:
                model = Llama(
                    model_path=str(path.resolve()),
                    n_ctx=512,
                    # Leave capacity for Textual and the rest of the system. Using
                    # every logical CPU makes the app appear frozen on CPU-only
                    # machines while llama.cpp loads and evaluates the prompt.
                    n_threads=max(1, min(8, (os.cpu_count() or 2) - 1)),
                    verbose=False,
                )
                loaded = time.perf_counter()
                result = model(
                    "Reply with OK",
                    max_tokens=16,
                    temperature=0.0,
                )
                finished = time.perf_counter()
                usage = result.get("usage", {})
                completion_tokens = int(usage.get("completion_tokens", 0))
                generation_seconds = max(finished - loaded, 0.000001)
                text = str(result.get("choices", [{}])[0].get("text", "")).strip()
                return {
                    "ok": True,
                    "load_seconds": round(loaded - started, 2),
                    "generation_seconds": round(generation_seconds, 2),
                    "completion_tokens": completion_tokens,
                    "tokens_per_second": round(completion_tokens / generation_seconds, 2),
                    "output": text,
                }
            except Exception as exc:  # noqa: BLE001 - surface native model-load failures in the UI
                return {
                    "ok": False,
                    "error": str(exc),
                    "detail": type(exc).__name__,
                }

        return await asyncio.to_thread(run)

    def status_json(self) -> str:
        running = bool(self.process and self.process.returncode is None)
        return json.dumps({
            "package": self.package_version(),
            "models": [p.name for p in self.models()],
            "server": {
                "running": running,
                "pid": self.process.pid if running else None,
                "log": str(self.log_path),
            },
        }, indent=2)
