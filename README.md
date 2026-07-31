# Humoid Agent TUI

An advanced Python terminal agent harness modeled after the supplied split-pane agent screenshot.

## Providers

The same orchestration loop can switch at runtime between:

- DigitalOcean Gradient AI / GLM 5.2
- Meta Model API / Muse Spark 1.1
- OpenAI
- local llama.cpp
- LiteRT-LM through an OpenAI-compatible local shim

Use `/provider meta`, `/provider digitalocean`, `/provider openai`, `/provider llamacpp`, or `/provider litert`.

## Memory architecture

`MemoryRouter` attempts Weaviate first, then safely falls back to SQLite. The Weaviate startup gate checks:

1. Liveness/readiness
2. Collection schema
3. Full sentinel insert → fetch → BM25 query → delete round trip

The memory packet includes typed `MemoryHit` fields (`memory_tier`, `channel`, `task_id`,
`validation_status`), temporal neighbors, and an MMR-like diversity pass. This directly prevents
the missing-`memory_tier` class mismatch seen in earlier Humoid-style pipelines.

## Install

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -e ".[weaviate,dev]"
docker compose up -d
humoid
```

For SQLite-only mode:

```bash
sed -i 's/HUMOID_MEMORY_BACKEND=auto/HUMOID_MEMORY_BACKEND=sqlite/' .env
pip install -e .
humoid
```

## Local llama.cpp

```bash
llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8080
```

Then set `LLAMACPP_MODEL` and run `/provider llamacpp`.

### Install and test Gemma 4 E2B locally (CPU only)

The following setup downloads the 4-bit Gemma 4 E2B GGUF and runs it entirely
on the CPU. Run these commands inside the project's virtual environment.

#### Step 1: Install the CPU runtime and downloader

Force a clean installation of `llama-cpp-python` and Hugging Face Hub:

```bash
pip install --force-reinstall --no-cache-dir llama-cpp-python huggingface-hub
```

Depending on the operating system, installing `llama-cpp-python` may compile
llama.cpp locally and therefore require a C/C++ compiler and CMake.

#### Step 2: Create the CPU-only boot script

Save the following as `boot_gemma_cpu.py`:

```python
import multiprocessing

from huggingface_hub import hf_hub_download
from llama_cpp import Llama


# Leave one logical CPU available for the operating system.
cpu_threads = max(1, multiprocessing.cpu_count() - 1)

# Download the 4-bit Gemma 4 E2B GGUF. Hugging Face Hub reuses its
# local cache on later runs.
repo_id = "google/gemma-4-E2B-it-qat-q4_0-gguf"
filename = "gemma-4-E2B_q4_0-it.gguf"

print("Checking for model file...")
model_path = hf_hub_download(repo_id=repo_id, filename=filename)
print(f"Model file loaded from cache: {model_path}\n")

# Set n_gpu_layers=0 to keep all model layers in system RAM.
print(
    f"Booting Gemma 4 with a 128K-token context using "
    f"{cpu_threads} CPU threads..."
)
llm = Llama(
    model_path=model_path,
    n_ctx=128000,
    n_threads=cpu_threads,
    n_gpu_layers=0,
    flash_attn=True,
    verbose=False,
)
print("Model loaded into RAM successfully!\n")

# Format a simple prompt with Gemma's turn markers.
prompt = (
    "<start_of_turn>user\n"
    "Give me a 3-word motivational phrase."
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)

print("Generating response...")
response = llm(
    prompt,
    max_tokens=50,
    temperature=0.7,
)

print("\nResult:")
print(response["choices"][0]["text"])
```

`n_ctx=128000` means approximately 128,000 tokens, not 128 KB. A context this
large can require substantial system RAM even with a quantized model. If the
process is killed or memory allocation fails, begin with `n_ctx=8192` or
`n_ctx=32768` and increase it gradually. Flash-attention availability and its
CPU behavior depend on the installed llama.cpp build.

#### Step 3: Run the script

```bash
python boot_gemma_cpu.py
```

This script is a standalone generation test. To connect Gemma to Humoid, serve
the downloaded GGUF with an OpenAI-compatible `llama-server`, set the
`LLAMACPP_*` values in `.env`, start Humoid, and select `/provider llamacpp`.

## Safety

File tools are restricted to `HUMOID_WORKSPACE`. Shell execution is disabled by default.
Turn it on only in an isolated repository/container. Shell commands start in
`HUMOID_WORKSPACE`, but the subprocess retains the permissions of the operating-system
user and is not contained by the file-tool path fence.

## Adaptive TOOL.CALL pipeline

The model never talks directly to executors. Every dialect first becomes a canonical
`CanonicalToolCall` IR:

```text
⟪TOOL.CALL/v1 REQUEST id=call_... agent=root provider=... model=...
protocol=... sha=... confidence=...⟫
{
  "tool": "read_file",
  "arguments": {"path": "src/main.py"},
  "metadata": {}
}
⟪/TOOL.CALL⟫
```

The envelope is the trace/audit representation, not a reason to force every model to
speak the same unnatural syntax. `ProtocolRegistry` picks the best edge codec:

- GPT/OpenAI: native structured function calls; strict call IDs and tool-result pairing.
- DigitalOcean GLM/Qwen-style models: OpenAI-compatible calls first, tagged JSON repair fallback.
- Gemma 4: server-normalized calls first, then native `<|tool_call>` control-token parsing.
- llama.cpp: native handler when `--jinja` recognizes the template, then generic/tagged fallback.
- LiteRT-LM shims: OpenAI-compatible first, native Gemma/tagged formats second.
- Meta Muse Spark: OpenAI-compatible transport with isolated Meta request extensions.

This separation enables one policy layer for allowlists, argument-size limits, workspace
containment, result correlation, timings, hashes, memory provenance, and future approvals.

## Gemma 4 native agent setting

Gemma 4 support can be changed from `.env` or live inside the TUI:

```text
/gemma4                 # show status
/gemma4 auto            # activate only for Gemma 4 model names
/gemma4 native          # force native Gemma 4 control-token protocol
/gemma4 off             # disable Gemma-specific behavior
/gemma-thinking off
/gemma-thinking low     # recommended adaptive low-thinking mode
/gemma-thinking on
```

Environment equivalents:

```env
HUMOID_GEMMA4_MODE=auto
HUMOID_GEMMA4_THINKING=low
HUMOID_GEMMA4_NATIVE_FALLBACK=true
HUMOID_GEMMA4_STRIP_COMPLETED_THOUGHTS=true
```

The mode renders Gemma 4 tool declarations with `<|tool>...<tool|>`, parses
`<|tool_call>...<tool_call|>`, returns `<|tool_response>...<tool_response|>`, and
uses `<|"|>` around strings. Thought channels are retained during a chained tool
turn and removed from completed conversation history by default.

## Ultra dashboard and model-native prompt profiles

Version 0.3 adds the screenshot-inspired dashboard: activity stream, agent tree, session
telemetry, conversation/workspace panel, native TOOL.CALL/TOOL.RESULT cards, context meter,
provider/protocol header, command line, and function-key footer.

The runtime now selects a prompt and protocol profile from the active provider/model:

- `gpt-5.6*`: Responses-oriented profile, direct vs programmatic tool routing, reasoning lineage.
- `gemma-4*`: native Gemma 4 lifecycle, single-call-safe prompting, strict literal arguments.
- `glm-5.2`: OpenAI-compatible structured calls first, tagged-JSON fallback.
- `muse-spark*`: parallel delegation, milestone compaction, evidence-preserving subagents.
- `llama.cpp`: embedded Jinja template first, portable fallback when parsing is unavailable.
- `LiteRT-LM`: conservative local profile designed for an OpenAI-compatible shim.

Runtime commands:

```text
/provider digitalocean|meta|openai|llamacpp|litert
/profile
/autonomy review|balanced|autonomous
/gemma4 auto|native|off
/gemma-thinking off|low|on
/health
```

## Advanced harness controls

The dashboard has a persistent bottom navigation bar containing **Help**,
**Sessions**, **Memory**, **Context**, **Settings + Models**, and a bottom-left
**Language** selector. From the command input,
press **Down** to focus the bar, use **Left/Right** to highlight a page, and
press **Enter** to open it. Press **Up** to return to the command input. Each
page includes a **Back** button at its top-left, so this navigation does not
depend on function keys or operating-system shortcuts.

Inside a page, **Up/Down** advances through ordinary controls and
**Left/Right** moves between buttons. **Enter** or **Space** activates the
highlighted control. Tables and multiline editors retain their arrow keys;
use **Tab/Shift+Tab** to enter or leave those controls.

In the Memory page, **Enter/Space** toggles a visible `✓` on the highlighted
row. **Up/Down** moves between memories; Down on the final row exits the table
to the editor and then the action buttons. Verify and Delete apply to all
checked rows, while Save Edit applies to the currently highlighted memory.

The Help page provides selectable All, Navigation, Memory, Models, and Tools
categories plus command search. The Context page shows live token utilization
and folded-archive counts with Expand All and Collapse All controls. Settings
lists downloaded GGUF files in a keyboard-navigable table; Enter selects a
model with a visible `✓`, and Inspect, Launch, or Benchmark use that selection.

The Sessions page creates independent conversation tabs and can switch,
rename, or close them while preserving each session's model messages and
Context Accordion. Active sessions appear beneath the top status bar.

The Language page applies translations immediately and saves the selected
locale as `HUMOID_LANGUAGE` in `.env`. The initial catalog includes English,
French, Kiswahili, Amharic, Hausa, Yoruba, isiZulu, Arabic, Hindi, Bengali,
Urdu, Chinese, Japanese, Korean, and Indonesian, with native-script labels and
an emphasis on African and Asian regions.

- **Ctrl+1** opens the complete keyboard and slash-command reference.
- **Ctrl+2** opens the memory browser with search, inspection, editing,
  verification, and deletion for SQLite or Weaviate memories.
- **Ctrl+3** opens the Context Accordion. Older context is folded into structured
  decisions, unresolved work, file references, and tool evidence while its
  original source remains available for expansion.
- **Ctrl+4** opens persistent settings and the local-model manager. It detects
  CPU, Apple Metal, AMD ROCm, or NVIDIA hardware and CUDA version; recommends
  a compatible `llama-cpp-python` install; and provides explicit install,
  reinstall, and uninstall actions.
- **F1/F5/F6/F12** remain secondary aliases for terminals and operating systems
  that forward function keys to the application.
- **Esc** or **Ctrl+C** cancels the active provider request or subprocess.

The local-model manager can download, inspect, launch, stop, and benchmark
GGUF models. Downloads run in a cancellable background worker and display
downloaded bytes, total size, percentage, transfer rate, ETA, and final status,
so the rest of the TUI remains responsive. Settings saved through the manager are written to `.env` while
preserving existing comments and unrelated values. NVIDIA prebuilt wheels are
selected only for the CUDA and Python versions published by the official
`llama-cpp-python` project; unsupported combinations fall back to the normal
source-install path instead of guessing a wheel URL.

Agent file writes now open a unified-diff preview. A user can approve the
original content, edit it before approval, reject it, or restore the previous
file using `/undo`. The `invent_tool` tool can construct session-scoped,
read-only instruments from existing tools, validate them against supplied
cases, and optionally promote their specifications into Humoid runtime state.

The implementation intentionally keeps model-native syntax at the transport boundary and
normalizes all calls to `CanonicalToolCall` before validation, audit logging, policy checks,
and execution.
