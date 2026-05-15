# R52 Agent

**Autonomous ARM Cortex-R52 bare-metal coding agent with build/run/validate/diagnose feedback loop.**

## Overview

R52 Agent is an AI-powered coding assistant for ARM Cortex-R52 bare-metal firmware development. Given a task description, it autonomously scouts the hardware platform, generates code, builds and runs it on a simulator, validates output, and diagnoses failures — iterating until the firmware works or the retry budget is exhausted.

The agent is deliberately evidence-driven: rather than encoding platform knowledge in Python, it uses LLM reasoning at runtime to investigate hardware facts, interpret failures, and adapt.

## Architecture

The agent follows a **LangGraph-based state machine**:

```
PLAN → SCOUT → GENERATE → REVIEW → BUILD → RUN → VALIDATE
                   ↑                        ↓
                   └──── PATCH ← DIAGNOSE ──┘
```

### Execution Pipeline

1. **PLAN** — Analyses the task and produces an implementation plan (files to create, build system, startup/linker changes needed).

2. **SCOUT** — Three-phase hardware model extraction:
   - Phase 1: LLM decides what probes to run given the simulator and plan.
   - Phase 2: Python executes probes deterministically (`run_command` / `read_file`).
   - Phase 3: LLM synthesises raw probe outputs into a structured `HardwareModel` with per-field trust levels (`runtime` > `source` > `prior`).
   No simulator-specific knowledge is encoded here — the LLM uses its own knowledge to choose probes (QMP queries, source file reads, grep, dtc, etc.).

3. **GENERATE** — Produces all source files using the verified hardware model as ground truth. Review rejection issues from the previous cycle are fed back explicitly so the generator can address them.

4. **REVIEW** — Functional self-critique: traces execution from reset and checks whether the code is consistent with the hardware model. Can approve with inline corrections or reject with specific issues. After 3 consecutive rejections the graph forces through to BUILD so errors surface concretely.

5. **BUILD** — Compiles with GNU or ARMClang toolchain.

6. **RUN** — Executes on the simulator. A timeout with non-empty stdout is treated as success (bare-metal firmware loops forever by design).

7. **VALIDATE** — Checks simulator stdout against `--expected-output`.

8. **DIAGNOSE** (on RUN failure) — Three-phase failure analysis mirroring SCOUT: LLM plans probes, Python executes, LLM synthesises a `DiagnosisResult` (failure class, root cause, evidence, fix hint, confidence). The diagnosis is passed to PATCH.

9. **PATCH** — Produces corrected files given the full failure context: current code, build/run/validation errors, DIAGNOSE result, hardware model, and prior attempt history.

### Hardware Model

SCOUT produces a `HardwareModel` — a dict of named fields (e.g. `UART0.base`, `BRAM.top`, `UART0.CTRL.TX_EN.bit`) each tagged with a trust level:

| Trust | Meaning |
|-------|---------|
| `runtime` | Value came from a live process (QEMU monitor, QMP query) |
| `source` | Value read from a source file, DTS, SVD, or similar |
| `prior` | Not found in any probe — LLM prior knowledge, treat with caution |

The hardware model is **required**: GENERATE and PATCH raise `RuntimeError` if it is absent. All downstream nodes receive it as ground truth and are instructed not to substitute their own prior knowledge for addresses or sizes.

## Project Structure

```
R52_coding_agent/
├── cli.py                      # Click-based CLI entry point
├── r52_types.py                # Shared Pydantic models and enums
├── agent/
│   ├── graph.py                # LangGraph state machine
│   ├── state.py                # AgentState Pydantic model
│   ├── probe_tools.py          # Shared run_command / read_file primitives
│   ├── nodes/
│   │   ├── scout.py            # Hardware model extraction (new)
│   │   ├── planner.py
│   │   ├── generator.py        # Code generation (uses hardware model)
│   │   ├── reviewer.py         # Functional review (uses hardware model)
│   │   ├── builder.py
│   │   ├── runner.py
│   │   ├── validator.py
│   │   ├── diagnoser.py        # Failure analysis (new)
│   │   └── patcher.py          # Fix generation (uses hardware model + diagnosis)
│   └── prompts/
│       └── system_r52.py
├── backends/
│   ├── anthropic_api.py        # Anthropic API (ANTHROPIC_API_KEY)
│   ├── openai_api.py           # OpenAI API (OPENAI_API_KEY)
│   ├── openrouter_api.py       # OpenRouter (OPENROUTER_API_KEY)
│   ├── claude_cli.py           # Claude Code CLI (no key needed)
│   ├── gemini_cli.py           # Gemini CLI (no key needed)
│   ├── codex_cli.py            # Codex CLI (no key needed)
│   └── base.py
├── toolchain/
│   ├── config.py
│   ├── gnu.py
│   ├── armclang.py
│   └── simulator.py            # mps3-an536 QEMU runner
├── context/
│   └── repo_reader.py
├── templates/
│   └── cortex-r52-baremetal/   # Correct ATCM/BRAM layout, UART at 0xe7c00000
├── observability/
│   ├── logger.py               # Structured JSONL event log
│   ├── tracer.py               # OpenTelemetry tracing
│   └── rich_ui.py
└── eval/
```

## CLI Usage

```bash
# Install
pip install -e .

# Single-shot task
r52 run "Print Hello from R52 over UART0 and loop" \
    --repo ./my_project \
    --simulator qemu \
    --backend claude-cli \
    --expected-output "Hello from R52"

# Reduce run timeout while debugging (default: 600s)
r52 run "..." --timeout 20

# Conversational mode
r52 chat --repo ./my_project

# Run evaluation suite
r52 eval --suite eval_suite.yaml

# View logs
r52 logs --last
```

## Backends

| `--backend` | Default model | Auth |
|-------------|--------------|------|
| `anthropic-api` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| `openai-api` | `gpt-4o` | `OPENAI_API_KEY` |
| `openrouter` | `qwen/qwen3-coder:free` | `OPENROUTER_API_KEY` |
| `claude-cli` | `claude-sonnet-4-6` | Claude Code CLI session |
| `gemini-cli` | `gemini-2.5-pro` | Gemini CLI session |
| `codex-cli` | `codex-latest` | Codex CLI session |

Override the model with `--model`, e.g. `--backend openrouter --model anthropic/claude-opus-4-7`.

### Backend comparison (Hello from R52, mps3-an536, `--timeout 20`)

| Backend | SCOUT fields | Verified | Generate cycles | Success |
|---------|-------------|----------|-----------------|---------|
| `claude-cli` | 24 | 19 (source) | 1 | ✓ |
| `codex-cli` | 57 | 42 (runtime via QMP) | 2 | ✓ |
| `gemini-cli` | 10 | 9 (source) | 2 | ✓ |
| `openrouter` (qwen3:free) | 7 | 3 | 15+ (stuck) | ✗ |

Codex queries the live QEMU machine via QMP — every memory region comes back with `runtime` trust. Smaller/free models tend to hallucinate platform details and burn retry budget in the review loop.

## Observability

Every run writes a structured JSONL log to `~/.r52agent/runs/<trace_id>.jsonl`.

Key event types:

| Event | Contents |
|-------|---------|
| `scout_result` | field count, verified count, per-field values and trust levels |
| `scout_probes` | each probe label, tool, success flag, output snippet |
| `review_result` | approved, issues list, consecutive rejection count |
| `build_result` | success, stderr snippet |
| `run_result` | success, timed_out, stdout snippet |
| `diagnosis_result` | full diagnosis text (failure class, root cause, fix hint) |
| `validation_result` | passed, detail |

```bash
# Tail a live run
tail -f ~/.r52agent/runs/<trace_id>.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    e = json.loads(line)
    print(e['type'], e.get('node',''), e.get('iteration',''))
"
```

## Design Principles

1. **Evidence-driven, not rule-based** — No platform addresses or register names are encoded in Python. The LLM uses two generic primitives (`run_command`, `read_file`) to discover hardware facts at runtime.
2. **Trust is explicit** — Every hardware model field carries a trust level. Unverified fields are surfaced to generator and patcher as warnings.
3. **Failures are inputs** — DIAGNOSE turns a failed run into structured evidence before PATCH sees it, rather than giving PATCH only raw stderr.
4. **Review feedback loops** — The generator receives the reviewer's rejection issues as explicit instructions, not just the code it wrote before.
5. **Retry budget is honest** — Iteration only increments on PATCH cycles; review re-generation cycles don't silently consume the budget.
6. **Hard requirements** — The hardware model is mandatory. No silent fallback to encoded simulator notes.

## License

MIT License
