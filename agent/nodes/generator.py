"""
GENERATE node — produces ARM R52 source files.

Design pattern: Structured Output via instructor.
The LLM is constrained to return a CodeGenerationOutput Pydantic model
rather than free-form text. This eliminates post-processing ambiguity.

For API backends: instructor enforces the schema via tool-use, retrying
on validation failure automatically.
For CLI backends: ResponseParser extracts code from fenced blocks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from agent.state import AgentState
from r52_types import CodeGenerationOutput
from agent.prompts.system_r52 import SYSTEM_R52
from backends.base import LLMBackend
from context.repo_reader import read_repo_context, format_context_for_prompt


GENERATOR_PROMPT = """
## Task
{task}

## Implementation Plan
{plan}

## Target Simulator
{simulator_note}

## Current Repository
{repo_context}

## Previous Attempts (if any)
{history}

Generate all files needed to implement this feature on ARM Cortex-R52.

Rules:
- Every file must be complete and compilable — no placeholders, no TODOs.
- Use C99 and ARM AAPCS ABI conventions.
- Include appropriate memory barriers (DSB, DMB, ISB) for hardware access.
- Mark hardware register pointers as `volatile`.
- For each file, include its full path relative to the repo root.
- If creating a new project, generate: startup.s, link.ld, main.c, Makefile.
- Do NOT modify startup.s or link.ld unless the plan explicitly requires it.
- Use the correct UART address and memory map for the Target Simulator above.
"""

_SIMULATOR_NOTES = {
    "fvp": (
        "FVP_BaseR_Cortex-R52 — true Cortex-R52 simulation.\n"
        "UART: 0x1C090000 (PL011). RAM starts at 0x00000000. "
        "Use semihosting (SVC #0x123456 / HLT 0xF000) for debug output."
    ),
    "qemu": (
        "QEMU versatilepb (qemu-system-arm -M versatilepb -m 128M).\n"
        "UART: 0x101F1000 — write bytes directly to this address for output. "
        "RAM: 128MB at 0x00000000, load code at 0x10000. "
        "No semihosting output — use UART writes only. "
        "For program exit use semihosting SYS_EXIT: r0=0x18, r1=0, svc #0x123456."
    ),
}


def run_generator(state: AgentState, backend: LLMBackend) -> AgentState:
    """Generate code files and write them to the repo."""
    plan = state.repo_context.get("plan", {})
    history_summary = _format_history(state)

    # On retry cycles the repo has been modified by PATCH — re-read from disk
    # so the generator sees the corrected files rather than the stale planner snapshot.
    # On the first run state.history is empty, so we use the planner's cached scan.
    if state.history:
        repo_context_str = format_context_for_prompt(read_repo_context(state.repo_path))
    else:
        repo_context_str = state.repo_context.get("formatted_context", "")

    simulator_note = _SIMULATOR_NOTES.get(state.simulator.value, _SIMULATOR_NOTES["fvp"])
    user_prompt = GENERATOR_PROMPT.format(
        task=state.task,
        plan=json.dumps(plan, indent=2),
        simulator_note=simulator_note,
        repo_context=repo_context_str,
        history=history_summary,
    )

    output: CodeGenerationOutput = backend.complete_structured(
        system=SYSTEM_R52,
        user=user_prompt,
        response_model=CodeGenerationOutput,
        temperature=0.1,   # low temperature for code — determinism over creativity
    )

    # Write generated files to disk
    repo = Path(state.repo_path)
    generated: dict[str, str] = {}

    for gf in output.files:
        file_path = repo / gf.path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(gf.content)
        generated[gf.path] = gf.content

    return state.model_copy(update={
        "generated_files": generated,
        "iteration": state.iteration + 1,
    })


def _format_history(state: AgentState) -> str:
    if not state.history:
        return "None — this is the first attempt."

    lines = []
    for attempt in state.history[-3:]:   # last 3 attempts to keep context bounded
        lines.append(f"### Attempt {attempt.iteration}")
        if attempt.build_result and not attempt.build_result.success:
            lines.append(f"BUILD FAILED:\n{attempt.build_result.stderr[:800]}")
        elif attempt.run_result and not attempt.run_result.success:
            lines.append(f"RUN FAILED (timeout={attempt.run_result.timed_out}):\n"
                         f"stdout: {attempt.run_result.stdout[:400]}\n"
                         f"stderr: {attempt.run_result.stderr[:400]}")
        elif attempt.validation_result and not attempt.validation_result.passed:
            lines.append(f"VALIDATION FAILED: {attempt.validation_result.detail}")
        lines.append("")

    return "\n".join(lines)
