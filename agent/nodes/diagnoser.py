"""
DIAGNOSE node — LLM-directed failure analysis.

Sits between a failed RUN and the PATCH node. Follows the same pattern as SCOUT:
  Phase 1 — LLM reads the failure symptom and decides what evidence to gather.
  Phase 2 — Python executes the investigation commands deterministically.
  Phase 3 — LLM reads raw evidence and produces a structured diagnosis.

No failure modes are pre-encoded here. The LLM uses its knowledge of embedded
systems debugging to decide what to collect and how to interpret it. This handles
failure classes we haven't seen yet, not just the ones we have.

The hardware model is already verified by the time DIAGNOSE runs (SCOUT ran,
GENERATE and REVIEW used it). It is passed as ground truth for cross-referencing
against what the code actually does.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.probe_tools import run_command, read_file
from agent.state import AgentState
from agent.nodes.scout import format_hardware_model
from agent.prompts.system_r52 import SYSTEM_R52
from backends.base import LLMBackend
from toolchain.config import ToolchainConfig
from toolchain.simulator import find_elf


# ---------------------------------------------------------------------------
# Phase 1 — investigation plan
# ---------------------------------------------------------------------------

class DiagnosticProbe(BaseModel):
    label: str = Field(description="Short identifier for this probe.")
    purpose: str = Field(description="What diagnostic information this is expected to yield.")
    tool: Literal["run_command", "read_file"]
    command: str | None = Field(
        default=None,
        description="Shell command for run_command probes."
    )
    path: str | None = Field(
        default=None,
        description="File path for read_file probes."
    )


class DiagnosticPlan(BaseModel):
    reasoning: str = Field(
        description="Based on the symptom, what do you suspect and what evidence "
                    "would confirm or rule it out?"
    )
    probes: list[DiagnosticProbe]


# ---------------------------------------------------------------------------
# Phase 3 — diagnosis
# ---------------------------------------------------------------------------

class DiagnosisResult(BaseModel):
    failure_class: str = Field(
        description="Short tag, e.g. 'crash_before_uart', 'wrong_address', "
                    "'build_error', 'wrong_output', 'infinite_loop'."
    )
    root_cause: str = Field(description="Plain English explanation of what went wrong.")
    evidence: str = Field(description="Specific probe output lines that support this conclusion.")
    fix_hint: str = Field(description="What specifically needs to change.")
    confidence: str = Field(description="'high', 'medium', or 'low'.")


def _execute_probe(probe: DiagnosticProbe) -> tuple[str, bool]:
    if probe.tool == "run_command" and probe.command:
        return run_command(probe.command)
    if probe.tool == "read_file" and probe.path:
        return read_file(probe.path)
    return "(probe misconfigured)", False


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_PHASE1_SYSTEM = """
You are an embedded systems debugging specialist.

A bare-metal firmware run has failed. You will be given:
- The failure symptom (what was observed)
- The verified hardware model (ground truth for this platform)
- A summary of what the firmware is supposed to do
- The path to the compiled ELF (if available)

Your job: decide what evidence to gather to diagnose the root cause.

You have two tools:

  run_command(command)
    Execute any read-only shell command and capture output.
    Use your knowledge of embedded debugging tools to choose commands.
    For example: disassembling the ELF, running the simulator with debug flags,
    examining build artifacts, querying tool versions, etc.

  read_file(path)
    Read a file and return its contents.
    Use for generated source files, linker scripts, Makefiles, or any artifact
    that might reveal a mismatch with the hardware model.

Think about what symptom you're seeing and what evidence would most quickly
confirm or rule out the most likely causes. Prefer probes that give the most
diagnostic information for the least cost.
"""

_PHASE1_USER = """
## Failure symptom
{symptom}

## Verified hardware model (ground truth)
{hw_map}

## Firmware task
{task}

## Compiled ELF path (if available)
{elf_path}

## Simulator
{simulator}

Produce a DiagnosticPlan: what probes would you run to understand this failure?
"""

_PHASE3_SYSTEM = """
You are diagnosing a bare-metal firmware failure from probe evidence.

Read the probe outputs carefully. Cross-reference:
- What the code actually does (from disassembly or source)
- What the hardware model says the platform looks like
- What the simulator trace shows actually happened

Identify the specific mismatch or error that caused the failure.
If the evidence is ambiguous, say so and set confidence to 'low' or 'medium'.
"""

_PHASE3_USER = """
## Failure symptom
{symptom}

## Verified hardware model
{hw_map}

## Probe results
{probe_results}

Produce a DiagnosisResult.
"""


def _format_symptom(state: AgentState) -> str:
    parts = []
    if state.build_result and not state.build_result.success:
        parts.append(f"BUILD FAILED:\n{state.build_result.stderr[:800]}")
    if state.run_result:
        rr = state.run_result
        if rr.timed_out and not rr.stdout.strip():
            parts.append(
                "RUN: timed out with ZERO stdout.\n"
                "The firmware ran for the full timeout but produced no output.\n"
                "Possible causes include a crash before any peripheral is initialised, "
                "an infinite loop entered before reaching the intended code path, or "
                "the output peripheral not being connected to the captured stream."
            )
        elif rr.timed_out:
            parts.append(f"RUN: timed out (may be normal for looping firmware).\nstdout: {rr.stdout[:400]}")
        elif not rr.success:
            parts.append(f"RUN FAILED (rc={rr.returncode}):\nstdout: {rr.stdout[:400]}\nstderr: {rr.stderr[:400]}")
    if state.validation_result and not state.validation_result.passed:
        parts.append(f"VALIDATION FAILED:\n{state.validation_result.detail}")
    return "\n\n".join(parts) or "Unknown failure."


def _format_probe_results(results: list[dict]) -> str:
    parts = []
    for r in results:
        status = "OK" if r["success"] else "FAILED"
        parts.append(
            f"=== {r['label']} [{r['tool']}] [{status}] ===\n"
            f"Purpose: {r['purpose']}\n"
            f"{r['output']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_diagnoser(state: AgentState, backend: LLMBackend, config: ToolchainConfig) -> AgentState:
    """
    Phase 1: LLM decides what to probe given the failure symptom.
    Phase 2: Execute probes deterministically.
    Phase 3: LLM synthesises evidence into a structured diagnosis.

    Stores diagnosis in state.repo_context['diagnosis'] for PATCH to use.
    Non-fatal: if diagnosis fails, PATCH proceeds without it.
    """
    elf = find_elf(state.repo_path) or "(not found)"
    symptom = _format_symptom(state)
    hw_map = format_hardware_model(state.repo_context.get("hardware_model", {}))

    try:
        # Phase 1 — decide what to investigate
        plan: DiagnosticPlan = backend.complete_structured(
            system=_PHASE1_SYSTEM,
            user=_PHASE1_USER.format(
                symptom=symptom,
                hw_map=hw_map,
                task=state.task,
                elf_path=elf,
                simulator=state.simulator.value,
            ),
            response_model=DiagnosticPlan,
            temperature=0.0,
        )

        # Phase 2 — execute probes
        probe_results: list[dict] = []
        for probe in plan.probes:
            output, success = _execute_probe(probe)
            probe_results.append({
                "label": probe.label,
                "tool": probe.tool,
                "purpose": probe.purpose,
                "output": output[:8000],
                "success": success,
            })

        # Phase 3 — synthesise diagnosis
        result: DiagnosisResult = backend.complete_structured(
            system=_PHASE3_SYSTEM,
            user=_PHASE3_USER.format(
                symptom=symptom,
                hw_map=hw_map,
                probe_results=_format_probe_results(probe_results),
            ),
            response_model=DiagnosisResult,
            temperature=0.0,
        )

        diagnosis_text = (
            f"[{result.failure_class}] confidence={result.confidence}\n"
            f"Root cause: {result.root_cause}\n"
            f"Evidence: {result.evidence}\n"
            f"Fix: {result.fix_hint}"
        )

    except Exception as exc:
        diagnosis_text = f"(diagnosis failed: {exc})"

    new_ctx = {**state.repo_context, "diagnosis": diagnosis_text}
    return state.model_copy(update={"repo_context": new_ctx})
