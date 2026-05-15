"""
PATCH node — error-driven code fix.

Given the full error context (build errors, run output, validation failure),
the patcher asks the LLM to produce corrected files.

Design pattern: Error as Context.
Rather than starting from scratch, the patcher sends the LLM:
  1. The original task
  2. The current code
  3. The exact error message
  4. All prior attempts and their outcomes
This gives the LLM the full diagnostic picture — the same information a
human engineer would need to fix the bug.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.state import AgentState
from r52_types import CodeGenerationOutput
from r52_types import AgentStatus
from agent.prompts.system_r52 import SYSTEM_R52
from backends.base import LLMBackend
from agent.nodes.scout import format_hardware_model


PATCHER_PROMPT = """
## Original Task
{task}

## Hardware Model
{hardware_map}

## Trust note
{trust_note}

## Current Code (files that exist on disk)
{current_files}

## What Failed
{failure_summary}

## Diagnosis
{diagnosis}

## All Previous Attempts
{history}

Fix the code to resolve the failure. Produce the complete corrected file set.

Rules:
- Only change what is needed to fix the failure — do not rewrite unrelated logic.
- The hardware model above is your only source of truth for addresses, sizes,
  and memory region boundaries. Do not substitute your own prior knowledge.
- If the diagnosis section above identifies a root cause, fix that specifically.
- Output ALL files (not just the changed ones) so the repo is in a consistent state.
"""


def run_patcher(state: AgentState, backend: LLMBackend) -> AgentState:
    """Produce corrected files based on the failure context."""
    # Snapshot the failed attempt into history before patching
    attempt = state.snapshot_attempt()
    attempt.patch_reasoning = _failure_summary(state)
    new_history = state.history + [attempt]

    # Check if we've hit the retry limit
    if state.iteration >= state.max_iterations:
        return state.model_copy(update={
            "history": new_history,
            "status": AgentStatus.MAX_RETRIES,
            "final_message": (
                f"Reached maximum iterations ({state.max_iterations}). "
                f"Last error: {_failure_summary(state)[:300]}"
            ),
        })

    failure_summary = _failure_summary(state)
    current_files = _format_files(state.generated_files)
    history_str = _format_history(new_history)

    hw_model = state.repo_context.get("hardware_model", {})
    if not hw_model or not hw_model.get("fields"):
        raise RuntimeError(
            "Hardware model is absent — cannot patch without verified hardware facts."
        )

    diagnosis = state.repo_context.get("diagnosis", "(no diagnosis available)")
    user_prompt = PATCHER_PROMPT.format(
        task=state.task,
        hardware_map=format_hardware_model(hw_model),
        trust_note=_trust_summary(hw_model),
        current_files=current_files,
        failure_summary=failure_summary,
        diagnosis=diagnosis,
        history=history_str,
    )

    output: CodeGenerationOutput = backend.complete_structured(
        system=SYSTEM_R52,
        user=user_prompt,
        response_model=CodeGenerationOutput,
        temperature=0.1,
    )

    # Write corrected files
    repo = Path(state.repo_path)
    generated: dict[str, str] = {}
    for gf in output.files:
        fp = repo / gf.path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(gf.content)
        generated[gf.path] = gf.content

    return state.model_copy(update={
        "history": new_history,
        "generated_files": generated,
        # Reset build/run/validation results for next cycle
        "build_result": None,
        "run_result": None,
        "validation_result": None,
    })


def should_retry(state: AgentState) -> bool:
    return state.iteration < state.max_iterations and state.status == AgentStatus.RUNNING


def _failure_summary(state: AgentState) -> str:
    parts = []
    if state.build_result and not state.build_result.success:
        parts.append(f"BUILD ERROR (rc={state.build_result.returncode}):\n"
                     f"{state.build_result.stderr[:1200]}")
    if state.run_result and not state.run_result.success:
        prefix = "RUN TIMEOUT" if state.run_result.timed_out else "RUN ERROR"
        parts.append(f"{prefix} (rc={state.run_result.returncode}):\n"
                     f"stdout: {state.run_result.stdout[:600]}\n"
                     f"stderr: {state.run_result.stderr[:600]}")
    if state.validation_result and not state.validation_result.passed:
        parts.append(f"VALIDATION FAILED:\n{state.validation_result.detail}")
    return "\n\n".join(parts) or "Unknown failure."


def _format_files(files: dict[str, str]) -> str:
    return "\n\n".join(f"### {p}\n```\n{c}\n```" for p, c in files.items())


def _format_history(history: list) -> str:
    if not history:
        return "None."
    lines = []
    for a in history[-5:]:
        lines.append(f"Attempt {a.iteration}: {a.patch_reasoning[:300] or 'no notes'}")
    return "\n".join(lines)


def _trust_summary(hw_model: dict) -> str:
    fields = hw_model.get("fields", {})
    if not fields:
        return ""
    prior = [k for k, v in fields.items() if v.get("trust") == "prior"]
    verified = len(fields) - len(prior)
    if not prior:
        return f"All {verified} hardware fields are verified from live probes or source."
    return (
        f"{verified}/{len(fields)} fields verified. "
        f"UNVERIFIED (LLM prior — suspect these if the bug relates to hardware values): "
        f"{', '.join(prior)}"
    )
