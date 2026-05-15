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
from agent.nodes.scout import format_hardware_model


GENERATOR_PROMPT = """
## Task
{task}

## Implementation Plan
{plan}

## Hardware Model
{hardware_map}

## Trust note
{trust_note}

## Current Repository
{repo_context}

## Review feedback (fix these specific issues before proceeding)
{review_feedback}

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
- The hardware model above is your only source of truth for addresses, sizes,
  and memory region boundaries. Do not substitute your own prior knowledge.
- If review feedback is present above, address every listed issue explicitly.
"""


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

    hw_model = state.repo_context.get("hardware_model", {})
    if not hw_model or not hw_model.get("fields"):
        raise RuntimeError(
            "Hardware model is absent — SCOUT must run and produce a model "
            "before code can be generated. Cannot proceed without verified hardware facts."
        )

    hw_map_str = format_hardware_model(hw_model)
    trust_note = _trust_summary(hw_model)
    review_feedback = _format_review_feedback(state.repo_context.get("review", {}))

    user_prompt = GENERATOR_PROMPT.format(
        task=state.task,
        plan=json.dumps(plan, indent=2),
        hardware_map=hw_map_str,
        trust_note=trust_note,
        repo_context=repo_context_str,
        review_feedback=review_feedback,
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

    # Only count iterations when coming from PATCH (state.history is non-empty).
    # Review re-generation cycles don't consume retry budget.
    new_iteration = state.iteration + 1 if state.history else state.iteration
    return state.model_copy(update={
        "generated_files": generated,
        "iteration": new_iteration,
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


def _format_review_feedback(review: dict) -> str:
    if not review or review.get("approved", True):
        return "None — proceeding with initial generation."
    issues = review.get("issues", [])
    if not issues:
        return "Reviewer rejected but provided no specific issues."
    lines = ["The previous generation was REJECTED. Fix ALL of these issues:"]
    for i, issue in enumerate(issues, 1):
        lines.append(f"  {i}. {issue}")
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
        f"{verified}/{len(fields)} fields verified from live probes or source. "
        f"The following fields are UNVERIFIED (LLM prior — scrutinise carefully): "
        f"{', '.join(prior)}"
    )
