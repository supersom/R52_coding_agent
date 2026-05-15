"""
REVIEW node — self-critique pass before attempting to build.

Design pattern: Reflection / Self-Critique.
A separate LLM call reads the generated code and checks for issues before
we spend time on a build+run cycle. This is cheaper than discovering
obvious errors at build time and improves overall success rate.

The reviewer can:
  - Approve the code (proceed to build)
  - Request fixes (back to generate with its critique embedded)

Research background: 'Reflexion' (Shinn et al., 2023) showed that
giving LLMs a verbal self-reflection step before retrying consistently
improves success rates on code generation benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from agent.state import AgentState
from agent.prompts.system_r52 import SYSTEM_R52
from agent.nodes.scout import format_hardware_model
from backends.base import LLMBackend


class ReviewResult(BaseModel):
    approved: bool
    issues: list[str]
    corrected_files: dict[str, str] = {}
    severity: str = "none"  # "none" | "warning" | "error"


REVIEWER_PROMPT = """
## Task
{task}

## Verified Hardware Model
{hardware_map}

## Generated Files
{files}

Reason about whether this implementation will actually work on this platform.

Trace the execution from reset: what does the CPU do first, what state is it
in at each step, and does it reach the intended behaviour? Think about what
could prevent the task from being accomplished before the code even gets to
the interesting part.

Then consider the hardware model above as ground truth. Does the code make
assumptions about the platform that conflict with it? An assumption doesn't
have to be explicit — a hardcoded address, a size calculation, or an offset
that silently disagrees with the model is just as dangerous.

The goal is to catch anything that would cause a silent failure: a run that
produces no output, wrong output, or a hang — where the mismatch isn't obvious
from the code alone but becomes clear when you compare it against what the
hardware actually looks like.

If you find fixable issues, correct them inline and set approved=True with
corrected_files. If re-generation is needed, set approved=False and explain
what specifically would prevent the task from being accomplished.
"""


def run_reviewer(state: AgentState, backend: LLMBackend) -> AgentState:
    """
    Review generated files against the verified hardware model.
    If reviewer corrects issues inline, apply those corrections.
    Returns state with possibly-corrected files and a review note in repo_context.
    """
    files_str = _format_files(state.generated_files)
    hw_map = format_hardware_model(state.repo_context.get("hardware_model", {}))

    user_prompt = REVIEWER_PROMPT.format(
        task=state.task,
        hardware_map=hw_map,
        files=files_str,
    )

    result: ReviewResult = backend.complete_structured(
        system=SYSTEM_R52,
        user=user_prompt,
        response_model=ReviewResult,
        temperature=0.0,
    )

    review_data = result.model_dump()
    # Track consecutive rejections so the graph can break infinite review loops.
    prev_rejections = state.repo_context.get("review", {}).get("rejection_count", 0)
    review_data["rejection_count"] = 0 if result.approved else prev_rejections + 1
    state.repo_context["review"] = review_data

    if result.corrected_files:
        # Apply inline corrections to disk and state
        repo = Path(state.repo_path)
        updated = dict(state.generated_files)
        for path, content in result.corrected_files.items():
            (repo / path).write_text(content)
            updated[path] = content
        state = state.model_copy(update={"generated_files": updated})

    return state


def review_approved(state: AgentState) -> bool:
    """Edge condition: did the reviewer approve?"""
    review = state.repo_context.get("review", {})
    return review.get("approved", True)


def _format_files(files: dict[str, str]) -> str:
    parts = []
    for path, content in files.items():
        parts.append(f"### {path}\n```\n{content}\n```")
    return "\n\n".join(parts)
