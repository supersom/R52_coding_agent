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
from backends.base import LLMBackend


class ReviewResult(BaseModel):
    approved: bool
    issues: list[str]
    corrected_files: dict[str, str] = {}   # path → corrected content (if approved=False and fixable inline)
    severity: str = "none"                 # "none" | "warning" | "error"


REVIEWER_PROMPT = """
## Task
{task}

## Generated Files
{files}

Review the generated ARM Cortex-R52 bare-metal code for:

1. **Correctness**: Does it implement what the task asks?
2. **ARM R52 idioms**: Correct register access, memory barriers, AAPCS compliance?
3. **Startup/linker compatibility**: Will it link correctly with existing startup.s / link.ld?
4. **Build system**: Is the Makefile/CMakeLists.txt correct for arm-none-eabi-gcc?
5. **Common bugs**: Uninitialized stack, missing .bss zeroing, non-volatile hardware pointers?

If you find issues that you can fix inline, correct them and set approved=True with corrected_files.
If the issues require re-generation, set approved=False and list the issues.
If code looks good, set approved=True with empty issues list.
"""


def run_reviewer(state: AgentState, backend: LLMBackend) -> AgentState:
    """
    Review generated files. If reviewer corrects them inline, apply those corrections.
    Returns state with possibly-corrected files and a review note in repo_context.
    """
    files_str = _format_files(state.generated_files)

    user_prompt = REVIEWER_PROMPT.format(
        task=state.task,
        files=files_str,
    )

    result: ReviewResult = backend.complete_structured(
        system=SYSTEM_R52,
        user=user_prompt,
        response_model=ReviewResult,
        temperature=0.0,
    )

    state.repo_context["review"] = result.model_dump()

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
