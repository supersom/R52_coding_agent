"""
PLAN node — analyses the task and existing codebase.

Design pattern: Plan-Execute separation.
The planner's job is to understand context before generating any code.
It produces a structured plan that the generator uses as its brief.

Keeping planning separate from generation means:
  - The generator has focused context (the plan) rather than raw repo files
  - We can inspect/log the plan independently for debugging
  - The plan can be shown to the user for approval (human-in-loop)
"""

from __future__ import annotations

from pydantic import BaseModel

from agent.state import AgentState, BuildSystem
from agent.prompts.system_r52 import SYSTEM_R52
from context.repo_reader import read_repo_context, format_context_for_prompt
from backends.base import LLMBackend


class TaskPlan(BaseModel):
    """Structured plan output from the planner LLM call."""
    files_to_create: list[str]
    files_to_modify: list[str]
    new_symbols: list[str]          # function/variable names to be added
    build_system_notes: str
    implementation_steps: list[str]
    startup_changes_needed: bool
    linker_changes_needed: bool
    rationale: str


PLANNER_PROMPT = """
You are planning the implementation of a feature for an ARM Cortex-R52 bare-metal project.

## Task
{task}

## Repository Context
{repo_context}

Analyse the task and the existing codebase, then produce a structured implementation plan.

Your plan must:
1. List exactly which files need to be created and which need to be modified.
2. List new function/variable names that will be introduced.
3. Note any required changes to startup.s or linker scripts (only if truly necessary).
4. Break the implementation into numbered steps.
5. Justify your approach given the existing code structure.

Be specific and concrete. The code generator will use your plan as its brief.
"""


def run_planner(state: AgentState, backend: LLMBackend) -> AgentState:
    """
    Read the repo, build context, call the LLM for a plan.
    Returns updated state with repo_context, build_system, and is_new_project set.
    """
    # Read repo context
    ctx = read_repo_context(state.repo_path)
    formatted_ctx = format_context_for_prompt(ctx)

    # Detect build system and whether this is a new (empty) project
    is_new = ctx.get("total_files", 0) == 0
    build_sys_str = ctx.get("build_system", "none")
    build_system = {
        "cmake": BuildSystem.CMAKE,
        "make": BuildSystem.MAKE,
    }.get(build_sys_str, BuildSystem.MAKE if not is_new else BuildSystem.MAKE)

    # Store context in state
    state = state.model_copy(update={
        "repo_context": ctx,
        "build_system": build_system,
        "is_new_project": is_new,
    })

    user_prompt = PLANNER_PROMPT.format(
        task=state.task,
        repo_context=formatted_ctx,
    )

    plan: TaskPlan = backend.complete_structured(
        system=SYSTEM_R52,
        user=user_prompt,
        response_model=TaskPlan,
    )

    # Embed plan into state's repo_context for downstream nodes
    state.repo_context["plan"] = plan.model_dump()
    state.repo_context["formatted_context"] = formatted_ctx

    return state
