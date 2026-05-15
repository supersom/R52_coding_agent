"""
LangGraph StateGraph definition — the orchestration core.

Design pattern: Typed StateGraph with conditional edges.

Why StateGraph over ReAct/AgentExecutor:
  - Explicit, inspectable state transitions (not hidden in tool call loops)
  - Conditional edges give us clean build→run→validate→patch routing
  - Built-in checkpointing: any run can be resumed from any node
  - Human-in-loop: add interrupt_before=["BUILD"] to pause and show user

Node execution order:
  PLAN → GENERATE → REVIEW → BUILD → RUN → VALIDATE
                         ↑                      ↓
                         └──── PATCH ←──────────┘ (on failure)

Conditional edges:
  After REVIEW:    approved? → BUILD, else → GENERATE (re-generate with review feedback)
  After BUILD:     success? → RUN, else → PATCH
  After RUN:       success? → VALIDATE, else → PATCH
  After VALIDATE:  passed? → END(success), else → PATCH
  After PATCH:     retries_left? → GENERATE, else → END(failed)
"""

from __future__ import annotations

import time
from typing import Literal

from langgraph.graph import StateGraph, END

from agent.state import AgentState, AgentStatus, BuildResult, BuildSystem
from agent.nodes import (
    run_scout,
    run_planner, run_generator, run_reviewer, review_approved,
    run_builder, build_succeeded, run_runner, run_succeeded,
    run_validator, validation_passed, run_patcher, should_retry,
    run_diagnoser,
)
from backends.base import LLMBackend
from toolchain.config import ToolchainConfig
from context.repo_reader import read_repo_context, format_context_for_prompt
from observability.logger import RunLogger, NullLogger
from observability.tracer import AgentTracer
from observability.rich_ui import AgentUI


# ---------------------------------------------------------------------------
# Node wrappers — bind backend/config into each node and emit observability
# ---------------------------------------------------------------------------

def _make_nodes(
    backend: LLMBackend,
    config: ToolchainConfig,
    logger: RunLogger | NullLogger,
    tracer: AgentTracer,
    ui: AgentUI,
):
    """
    Return a dict of node functions with backend/config/logger bound in.
    Using closures here keeps the graph definition clean — each node function
    has exactly the signature StateGraph expects: (state) → state.
    """

    def node_scout(state) -> dict:
        state = AgentState.model_validate(state)
        t0 = time.monotonic()
        ui.update(phase="SCOUT", iteration=state.iteration)
        logger.node_start("scout", state.iteration)
        with tracer.span("scout", iteration=state.iteration) as sp:
            new_state = run_scout(state, backend)
            hw = new_state.repo_context.get("hardware_model", {})
            fields = hw.get("fields", {})
            verified = sum(1 for f in fields.values() if f.get("trust") != "prior")
            total = len(fields)
            sp.event("scout.result",
                     verified=str(verified), total=str(total),
                     machine=hw.get("machine", ""))
        dur = time.monotonic() - t0
        logger.node_end("scout", state.iteration, dur)
        logger.scout_result(state.iteration, total, verified, hw.get("machine", ""),
                            hw.get("fields", {}))
        probe_results = new_state.repo_context.get("scout_probe_results", [])
        if probe_results:
            logger.scout_probes(state.iteration, probe_results)
        ui.update(detail=f"Hardware model: {verified}/{total} fields verified from live probes or source")
        return new_state.model_dump()

    def node_plan(state) -> dict:
        state = AgentState.model_validate(state)
        t0 = time.monotonic()
        ui.update(phase="PLAN", iteration=state.iteration)
        logger.node_start("plan", state.iteration)
        try:
            with tracer.span("plan", iteration=state.iteration) as sp:
                new_state = run_planner(state, backend)
                plan = new_state.repo_context.get("plan", {})
                sp.event("plan.result",
                         build_system=new_state.build_system.value,
                         is_new_project=str(new_state.is_new_project),
                         files_to_create=str(plan.get("files_to_create", [])),
                         files_to_modify=str(plan.get("files_to_modify", [])),
                         implementation_steps=str(plan.get("implementation_steps", [])),
                         rationale=plan.get("rationale", ""))
            dur = time.monotonic() - t0
            logger.node_end("plan", state.iteration, dur)
            logger.plan_result(
                state.iteration,
                build_system=new_state.build_system.value,
                is_new_project=new_state.is_new_project,
                files_to_create=plan.get("files_to_create", []),
                files_to_modify=plan.get("files_to_modify", []),
                implementation_steps=plan.get("implementation_steps", []),
                rationale=plan.get("rationale", ""),
            )
            ui.update(detail=f"Plan complete. Build system: {new_state.build_system.value}")
            return new_state.model_dump()
        except Exception as exc:
            dur = time.monotonic() - t0
            logger.node_end("plan", state.iteration, dur)
            err_msg = f"Backend plan failed: {exc}"
            ui.update(detail=err_msg)
            # Read repo context directly so downstream nodes have it despite the LLM failure.
            ctx = read_repo_context(state.repo_path)
            is_new = ctx.get("total_files", 0) == 0
            default_plan = {
                "files_to_create": [], "files_to_modify": [],
                "new_symbols": [], "build_system_notes": "",
                "implementation_steps": ["Implement the task as described."],
                "startup_changes_needed": False, "linker_changes_needed": False,
                "rationale": "Plan generation failed; proceeding with minimal plan.",
            }
            ctx["plan"] = default_plan
            ctx["formatted_context"] = format_context_for_prompt(ctx)
            new_state = state.model_copy(update={
                "repo_context": ctx,
                "build_system": BuildSystem.MAKE,
                "is_new_project": is_new,
            })
            return new_state.model_dump()

    def node_generate(state) -> dict:
        state = AgentState.model_validate(state)
        t0 = time.monotonic()
        ui.update(phase="GENERATE", iteration=state.iteration)
        logger.node_start("generate", state.iteration)
        try:
            with tracer.span("generate", iteration=state.iteration) as sp:
                new_state = run_generator(state, backend)
                # Record every generated file so you can read them in Jaeger
                sp.generated_files(new_state.generated_files)
                sp.set("file_count", len(new_state.generated_files))
            dur = time.monotonic() - t0
            logger.node_end("generate", state.iteration, dur)
            ui.update(detail=f"Generated {len(new_state.generated_files)} file(s).")
            return new_state.model_dump()
        except Exception as exc:
            dur = time.monotonic() - t0
            logger.node_end("generate", state.iteration, dur)
            err_msg = f"Backend generation failed: {exc}"
            ui.update(detail=err_msg)
            # Surface as a synthetic build failure so the graph routes to
            # BUILD (fails on missing files) → PATCH → GENERATE (retry),
            # consuming the retry budget rather than crashing the agent.
            synthetic_failure = BuildResult(
                success=False,
                command="(llm generation)",
                stdout="",
                stderr=err_msg,
                returncode=1,
                duration_s=dur,
            )
            new_state = state.model_copy(update={
                "generated_files": {},
                "build_result": synthetic_failure,
                "iteration": state.iteration + 1,
            })
            return new_state.model_dump()

    def node_review(state) -> dict:
        state = AgentState.model_validate(state)
        t0 = time.monotonic()
        ui.update(phase="REVIEW", iteration=state.iteration)
        logger.node_start("review", state.iteration)
        try:
            with tracer.span("review", iteration=state.iteration) as sp:
                new_state = run_reviewer(state, backend)
                review = new_state.repo_context.get("review", {})
                approved = review.get("approved", True)
                issues = review.get("issues", [])
                sp.set("approved", approved)
                sp.event("review.result",
                         approved=str(approved),
                         severity=review.get("severity", "none"),
                         issues=str(issues),
                         corrections=str(list(review.get("corrected_files", {}).keys())))
            dur = time.monotonic() - t0
            logger.node_end("review", state.iteration, dur)
            logger.review_result(
                state.iteration, approved, issues,
                review.get("rejection_count", 0),
            )
            ui.update(detail=f"Review: {'approved' if approved else 'issues found: ' + str(issues[:2])}")
            return new_state.model_dump()
        except Exception as exc:
            dur = time.monotonic() - t0
            logger.node_end("review", state.iteration, dur)
            err_msg = f"Backend review failed: {exc}"
            ui.update(detail=err_msg)
            # Treat as approved so the graph proceeds to BUILD naturally.
            # BUILD will fail if files are missing/broken and route to PATCH.
            synthetic_review = {"approved": True, "issues": [], "severity": "none", "corrected_files": {}}
            new_repo_context = {**state.repo_context, "review": synthetic_review}
            new_state = state.model_copy(update={"repo_context": new_repo_context})
            return new_state.model_dump()

    def node_build(state) -> dict:
        state = AgentState.model_validate(state)
        t0 = time.monotonic()
        ui.update(phase="BUILD", iteration=state.iteration)
        logger.node_start("build", state.iteration)
        with tracer.span("build", iteration=state.iteration) as sp:
            new_state = run_builder(state, config)
            br = new_state.build_result
            if br:
                sp.set("success", br.success)
                sp.build_result(br.command, br.stdout, br.stderr,
                                br.success, br.duration_s)
        dur = time.monotonic() - t0
        br = new_state.build_result
        logger.node_end("build", state.iteration, dur)
        logger.build_result(
            state.iteration,
            br.success if br else False,
            br.duration_s if br else 0,
            br.stderr[:500] if br else "",
        )
        if br:
            ui.update(detail=(
                f"Build {'OK' if br.success else 'FAILED'} in {br.duration_s:.1f}s\n"
                + (br.stderr[:300] if not br.success else "")
            ))
        return new_state.model_dump()

    def node_run(state) -> dict:
        state = AgentState.model_validate(state)
        t0 = time.monotonic()
        ui.update(phase="RUN", iteration=state.iteration)
        logger.node_start("run", state.iteration)
        with tracer.span("run", iteration=state.iteration,
                         simulator=state.simulator.value) as sp:
            new_state = run_runner(state, config)
            rr = new_state.run_result
            if rr:
                sp.set("success", rr.success)
                sp.run_result(rr.stdout, rr.stderr, rr.success,
                              rr.timed_out, rr.duration_s)
        dur = time.monotonic() - t0
        rr = new_state.run_result
        logger.node_end("run", state.iteration, dur)
        logger.run_result(
            state.iteration,
            rr.success if rr else False,
            rr.timed_out if rr else False,
            rr.duration_s if rr else 0,
            rr.stdout[:500] if rr else "",
        )
        if rr:
            ui.update(detail=(
                f"Run {'OK' if rr.success else 'TIMED OUT' if rr.timed_out else 'FAILED'}"
                f" in {rr.duration_s:.1f}s\n"
                + rr.stdout[:300]
            ))
        return new_state.model_dump()

    def node_validate(state) -> dict:
        state = AgentState.model_validate(state)
        ui.update(phase="VALIDATE", iteration=state.iteration)
        with tracer.span("validate", iteration=state.iteration) as sp:
            new_state = run_validator(state)
            vr = new_state.validation_result
            if vr:
                sp.set("passed", vr.passed)
                sp.validation_result(vr.passed, vr.expected, vr.actual, vr.detail)
        if vr:
            logger.validation_result(state.iteration, vr.passed, vr.detail)
            ui.update(detail=f"Validation: {'PASSED' if vr.passed else 'FAILED'}\n{vr.detail[:200]}")
        return new_state.model_dump()

    def node_diagnose(state) -> dict:
        state = AgentState.model_validate(state)
        t0 = time.monotonic()
        ui.update(phase="DIAGNOSE", iteration=state.iteration)
        logger.node_start("diagnose", state.iteration)
        with tracer.span("diagnose", iteration=state.iteration) as sp:
            new_state = run_diagnoser(state, backend, config)
            diagnosis = new_state.repo_context.get("diagnosis", "")
            sp.event("diagnose.result", summary=diagnosis[:200])
        dur = time.monotonic() - t0
        logger.node_end("diagnose", state.iteration, dur)
        logger.diagnosis_result(state.iteration, diagnosis)
        diagnose_probes = new_state.repo_context.get("diagnose_probe_results", [])
        if diagnose_probes:
            logger.diagnose_probes(state.iteration, diagnose_probes)
        ui.update(detail=f"Diagnosis: {diagnosis[:150]}")
        return new_state.model_dump()

    def node_patch(state) -> dict:
        state = AgentState.model_validate(state)
        t0 = time.monotonic()
        ui.update(phase="PATCH", iteration=state.iteration)
        logger.node_start("patch", state.iteration)
        try:
            with tracer.span("patch", iteration=state.iteration) as sp:
                new_state = run_patcher(state, backend)
                sp.generated_files(new_state.generated_files)
                sp.set("file_count", len(new_state.generated_files))
            dur = time.monotonic() - t0
            logger.node_end("patch", state.iteration, dur)
            ui.update(detail=f"Patch generated. Iteration {new_state.iteration}/{new_state.max_iterations}")
            return new_state.model_dump()
        except Exception as exc:
            dur = time.monotonic() - t0
            logger.node_end("patch", state.iteration, dur)
            err_msg = f"Backend patch failed: {exc}"
            ui.update(detail=err_msg)
            # Increment iteration to consume retry budget; keep generated_files
            # empty so the next GENERATE starts fresh.
            new_state = state.model_copy(update={
                "generated_files": {},
                "iteration": state.iteration + 1,
            })
            return new_state.model_dump()

    return {
        "scout":    node_scout,
        "plan":     node_plan,
        "generate": node_generate,
        "review":   node_review,
        "build":    node_build,
        "run":      node_run,
        "diagnose": node_diagnose,
        "validate": node_validate,
        "patch":    node_patch,
    }


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def _coerce(state) -> AgentState:
    """LangGraph may pass either a dict or an AgentState — normalise to AgentState."""
    if isinstance(state, AgentState):
        return state
    return AgentState.model_validate(state)


_MAX_REVIEW_REJECTIONS = 3


def _after_review(state) -> Literal["build", "generate"]:
    s = _coerce(state)
    # Count consecutive rejections via the review context
    review = s.repo_context.get("review", {})
    rejections = review.get("rejection_count", 0)
    if review_approved(s):
        return "build"
    # After too many rejections, force to BUILD so errors surface concretely.
    if rejections >= _MAX_REVIEW_REJECTIONS:
        return "build"
    return "generate"


def _after_build(state) -> Literal["run", "patch"]:
    return "run" if build_succeeded(_coerce(state)) else "patch"


def _after_run(state) -> Literal["validate", "diagnose"]:
    return "validate" if run_succeeded(_coerce(state)) else "diagnose"


def _after_validate(state) -> Literal["__end__", "patch"]:
    return "__end__" if validation_passed(_coerce(state)) else "patch"


def _after_patch(state) -> Literal["generate", "__end__"]:
    s = _coerce(state)
    if s.status == AgentStatus.MAX_RETRIES:
        return "__end__"
    return "generate" if should_retry(s) else "__end__"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    backend: LLMBackend,
    config: ToolchainConfig,
    logger: RunLogger | NullLogger,
    tracer: AgentTracer,
    ui: AgentUI,
):
    """
    Construct and compile the LangGraph StateGraph.

    The graph is compiled once per run (not per node invocation).
    Compilation validates the graph structure (no dead ends, valid edge targets).
    """
    nodes = _make_nodes(backend, config, logger, tracer, ui)

    # StateGraph takes a state schema; it type-checks state transitions.
    # We use dict as the channel type since Pydantic model_dump/model_validate
    # handles serialisation at each node boundary.
    g = StateGraph(dict)

    # Register nodes
    for name, fn in nodes.items():
        g.add_node(name, fn)

    # Entry point
    g.set_entry_point("plan")

    # Linear edges
    g.add_edge("plan", "scout")
    g.add_edge("scout", "generate")
    g.add_edge("generate", "review")

    # Conditional edges
    g.add_conditional_edges("review",   _after_review,   {"build": "build", "generate": "generate"})
    g.add_conditional_edges("build",    _after_build,    {"run": "run", "patch": "patch"})
    g.add_conditional_edges("run",      _after_run,      {"validate": "validate", "diagnose": "diagnose"})
    g.add_edge("diagnose", "patch")
    g.add_conditional_edges("validate", _after_validate, {"__end__": END, "patch": "patch"})
    g.add_conditional_edges("patch",    _after_patch,    {"generate": "generate", "__end__": END})

    return g.compile()


# ---------------------------------------------------------------------------
# High-level run function
# ---------------------------------------------------------------------------

def run_agent(
    initial_state: AgentState,
    backend: LLMBackend,
    config: ToolchainConfig,
    logger: RunLogger | NullLogger,
    tracer: AgentTracer,
    ui: AgentUI,
) -> AgentState:
    """
    Run the full agent graph and return the final state.
    """
    graph = build_graph(backend, config, logger, tracer, ui)

    t0 = time.monotonic()
    logger.run_start(
        task=initial_state.task,
        repo=initial_state.repo_path,
        backend=backend.name,
        model=backend.model,
    )

    final_dict = graph.invoke(initial_state.model_dump())
    final = AgentState(**final_dict)

    total = time.monotonic() - t0
    logger.run_end(final.status.value, final.iteration, total)
    tracer.shutdown()

    return final
