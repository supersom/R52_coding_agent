"""
AgentState — the single source of truth threaded through every LangGraph node.

All enum/result types are imported from r52_types (root level) to avoid
circular imports with the toolchain package.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# Shared types (no circular dependency risk)
from r52_types import (
    AgentStatus, MatchMode, Toolchain, BuildSystem, Simulator,
    BuildResult, RunResult, ValidationResult, CodeGenerationOutput, GeneratedFile,
)

# Re-export for convenience so other modules can import from agent.state
__all__ = [
    "AgentState", "Attempt",
    "AgentStatus", "MatchMode", "Toolchain", "BuildSystem", "Simulator",
    "BuildResult", "RunResult", "ValidationResult", "CodeGenerationOutput", "GeneratedFile",
]


class Attempt(BaseModel):
    """Record of one generate→build→run→validate cycle, kept in history for the patcher."""
    iteration: int
    generated_files: dict[str, str] = Field(default_factory=dict)
    build_result: BuildResult | None = None
    run_result: RunResult | None = None
    validation_result: ValidationResult | None = None
    patch_reasoning: str = ""


class AgentState(BaseModel):
    """
    Single source of truth threaded through every LangGraph node.

    Design: Pydantic v2 BaseModel. Each node receives the full state,
    returns model_copy(update={...}) with only changed fields.
    LangGraph serialises via model_dump() / model_validate().
    """

    # ---- identity ----
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # ---- task input ----
    task: str = ""
    repo_path: str = ""
    expected_output: str | None = None
    match_mode: MatchMode = MatchMode.CONTAINS

    # ---- configuration ----
    toolchain: Toolchain = Toolchain.GNU
    simulator: Simulator = Simulator.FVP
    max_iterations: int = 10
    simulator_timeout: int = 600

    # ---- repo context (populated by planner) ----
    repo_context: dict[str, Any] = Field(default_factory=dict)
    build_system: BuildSystem = BuildSystem.MAKE
    is_new_project: bool = False
    template_used: str | None = None

    # ---- current iteration state ----
    iteration: int = 0
    generated_files: dict[str, str] = Field(default_factory=dict)
    build_result: BuildResult | None = None
    run_result: RunResult | None = None
    validation_result: ValidationResult | None = None

    # ---- history ----
    history: list[Attempt] = Field(default_factory=list)

    # ---- conversational messages (for chat mode) ----
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # ---- outcome ----
    status: AgentStatus = AgentStatus.RUNNING
    final_message: str = ""

    class Config:
        arbitrary_types_allowed = True

    def snapshot_attempt(self) -> Attempt:
        return Attempt(
            iteration=self.iteration,
            generated_files=dict(self.generated_files),
            build_result=self.build_result,
            run_result=self.run_result,
            validation_result=self.validation_result,
        )
