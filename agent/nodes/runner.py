"""
RUN node — launches the compiled ELF on FVP or QEMU.

Captures simulator stdout (which receives semi-hosting output and UART output).
"""

from __future__ import annotations

from agent.state import AgentState
from r52_types import RunResult
from toolchain.simulator import run_simulator
from toolchain.config import ToolchainConfig


def run_runner(state: AgentState, config: ToolchainConfig) -> AgentState:
    """Run the compiled binary on the configured simulator."""
    result: RunResult = run_simulator(
        repo_path=state.repo_path,
        simulator=state.simulator,
        config=config,
    )
    return state.model_copy(update={"run_result": result})


def run_succeeded(state: AgentState) -> bool:
    return state.run_result is not None and state.run_result.success
