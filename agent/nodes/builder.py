"""
BUILD node — invokes the configured ARM toolchain.

Includes deterministic pre-build fixups so trivial LLM mistakes
(wrong Makefile indentation, missing newline at EOF) don't consume
a full LLM patcher iteration.
"""

from __future__ import annotations

import re
from pathlib import Path

from agent.state import AgentState
from r52_types import BuildResult
from toolchain.build_system import run_build
from toolchain.config import ToolchainConfig


def _fix_makefile_tabs(repo_path: str) -> bool:
    """
    Replace leading spaces with a tab on Makefile recipe lines.

    Makefiles require a hard tab at the start of every recipe line.
    LLMs almost always emit spaces instead. This is a deterministic
    transformation — no LLM call needed.

    Returns True if any fix was applied.
    """
    for name in ("Makefile", "makefile", "GNUmakefile"):
        mf = Path(repo_path) / name
        if not mf.exists():
            continue

        lines = mf.read_text().splitlines(keepends=True)
        fixed = []
        changed = False
        in_rule = False

        for line in lines:
            # A rule target line looks like: "target: deps"
            # After a target, recipe lines must start with a tab.
            stripped = line.rstrip("\n\r")

            if re.match(r'^[a-zA-Z0-9_./%\$][^=]*:', stripped):
                in_rule = True
                fixed.append(line)
            elif in_rule and stripped and stripped[0] == " ":
                # Recipe line with spaces — convert leading spaces to one tab
                new_line = "\t" + stripped.lstrip() + "\n"
                fixed.append(new_line)
                changed = True
            else:
                if not stripped:
                    in_rule = False   # blank line ends a rule block
                fixed.append(line)

        if changed:
            mf.write_text("".join(fixed))
        return changed

    return False


def run_builder(state: AgentState, config: ToolchainConfig) -> AgentState:
    """Attempt to build. Applies deterministic pre-build fixups first."""
    # Fix Makefile tab indentation silently — no LLM needed for this
    _fix_makefile_tabs(state.repo_path)

    result: BuildResult = run_build(
        repo_path=state.repo_path,
        build_system=state.build_system,
        toolchain=state.toolchain,
        config=config,
    )
    return state.model_copy(update={"build_result": result})


def build_succeeded(state: AgentState) -> bool:
    return state.build_result is not None and state.build_result.success
