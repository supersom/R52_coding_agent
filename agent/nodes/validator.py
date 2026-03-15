"""
VALIDATE node — compares simulator output to expected output.

Three match modes (as agreed with user):
  exact    — full string equality (after stripping whitespace)
  contains — expected string is a substring of actual output
  regex    — expected is a regex pattern matched against actual output

If no expected_output is configured, validation always passes (run success = done).
"""

from __future__ import annotations

import re

from agent.state import AgentState
from r52_types import MatchMode, ValidationResult


def run_validator(state: AgentState) -> AgentState:
    """Compare run output to expected_output."""
    if not state.expected_output or not state.run_result:
        # No expectation set — treat as pass if run succeeded
        result = ValidationResult(
            passed=True,
            expected="(none)",
            actual=state.run_result.stdout if state.run_result else "",
            match_mode=state.match_mode,
            detail="No expected output specified; run success is sufficient.",
        )
        return state.model_copy(update={"validation_result": result})

    actual = state.run_result.stdout + state.run_result.stderr
    expected = state.expected_output

    passed, detail = _check(actual, expected, state.match_mode)

    result = ValidationResult(
        passed=passed,
        expected=expected,
        actual=actual[:1000],
        match_mode=state.match_mode,
        detail=detail,
    )
    return state.model_copy(update={"validation_result": result})


def validation_passed(state: AgentState) -> bool:
    return state.validation_result is not None and state.validation_result.passed


def _check(actual: str, expected: str, mode: MatchMode) -> tuple[bool, str]:
    if mode == MatchMode.EXACT:
        passed = actual.strip() == expected.strip()
        return passed, (
            "Exact match succeeded." if passed
            else f"Expected:\n{expected!r}\n\nActual:\n{actual[:500]!r}"
        )
    if mode == MatchMode.CONTAINS:
        passed = expected in actual
        return passed, (
            f"Found {expected!r} in output." if passed
            else f"Expected to find:\n{expected!r}\n\nIn output:\n{actual[:500]!r}"
        )
    if mode == MatchMode.REGEX:
        m = re.search(expected, actual, re.MULTILINE)
        passed = m is not None
        return passed, (
            f"Regex {expected!r} matched at {m.span()}." if passed
            else f"Regex {expected!r} did not match in:\n{actual[:500]!r}"
        )
    return False, f"Unknown match mode: {mode}"
