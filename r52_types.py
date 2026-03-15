"""
Shared types used by both agent/ and toolchain/ packages.

Lives at root level to break the circular import that would occur if
toolchain/ imported from agent/ (and agent/ imports from toolchain/).

Rule: this file must not import from agent/ or toolchain/.
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel


class AgentStatus(str, Enum):
    RUNNING     = "running"
    SUCCESS     = "success"
    FAILED      = "failed"
    MAX_RETRIES = "max_retries"


class MatchMode(str, Enum):
    EXACT    = "exact"
    CONTAINS = "contains"
    REGEX    = "regex"


class Toolchain(str, Enum):
    GNU      = "gnu"
    ARMCLANG = "armclang"


class BuildSystem(str, Enum):
    MAKE  = "make"
    CMAKE = "cmake"
    NONE  = "none"


class Simulator(str, Enum):
    FVP  = "fvp"
    QEMU = "qemu"


class BuildResult(BaseModel):
    success: bool
    command: str
    stdout: str
    stderr: str
    returncode: int
    duration_s: float


class RunResult(BaseModel):
    success: bool
    timed_out: bool
    stdout: str
    stderr: str
    returncode: int
    duration_s: float


class ValidationResult(BaseModel):
    passed: bool
    expected: str
    actual: str
    match_mode: MatchMode
    detail: str


class GeneratedFile(BaseModel):
    path: str
    content: str
    rationale: str = ""


class CodeGenerationOutput(BaseModel):
    """Structured output from the LLM code generation step (enforced via instructor)."""
    files: list[GeneratedFile]
    explanation: str
    build_notes: str = ""
