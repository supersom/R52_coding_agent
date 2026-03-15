"""
Build system dispatcher — selects make/cmake/direct based on repo and toolchain.
"""

from __future__ import annotations

from r52_types import BuildResult, BuildSystem, Toolchain
from .config import ToolchainConfig
from . import gnu, armclang as armclang_tc


def run_build(
    repo_path: str,
    build_system: BuildSystem,
    toolchain: Toolchain,
    config: ToolchainConfig,
    sources: list[str] | None = None,
    output_elf: str | None = None,
    linker_script: str | None = None,
    include_dirs: list[str] | None = None,
) -> BuildResult:
    """
    Unified build entry point.

    Priority:
      cmake  → cmake configure + build
      make   → make -C repo_path
      none   → direct compiler invocation (sources must be provided)
    """
    inc = include_dirs or []

    if build_system in (BuildSystem.CMAKE, BuildSystem.MAKE) and not repo_path:
        return BuildResult(
            success=False,
            command="",
            stdout="",
            stderr="repo_path is empty; cannot run make/cmake without a target directory",
            returncode=2,
            duration_s=0.0,
        )

    if build_system == BuildSystem.CMAKE:
        if toolchain == Toolchain.ARMCLANG:
            # CMake + armclang: use make fallback (armclang cmake toolchain file is complex)
            return armclang_tc.build_with_make(repo_path, config)
        return gnu.build_with_cmake(repo_path, config)

    if build_system == BuildSystem.MAKE:
        if toolchain == Toolchain.ARMCLANG:
            return armclang_tc.build_with_make(repo_path, config)
        return gnu.build_with_make(repo_path, config)

    # Direct — no build system
    if not sources or not output_elf:
        return BuildResult(
            success=False,
            command="",
            stdout="",
            stderr="No sources or output ELF specified for direct build",
            returncode=1,
            duration_s=0.0,
        )
    if toolchain == Toolchain.ARMCLANG:
        return armclang_tc.build_direct(sources, output_elf, linker_script, inc, config)
    return gnu.build_direct(sources, output_elf, linker_script, inc, config)
