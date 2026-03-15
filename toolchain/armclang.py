"""
ARM Compiler 6 (armclang) toolchain wrapper.

armclang targets AArch64 by default for Cortex-R52 with:
  --target=aarch64-arm-none-eabi
  -mcpu=cortex-r52
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from r52_types import BuildResult
from .config import ToolchainConfig


ARMCLANG_DEFAULT_CFLAGS = [
    "--target=aarch64-arm-none-eabi",
    "-mcpu=cortex-r52",
    "-O2",
    "-g",
    "-Wall",
    "-ffreestanding",
    "-nostdlib",
]


def build_with_make(
    repo_path: str,
    config: ToolchainConfig,
) -> BuildResult:
    """Run `make` with armclang as compiler."""
    import os
    env = os.environ.copy()
    env["CC"] = config.armclang_path
    env["LD"] = config.armlink_path

    start = time.monotonic()
    result = subprocess.run(
        ["make", "-C", repo_path],
        capture_output=True,
        text=True,
        env=env,
    )
    return BuildResult(
        success=result.returncode == 0,
        command=f"make -C {repo_path} CC={config.armclang_path}",
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        duration_s=time.monotonic() - start,
    )


def build_direct(
    sources: list[str],
    output_elf: str,
    linker_script: str | None,
    include_dirs: list[str],
    config: ToolchainConfig,
) -> BuildResult:
    """Direct armclang compilation."""
    cmd = [config.armclang_path] + ARMCLANG_DEFAULT_CFLAGS + config.extra_cflags
    cmd += [f"-I{d}" for d in include_dirs]
    cmd += sources
    cmd += ["-o", output_elf]
    if linker_script:
        cmd += [f"-T{linker_script}"]

    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    return BuildResult(
        success=result.returncode == 0,
        command=" ".join(cmd),
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        duration_s=time.monotonic() - start,
    )
