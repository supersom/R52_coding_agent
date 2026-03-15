"""
GNU arm-none-eabi toolchain wrapper.

Wraps arm-none-eabi-gcc and arm-none-eabi-ld with sensible Cortex-R52 defaults.
Supports both:
  - Direct compilation (list of .c and .s files)
  - Make/CMake invocation (when a build system is present)
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from r52_types import BuildResult
from .config import ToolchainConfig


# Default compiler flags for Cortex-R52 AArch32 (arm-none-eabi targets AArch32)
GNU_DEFAULT_CFLAGS = [
    "-mcpu=cortex-r52",
    "-marm",                  # AArch32 ARM mode (arm-none-eabi targets AArch32)
    "-mfpu=crypto-neon-fp-armv8",
    "-mfloat-abi=hard",
    "-O2",
    "-g",
    "-Wall",
    "-Wextra",
    "-ffreestanding",
    "-nostdlib",
    "-ffunction-sections",
    "-fdata-sections",
]

GNU_DEFAULT_LDFLAGS = [
    "-nostdlib",
    "--gc-sections",
]


def build_with_make(
    repo_path: str,
    config: ToolchainConfig,
    extra_env: dict | None = None,
) -> BuildResult:
    """Run `make` in repo_path and capture result."""
    import os
    env = os.environ.copy()
    env["CROSS_COMPILE"] = "arm-none-eabi-"
    env["CC"] = config.gnu_gcc_path
    if extra_env:
        env.update(extra_env)

    start = time.monotonic()
    result = subprocess.run(
        ["make", "-C", repo_path],
        capture_output=True,
        text=True,
        env=env,
    )
    duration = time.monotonic() - start

    return BuildResult(
        success=result.returncode == 0,
        command=f"make -C {repo_path}",
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        duration_s=duration,
    )


def build_with_cmake(
    repo_path: str,
    config: ToolchainConfig,
) -> BuildResult:
    """Configure (if needed) and build with CMake."""
    import os
    repo = Path(repo_path)
    build_dir = repo / "build"
    build_dir.mkdir(exist_ok=True)

    start = time.monotonic()

    # Configure step
    cmake_configure = subprocess.run(
        [
            "cmake",
            "-S", str(repo),
            "-B", str(build_dir),
            f"-DCMAKE_C_COMPILER={config.gnu_gcc_path}",
            "-DCMAKE_SYSTEM_NAME=Generic",
            f"-DCMAKE_C_FLAGS={' '.join(GNU_DEFAULT_CFLAGS)}",
        ],
        capture_output=True,
        text=True,
    )
    if cmake_configure.returncode != 0:
        return BuildResult(
            success=False,
            command="cmake configure",
            stdout=cmake_configure.stdout,
            stderr=cmake_configure.stderr,
            returncode=cmake_configure.returncode,
            duration_s=time.monotonic() - start,
        )

    # Build step
    cmake_build = subprocess.run(
        ["cmake", "--build", str(build_dir)],
        capture_output=True,
        text=True,
    )
    return BuildResult(
        success=cmake_build.returncode == 0,
        command=f"cmake --build {build_dir}",
        stdout=cmake_configure.stdout + cmake_build.stdout,
        stderr=cmake_configure.stderr + cmake_build.stderr,
        returncode=cmake_build.returncode,
        duration_s=time.monotonic() - start,
    )


def build_direct(
    sources: list[str],
    output_elf: str,
    linker_script: str | None,
    include_dirs: list[str],
    config: ToolchainConfig,
) -> BuildResult:
    """Compile and link a list of source files directly (no build system)."""
    cmd = [config.gnu_gcc_path] + GNU_DEFAULT_CFLAGS + config.extra_cflags
    cmd += [f"-I{d}" for d in include_dirs]
    cmd += sources
    cmd += ["-o", output_elf]
    if linker_script:
        cmd += [f"-T{linker_script}"]
    cmd += [f"-Wl,{f}" for f in GNU_DEFAULT_LDFLAGS + config.extra_ldflags]

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
