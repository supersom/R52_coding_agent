"""
FVP / QEMU simulator wrapper.

Launches the simulator with the compiled ELF, captures stdout/stderr,
enforces a timeout, and returns a structured RunResult.

FVP semi-hosting note:
  The FVP_BaseR_Cortex-R52 supports ARM semi-hosting.
  Programs using semi-hosting write to the host terminal via HLT instructions.
  We capture this output through the FVP's --semihosting-enable flag
  and its stdout (which the FVP forwards to the host shell's stdout).
"""

from __future__ import annotations

import subprocess
import time
import signal
import os
from pathlib import Path

from r52_types import RunResult, Simulator
from .config import ToolchainConfig


# Default FVP flags for bare-metal Cortex-R52 run
FVP_DEFAULT_FLAGS = [
    # Disable interactive UI / telnet terminals so FVP runs headless
    "-C", "bp.terminal_0.start_telnet=0",
    "-C", "bp.terminal_1.start_telnet=0",
    "-C", "bp.terminal_2.start_telnet=0",
    "-C", "bp.terminal_3.start_telnet=0",
    "-C", "bp.vis.disable_visualisation=1",
    # Route PL011 UART0 output to our stdout (-)
    "-C", "bp.pl011_uart0.out_file=-",
    "-C", "bp.pl011_uart0.unbuffered_output=1",
    # Semihosting is enabled by default; SVC 0x123456 is the default ARM semihosting SVC
    # cluster0.cpu0.semihosting-ARM_SVC defaults to 0x123456, no override needed.
]

QEMU_DEFAULT_FLAGS = [
    "-M", "versatilepb",
    "-m", "128M",
    "-nographic",
]


def find_elf(repo_path: str) -> str | None:
    """Find the most recently modified ELF in the repo build output."""
    repo = Path(repo_path)
    candidates = list(repo.rglob("*.elf")) + list(repo.rglob("*.axf"))
    # Filter out anything in source control folders
    candidates = [c for c in candidates if ".git" not in str(c)]
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def run_on_fvp(
    elf_path: str,
    config: ToolchainConfig,
    extra_flags: list[str] | None = None,
) -> RunResult:
    """Launch FVP with the given ELF and capture output."""
    cmd = [config.fvp_path] + FVP_DEFAULT_FLAGS
    if extra_flags:
        cmd += extra_flags
    cmd += ["--application", elf_path]

    return _run_subprocess(cmd, config.simulator_timeout)


def run_on_qemu(
    elf_path: str,
    config: ToolchainConfig,
    extra_flags: list[str] | None = None,
) -> RunResult:
    """Launch QEMU with the given ELF and capture output."""
    cmd = [config.qemu_path] + QEMU_DEFAULT_FLAGS
    if extra_flags:
        cmd += extra_flags
    cmd += ["-kernel", elf_path]

    return _run_subprocess(cmd, config.simulator_timeout)


def _run_subprocess(cmd: list[str], timeout: int) -> RunResult:
    start = time.monotonic()
    timed_out = False
    proc = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Create new process group so we can kill the whole group on timeout
            preexec_fn=os.setsid,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Kill the entire process group (FVP spawns sub-processes)
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            stdout, stderr = proc.communicate()
    except FileNotFoundError as e:
        return RunResult(
            success=False,
            timed_out=False,
            stdout="",
            stderr=str(e),
            returncode=-1,
            duration_s=time.monotonic() - start,
        )

    duration = time.monotonic() - start
    return RunResult(
        success=proc.returncode == 0 and not timed_out,
        timed_out=timed_out,
        stdout=stdout or "",
        stderr=stderr or "",
        returncode=proc.returncode if proc else -1,
        duration_s=duration,
    )


def run_simulator(
    repo_path: str,
    simulator: Simulator,
    config: ToolchainConfig,
    elf_path: str | None = None,
) -> RunResult:
    """High-level dispatcher: find ELF and run on the configured simulator."""
    elf = elf_path or find_elf(repo_path)
    if not elf:
        return RunResult(
            success=False,
            timed_out=False,
            stdout="",
            stderr="No ELF found in repo. Build may have failed silently.",
            returncode=-1,
            duration_s=0.0,
        )

    if simulator == Simulator.FVP:
        return run_on_fvp(elf, config)
    return run_on_qemu(elf, config)
