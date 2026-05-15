"""
Probe primitives: run_command and read_file.

Two generic tools for LLM-directed investigation. Deny lists and access
controls are the caller's responsibility.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def run_command(command: str, timeout: int = 15) -> tuple[str, bool]:
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        output = (r.stdout + r.stderr).strip()
        return output or "(no output)", r.returncode == 0
    except subprocess.TimeoutExpired:
        return "(command timed out)", False
    except OSError as e:
        return f"(error: {e})", False


def read_file(path: str) -> tuple[str, bool]:
    p = Path(path)
    if not p.exists():
        return f"(not found: {path})", False
    try:
        content = p.read_text(errors="replace")
        if len(content) > 20000:
            content = content[:20000] + "\n... (truncated)"
        return content, True
    except OSError as e:
        return f"(read error: {e})", False
