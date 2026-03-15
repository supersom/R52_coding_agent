"""
Repo context extractor.

Walks a target repository and produces a structured summary:
  - File tree
  - Detected build system (make/cmake)
  - Key files (startup, linker script, main, headers)
  - Existing symbols (function names from C files)
  - Relevant file contents (truncated to fit LLM context)

Design note: We don't embed the entire repo into every LLM prompt — that
would blow context limits. Instead we produce a tiered summary:
  tier 1: file tree + build system (always included)
  tier 2: key file contents (startup, linker script, main)
  tier 3: all other .c/.h files, truncated
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


# File extensions we care about
CODE_EXTS = {".c", ".h", ".s", ".S", ".ld", ".mk", ".cmake"}
BUILD_FILES = {"Makefile", "makefile", "GNUmakefile", "CMakeLists.txt"}

# Files likely to be "key" files for bare-metal startup
STARTUP_PATTERNS = re.compile(
    r"startup|start|reset|boot|crt0|vectors|vector_table",
    re.IGNORECASE,
)
LINKER_PATTERNS = re.compile(r"\.ld$|linker|link\.ld|scatter", re.IGNORECASE)
MAX_FILE_CHARS = 4000   # per file content cap in context


def detect_build_system(repo: Path) -> str:
    """Return 'cmake', 'make', or 'none'."""
    if (repo / "CMakeLists.txt").exists():
        return "cmake"
    for name in BUILD_FILES:
        if (repo / name).exists():
            return "make"
    # Check one level deep
    for child in repo.iterdir():
        if child.is_dir():
            for name in BUILD_FILES:
                if (child / name).exists():
                    return "make"
    return "none"


def extract_c_symbols(content: str) -> list[str]:
    """Very simple C function signature extractor."""
    pattern = re.compile(
        r"^\s*(?:static\s+)?(?:inline\s+)?(?:volatile\s+)?"
        r"[\w\s\*]+\s+(\w+)\s*\([^)]*\)\s*\{",
        re.MULTILINE,
    )
    return pattern.findall(content)


def read_repo_context(repo_path: str, max_total_chars: int = 40_000) -> dict[str, Any]:
    """
    Walk the repo and return a structured context dict suitable for
    inclusion in LLM prompts.
    """
    repo = Path(repo_path).resolve()
    if not repo.exists():
        return {"error": f"Repo path not found: {repo_path}", "files": {}}

    build_system = detect_build_system(repo)

    # Collect all relevant files
    all_files: list[Path] = []
    for root, dirs, files in os.walk(repo):
        # Skip hidden dirs, build output dirs
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".")
            and d not in {"build", "out", "output", "_build", "cmake-build-debug"}
        ]
        for f in files:
            p = Path(root) / f
            if p.suffix in CODE_EXTS or p.name in BUILD_FILES:
                all_files.append(p)

    # Categorise files
    startup_files = [f for f in all_files if STARTUP_PATTERNS.search(f.name)]
    linker_files  = [f for f in all_files if LINKER_PATTERNS.search(f.name)]
    build_files   = [f for f in all_files if f.name in BUILD_FILES]
    other_files   = [
        f for f in all_files
        if f not in startup_files + linker_files + build_files
    ]

    # Build file tree string
    tree_lines = []
    for f in sorted(all_files):
        rel = f.relative_to(repo)
        tree_lines.append(str(rel))
    file_tree = "\n".join(tree_lines)

    # Read key file contents
    key_contents: dict[str, str] = {}
    budget = max_total_chars
    priority_files = startup_files + linker_files + build_files

    for f in priority_files + other_files:
        if budget <= 0:
            break
        try:
            content = f.read_text(errors="replace")
            rel = str(f.relative_to(repo))
            truncated = content[:MAX_FILE_CHARS]
            if len(content) > MAX_FILE_CHARS:
                truncated += f"\n... [truncated, {len(content)} chars total]"
            key_contents[rel] = truncated
            budget -= len(truncated)
        except (OSError, PermissionError):
            pass

    # Extract symbols from C files
    all_symbols: list[str] = []
    for path, content in key_contents.items():
        if path.endswith(".c"):
            all_symbols.extend(extract_c_symbols(content))

    return {
        "repo_path": str(repo),
        "build_system": build_system,
        "file_tree": file_tree,
        "startup_files": [str(f.relative_to(repo)) for f in startup_files],
        "linker_files":  [str(f.relative_to(repo)) for f in linker_files],
        "build_files":   [str(f.relative_to(repo)) for f in build_files],
        "file_contents": key_contents,
        "symbols": all_symbols,
        "total_files": len(all_files),
    }


def format_context_for_prompt(ctx: dict[str, Any]) -> str:
    """Format repo context dict as a string for LLM prompt inclusion."""
    if "error" in ctx:
        return f"[Repo context error: {ctx['error']}]"

    parts = [
        f"## Repository: {ctx['repo_path']}",
        f"Build system: {ctx['build_system']}",
        f"Total files: {ctx['total_files']}",
        "",
        "### File tree",
        ctx["file_tree"],
        "",
    ]

    if ctx["startup_files"]:
        parts += ["### Key files: Startup", "\n".join(ctx["startup_files"]), ""]
    if ctx["linker_files"]:
        parts += ["### Key files: Linker scripts", "\n".join(ctx["linker_files"]), ""]

    parts.append("### File contents")
    for path, content in ctx["file_contents"].items():
        parts += [f"\n#### {path}", "```", content, "```"]

    return "\n".join(parts)
