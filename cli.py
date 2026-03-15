"""
R52 Coding Agent CLI

Modes:
  r52 run      — single-shot task (prompt inline or from markdown file)
  r52 chat     — conversational mode (accumulates context across prompts)
  r52 repl     — REPL mode (independent tasks, cached repo context)
  r52 eval     — batch evaluation against a suite YAML
  r52 config   — view/set toolchain configuration
  r52 logs     — browse past run logs

Design: Click with subcommands. Each subcommand maps to a distinct agent mode.
Click is chosen over argparse/typer because it's the most widely used Python
CLI library, has excellent help generation, and supports nesting cleanly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import shutil

import click
from dotenv import load_dotenv
from rich.console import Console

# Load .env from cwd first (project-local), then from the script's own directory.
# cwd() can raise FileNotFoundError if the directory was deleted — ignore gracefully.
try:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
except FileNotFoundError:
    pass
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

console = Console()


# ---------------------------------------------------------------------------
# Backend auto-detection
# ---------------------------------------------------------------------------

# Priority: API backends first (more reliable), CLI tools as fallback.
# Within API backends: Anthropic → OpenAI → OpenRouter.
_BACKEND_PRIORITY = [
    ("anthropic-api", lambda: os.environ.get("ANTHROPIC_API_KEY")),
    ("openai-api",    lambda: os.environ.get("OPENAI_API_KEY")),
    ("openrouter",    lambda: os.environ.get("OPENROUTER_API_KEY")),
    ("claude-cli",    lambda: shutil.which("claude")),
    ("gemini-cli",    lambda: shutil.which("gemini")),
    ("codex-cli",     lambda: shutil.which("codex")),
]


def _detect_backend() -> str:
    """Return the highest-priority backend that has a key or binary available."""
    for name, check in _BACKEND_PRIORITY:
        if check():
            return name
    return "anthropic-api"  # fallback — will give a clear missing-key error


# ---------------------------------------------------------------------------
# Shared options (applied to multiple subcommands via @click.pass_context)
# ---------------------------------------------------------------------------

BACKEND_CHOICES = [
    "anthropic-api", "openai-api", "openrouter",
    "claude-cli", "gemini-cli", "codex-cli",
]
TOOLCHAIN_CHOICES = ["gnu", "armclang"]
SIMULATOR_CHOICES = ["fvp", "qemu"]
MATCH_CHOICES = ["exact", "contains", "regex"]


def _common_options(fn):
    """Decorator that adds shared options to a Click command."""
    fn = click.option("--backend", "-b", default=_detect_backend,
                      type=click.Choice(BACKEND_CHOICES),
                      help="LLM backend (auto-detected from env if not set).",
                      show_default=True)(fn)
    fn = click.option("--model", "-m", default=None,
                      help="Override model name for the selected backend.")(fn)
    fn = click.option("--toolchain", "-t", default="gnu",
                      type=click.Choice(TOOLCHAIN_CHOICES),
                      help="Compiler toolchain.", show_default=True)(fn)
    fn = click.option("--simulator", "-s", default="fvp",
                      type=click.Choice(SIMULATOR_CHOICES),
                      help="Target simulator.", show_default=True)(fn)
    fn = click.option("--max-iterations", default=10,
                      help="Max fix iterations before giving up.", show_default=True)(fn)
    fn = click.option("--timeout", default=600,
                      help="Simulator timeout in seconds.", show_default=True)(fn)
    fn = click.option("--verbose", is_flag=True, help="Verbose output.")(fn)
    return fn


def _load_backend_and_config(backend, model, toolchain, simulator_timeout):
    from backends import get_backend
    from toolchain.config import ToolchainConfig
    from agent.state import Toolchain, Simulator

    cfg = ToolchainConfig.load()
    cfg.simulator_timeout = simulator_timeout

    be = get_backend(backend, model)
    tc = Toolchain(toolchain)

    return be, cfg, tc


def _parse_prompt_file(prompt_file: str) -> tuple[str, str | None]:
    """
    Parse a markdown prompt file.
    Extracts task description and optional ## Expected Output section.
    Returns (task, expected_output).
    """
    content = Path(prompt_file).read_text()
    lines = content.splitlines()

    # Extract Expected Output section if present
    expected_output = None
    task_lines = []
    in_expected = False
    expected_lines = []

    for line in lines:
        if line.strip().lower().startswith("## expected output"):
            in_expected = True
            continue
        if in_expected and line.startswith("## "):
            in_expected = False
        if in_expected:
            expected_lines.append(line)
        else:
            task_lines.append(line)

    if expected_lines:
        expected_output = "\n".join(expected_lines).strip()

    task = "\n".join(task_lines).strip()
    return task, expected_output


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option("0.1.0", prog_name="r52")
def main():
    """R52 Coding Agent — autonomous ARM Cortex-R52 bare-metal code generation."""
    pass


# ---------------------------------------------------------------------------
# r52 run
# ---------------------------------------------------------------------------

@main.command()
@click.argument("prompt", required=False)
@click.option("--prompt-file", "-f", type=click.Path(exists=True),
              help="Path to markdown file containing the task description.")
@click.option("--repo", "-r", required=True, type=click.Path(),
              help="Path to target repository/project folder.")
@click.option("--expected-output", "-e", default=None,
              help="Expected simulator output string (for validation).")
@click.option("--match", default="contains", type=click.Choice(MATCH_CHOICES),
              help="How to match expected output.", show_default=True)
@click.option("--no-review", is_flag=True,
              help="Skip the code review step (faster, less reliable).")
@click.option("--new-project", is_flag=True,
              help="Force scaffold a new project (ignore existing repo contents).")
@_common_options
def run(
    prompt, prompt_file, repo, expected_output, match,
    no_review, new_project,
    backend, model, toolchain, simulator, max_iterations, timeout, verbose,
):
    """
    Run a single feature implementation task.

    PROMPT can be given inline or via --prompt-file path/to/feature.md

    Examples:
      r52 run "Add UART TX" --repo ./my_project --backend anthropic-api
      r52 run --prompt-file feature.md --repo ./my_project --expected-output "UART OK"
    """
    # Resolve prompt
    task = prompt
    file_expected = None

    if prompt_file:
        task, file_expected = _parse_prompt_file(prompt_file)

    if not task:
        # Try reading from stdin
        if not sys.stdin.isatty():
            task = sys.stdin.read().strip()
        else:
            raise click.UsageError("Provide a prompt inline, via --prompt-file, or via stdin.")

    # --expected-output flag takes precedence over file section
    effective_expected = expected_output or file_expected

    _run_single_task(
        task=task,
        repo=repo,
        expected_output=effective_expected,
        match=match,
        backend=backend,
        model=model,
        toolchain=toolchain,
        simulator=simulator,
        max_iterations=max_iterations,
        timeout=timeout,
        verbose=verbose,
        new_project=new_project,
        skip_review=no_review,
    )


def _run_single_task(
    task, repo, expected_output, match,
    backend, model, toolchain, simulator,
    max_iterations, timeout, verbose,
    new_project=False, skip_review=False,
):
    """Shared implementation for run and repl modes."""
    from agent.state import AgentState, MatchMode, Toolchain, Simulator
    from agent.graph import run_agent
    from backends import get_backend
    from toolchain.config import ToolchainConfig
    from observability.logger import RunLogger
    from observability.tracer import AgentTracer
    from observability.rich_ui import AgentUI, QuietUI

    repo_path = str(Path(repo).resolve())
    Path(repo_path).mkdir(parents=True, exist_ok=True)

    # For new projects, apply the default template scaffold before the agent runs.
    # This gives the generator concrete files to build on (startup.s, link.ld, Makefile)
    # rather than generating everything from scratch.
    template_used = None
    if new_project:
        from templates import apply_template
        template_used = "cortex-r52-baremetal"
        copied = apply_template(template_used, repo_path)
        console.print(f"[dim]Template applied: {template_used} ({len(copied)} files)[/]")

    be = get_backend(backend, model)
    cfg = ToolchainConfig.load()
    cfg.simulator_timeout = timeout

    initial = AgentState(
        task=task,
        repo_path=repo_path,
        expected_output=expected_output,
        match_mode=MatchMode(match),
        toolchain=Toolchain(toolchain),
        simulator=Simulator(simulator),
        max_iterations=max_iterations,
        simulator_timeout=timeout,
        is_new_project=new_project,
        template_used=template_used,
    )

    logger = RunLogger(initial.trace_id)
    tracer = AgentTracer(initial.trace_id)
    ui = (QuietUI if not sys.stdout.isatty() else AgentUI)(
        task=task, trace_id=initial.trace_id,
        max_iterations=max_iterations, verbose=verbose,
    )

    console.print(f"\n[dim]trace_id: {initial.trace_id}[/]")
    console.print(f"[dim]log: {logger.log_path}[/]\n")

    ui.start()
    try:
        final = run_agent(initial, be, cfg, logger, tracer, ui)
    finally:
        logger.close()
        tracer.shutdown()

    if final.status.value == "success":
        ui.success(f"Done in {final.iteration} iteration(s).")
    else:
        ui.failure(f"Failed after {final.iteration} iteration(s): {final.final_message[:200]}")

    return final


# ---------------------------------------------------------------------------
# r52 chat — conversational mode
# ---------------------------------------------------------------------------

@main.command()
@click.option("--repo", "-r", required=True, type=click.Path(),
              help="Path to target repository/project folder.")
@_common_options
def chat(repo, backend, model, toolchain, simulator, max_iterations, timeout, verbose):
    """
    Conversational mode — accumulates context across multiple prompts.

    The agent remembers previous code, build results, and your follow-up
    instructions within the session. Good for exploratory development.

    Type 'exit' or Ctrl-D to quit.
    """
    from agent.state import AgentState, Toolchain, Simulator, MatchMode
    from agent.graph import run_agent
    from backends import get_backend
    from toolchain.config import ToolchainConfig
    from observability.logger import RunLogger
    from observability.tracer import AgentTracer
    from observability.rich_ui import AgentUI, QuietUI

    be = get_backend(backend, model)
    cfg = ToolchainConfig.load()
    cfg.simulator_timeout = timeout
    repo_path = str(Path(repo).resolve())

    console.print(f"\n[bold cyan]R52 Chat Mode[/]  (repo: {repo_path})")
    console.print("[dim]Type 'exit' to quit. Provide multi-line input with a blank line to submit.[/]\n")

    # Persistent state across turns
    session_state: AgentState | None = None

    while True:
        try:
            lines = []
            console.print("[bold]>[/] ", end="")
            while True:
                try:
                    line = input()
                except EOFError:
                    line = "exit"
                if line.lower() == "exit":
                    console.print("[dim]Goodbye.[/]")
                    return
                if line == "" and lines:
                    break
                lines.append(line)
            task = "\n".join(lines).strip()
            if not task:
                continue

        except KeyboardInterrupt:
            console.print("\n[dim]Use 'exit' to quit.[/]")
            continue

        # Build state: carry over history and generated files from previous turn
        initial = AgentState(
            task=task,
            repo_path=repo_path,
            toolchain=Toolchain(toolchain),
            simulator=Simulator(simulator),
            max_iterations=max_iterations,
            simulator_timeout=timeout,
            # Carry conversational context from previous turn
            history=session_state.history if session_state else [],
            generated_files=session_state.generated_files if session_state else {},
            repo_context=session_state.repo_context if session_state else {},
        )

        logger = RunLogger(initial.trace_id)
        tracer = AgentTracer(initial.trace_id)
        ui_cls = QuietUI if not sys.stdout.isatty() else AgentUI
        ui = ui_cls(task=task, trace_id=initial.trace_id,
                    max_iterations=max_iterations, verbose=verbose)

        ui.start()
        try:
            final = run_agent(initial, be, cfg, logger, tracer, ui)
        finally:
            logger.close()
            tracer.shutdown()

        session_state = final   # accumulate for next turn

        if final.status.value == "success":
            ui.success(f"Done in {final.iteration} iteration(s).")
        else:
            ui.failure(f"Problem: {final.final_message[:200]}")

        console.print("")


# ---------------------------------------------------------------------------
# r52 repl — independent tasks with cached repo context
# ---------------------------------------------------------------------------

@main.command()
@click.option("--repo", "-r", required=True, type=click.Path(),
              help="Path to target repository/project folder.")
@_common_options
def repl(repo, backend, model, toolchain, simulator, max_iterations, timeout, verbose):
    """
    REPL mode — each prompt is an independent task, repo context is cached.

    Unlike chat mode, tasks don't share agent state — each runs the full
    plan→generate→build→run cycle fresh. Repo file scanning is cached
    between tasks to reduce latency.

    Type 'exit' to quit.
    """
    console.print(f"\n[bold cyan]R52 REPL Mode[/]  (repo: {repo})")
    console.print("[dim]Each prompt runs an independent task. Type 'exit' to quit.[/]\n")

    while True:
        try:
            task = click.prompt("task", prompt_suffix="> ")
        except (click.Abort, EOFError):
            break
        if task.lower() in ("exit", "quit"):
            break
        if task.startswith("--prompt-file "):
            fpath = task.split(None, 1)[1].strip()
            task, _ = _parse_prompt_file(fpath)

        _run_single_task(
            task=task, repo=repo,
            expected_output=None, match="contains",
            backend=backend, model=model, toolchain=toolchain,
            simulator=simulator, max_iterations=max_iterations,
            timeout=timeout, verbose=verbose,
        )
        console.print("")


# ---------------------------------------------------------------------------
# r52 eval
# ---------------------------------------------------------------------------

@main.command()
@click.argument("suite", type=click.Path(exists=True))
@click.option("--report", "-o", default=None,
              help="Output path for eval report (.md or .json).")
@_common_options
def eval(suite, report, backend, model, toolchain, simulator, max_iterations, timeout, verbose):
    """
    Run an eval suite YAML and produce a performance report.

    SUITE is the path to an eval_suite.yaml file.

    Example:
      r52 eval eval_suite.yaml --report report.md --backend anthropic-api
    """
    from eval.harness import run_eval_suite
    from toolchain.config import ToolchainConfig

    cfg = ToolchainConfig.load()
    cfg.simulator_timeout = timeout

    run_eval_suite(
        suite_path=suite,
        backend_name=backend,
        model=model,
        config=cfg,
        report_path=report,
    )


# ---------------------------------------------------------------------------
# r52 config
# ---------------------------------------------------------------------------

@main.group()
def config():
    """View and set toolchain/agent configuration."""
    pass


@config.command("show")
def config_show():
    """Print current configuration."""
    from toolchain.config import ToolchainConfig
    cfg = ToolchainConfig.load()
    from rich.table import Table
    t = Table(title="R52 Agent Configuration", show_header=True)
    t.add_column("Key")
    t.add_column("Value")
    for field in cfg.__dataclass_fields__:
        t.add_row(field, str(getattr(cfg, field)))
    console.print(t)


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a configuration value (e.g. r52 config set toolchain.fvp_path /path/to/fvp)."""
    from toolchain.config import ToolchainConfig
    cfg = ToolchainConfig.load()
    key = key.replace("toolchain.", "")
    if not hasattr(cfg, key):
        raise click.UsageError(f"Unknown config key: {key}")
    current = getattr(cfg, key)
    setattr(cfg, key, type(current)(value))
    cfg.save()
    console.print(f"[green]Set {key} = {value}[/]")


# ---------------------------------------------------------------------------
# r52 logs
# ---------------------------------------------------------------------------

@main.command()
@click.option("--last", "-n", default=10, help="Show last N runs.")
@click.option("--trace-id", default=None, help="Show full log for a specific trace ID.")
def logs(last, trace_id):
    """Browse past run logs from ~/.r52agent/runs/."""
    from observability.logger import RUNS_DIR
    import json

    runs_dir = RUNS_DIR
    if not runs_dir.exists():
        console.print("[dim]No runs logged yet.[/]")
        return

    if trace_id:
        log_file = runs_dir / f"{trace_id}.jsonl"
        if not log_file.exists():
            console.print(f"[red]No log for trace {trace_id}[/]")
            return
        for line in log_file.read_text().splitlines():
            console.print_json(line)
        return

    # List recent runs (each .jsonl = one run)
    files = sorted(runs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    from rich.table import Table
    t = Table(title="Recent Runs", show_header=True)
    t.add_column("Trace ID", style="cyan")
    t.add_column("Task")
    t.add_column("Status")
    t.add_column("Time")

    for f in files[:last]:
        lines = f.read_text().splitlines()
        start = end = task = status = ""
        for line in lines:
            try:
                ev = json.loads(line)
                if ev["type"] == "run_start":
                    task = ev.get("task", "")[:50]
                    start = ev.get("ts", "")[:19]
                if ev["type"] == "run_end":
                    status = ev.get("status", "")
                    end = ev.get("ts", "")[:19]
            except Exception:
                pass
        t.add_row(f.stem[:8], task, status, start)

    console.print(t)


if __name__ == "__main__":
    main()
