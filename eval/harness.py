"""
Eval harness — batch evaluation of agent performance.

Loads an eval suite YAML, runs each task through the full agent pipeline,
and produces a structured report (Markdown + JSON).

This is how you measure:
  - Success rate across task types
  - Average / p50 / p95 iterations needed
  - Average / p50 / p95 time to success
  - Which failure modes are most common (build, run, validation)

Eval suite YAML format:
  suite: "my-suite"
  tasks:
    - id: task_id
      description: "What to implement"
      repo: ./path/to/fixture  # or "new" for a fresh project
      template: cortex-r52-baremetal  # used when repo=new
      expected_output: "expected string"
      match: contains  # exact | contains | regex
      max_iterations: 5
      toolchain: gnu   # gnu | armclang
"""

from __future__ import annotations

import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table
from rich import box

from agent.graph import run_agent
from agent.state import AgentState, AgentStatus, MatchMode, Toolchain
from backends import get_backend
from toolchain.config import ToolchainConfig
from observability.logger import RunLogger
from observability.tracer import AgentTracer
from observability.rich_ui import QuietUI


console = Console()


def run_eval_suite(
    suite_path: str,
    backend_name: str,
    model: str | None,
    config: ToolchainConfig,
    report_path: str | None = None,
) -> dict[str, Any]:
    """Run all tasks in a suite and return aggregated results."""
    with open(suite_path) as f:
        suite = yaml.safe_load(f)

    suite_name = suite.get("suite", "unnamed")
    tasks = suite.get("tasks", [])
    backend = get_backend(backend_name, model)
    results = []

    console.print(f"\n[bold]Eval suite: {suite_name}[/] ({len(tasks)} tasks)\n")

    for task_cfg in tasks:
        result = _run_eval_task(task_cfg, backend, config)
        results.append(result)
        status_style = "green" if result["passed"] else "red"
        console.print(
            f"  [{status_style}]{result['id']:20s}[/] "
            f"{'PASS' if result['passed'] else 'FAIL':5s} "
            f"iter={result['iterations']}  "
            f"time={result['total_s']:.1f}s"
        )

    report = _build_report(suite_name, results)
    _print_report(report)

    if report_path:
        _write_report(report, report_path)
        console.print(f"\nReport written to: {report_path}")

    return report


def _run_eval_task(task_cfg: dict, backend, config: ToolchainConfig) -> dict[str, Any]:
    task_id   = task_cfg["id"]
    task_desc = task_cfg["description"]
    repo_spec = task_cfg.get("repo", "new")
    template  = task_cfg.get("template", "cortex-r52-baremetal")

    t0 = time.monotonic()

    # Set up repo: copy fixture or use temp dir for new project
    with tempfile.TemporaryDirectory() as tmpdir:
        if repo_spec == "new":
            repo_path = tmpdir
            # Apply template scaffold
            from templates import apply_template
            apply_template(template, repo_path)
        else:
            # Copy fixture to temp so each run is isolated
            repo_path = tmpdir
            shutil.copytree(repo_spec, repo_path, dirs_exist_ok=True)

        initial_state = AgentState(
            task=task_desc,
            repo_path=repo_path,
            expected_output=task_cfg.get("expected_output"),
            match_mode=MatchMode(task_cfg.get("match", "contains")),
            toolchain=Toolchain(task_cfg.get("toolchain", "gnu")),
            max_iterations=task_cfg.get("max_iterations", 5),
            simulator_timeout=config.simulator_timeout,
        )

        logger = RunLogger(initial_state.trace_id)
        tracer = AgentTracer(initial_state.trace_id)
        ui = QuietUI()

        final = run_agent(initial_state, backend, config, logger, tracer, ui)
        logger.close()

    total = time.monotonic() - t0

    return {
        "id":         task_id,
        "passed":     final.status == AgentStatus.SUCCESS,
        "status":     final.status.value,
        "iterations": final.iteration,
        "total_s":    total,
        "trace_id":   final.trace_id,
        "final_msg":  final.final_message,
    }


def _build_report(suite_name: str, results: list[dict]) -> dict[str, Any]:
    passed      = [r for r in results if r["passed"]]
    failed      = [r for r in results if not r["passed"]]
    times       = [r["total_s"] for r in results]
    iterations  = [r["iterations"] for r in results]

    def pct(lst, p):
        return statistics.quantiles(lst, n=100)[p - 1] if len(lst) >= 2 else (lst[0] if lst else 0)

    return {
        "suite":        suite_name,
        "total":        len(results),
        "passed":       len(passed),
        "failed":       len(failed),
        "success_rate": len(passed) / len(results) * 100 if results else 0,
        "avg_time_s":   statistics.mean(times) if times else 0,
        "p50_time_s":   pct(times, 50),
        "p95_time_s":   pct(times, 95),
        "avg_iters":    statistics.mean(iterations) if iterations else 0,
        "p50_iters":    pct(iterations, 50),
        "tasks":        results,
    }


def _print_report(report: dict) -> None:
    table = Table(
        title=f"\nEval Suite: {report['suite']}",
        box=box.ROUNDED,
        show_header=True,
    )
    table.add_column("Task", style="cyan")
    table.add_column("Result")
    table.add_column("Iterations", justify="right")
    table.add_column("Time (s)", justify="right")

    for t in report["tasks"]:
        style = "green" if t["passed"] else "red"
        table.add_row(
            t["id"],
            f"[{style}]{'PASS' if t['passed'] else 'FAIL'}[/]",
            str(t["iterations"]),
            f"{t['total_s']:.1f}",
        )

    console.print(table)
    console.print(
        f"\n[bold]Success rate:[/] {report['success_rate']:.1f}%  "
        f"[bold]Avg iters:[/] {report['avg_iters']:.1f}  "
        f"[bold]p50 time:[/] {report['p50_time_s']:.1f}s  "
        f"[bold]p95 time:[/] {report['p95_time_s']:.1f}s"
    )


def _write_report(report: dict, path: str) -> None:
    p = Path(path)
    if p.suffix == ".json":
        p.write_text(json.dumps(report, indent=2))
    else:
        # Markdown report
        lines = [
            f"# Eval Report: {report['suite']}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total tasks | {report['total']} |",
            f"| Passed | {report['passed']} |",
            f"| Failed | {report['failed']} |",
            f"| Success rate | {report['success_rate']:.1f}% |",
            f"| Avg iterations | {report['avg_iters']:.1f} |",
            f"| p50 time | {report['p50_time_s']:.1f}s |",
            f"| p95 time | {report['p95_time_s']:.1f}s |",
            "",
            "## Task Results",
            "",
            "| Task | Result | Iterations | Time (s) | Trace ID |",
            "|------|--------|------------|----------|----------|",
        ]
        for t in report["tasks"]:
            lines.append(
                f"| {t['id']} | {'PASS' if t['passed'] else 'FAIL'} | "
                f"{t['iterations']} | {t['total_s']:.1f} | {t['trace_id'][:8]} |"
            )
        p.write_text("\n".join(lines))
