"""
Rich terminal UI.

Uses Rich's Live display to show a real-time dashboard during an agent run:
  ┌─────────────────────────────────────────────────────┐
  │  R52 Coding Agent          [iteration 2/5]          │
  │  Phase: BUILD              trace: abc123...         │
  ├─────────────────────────────────────────────────────┤
  │  Task: "Add UART TX function"                       │
  ├─────────────────────────────────────────────────────┤
  │  BUILD stderr:                                      │
  │  > main.c:42: error: undeclared identifier 'uart'  │
  └─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box


console = Console()

PHASE_COLORS = {
    "PLAN":     "cyan",
    "GENERATE": "blue",
    "REVIEW":   "magenta",
    "BUILD":    "yellow",
    "RUN":      "green",
    "VALIDATE": "green",
    "PATCH":    "red",
    "DONE":     "bright_green",
    "FAILED":   "bright_red",
}


class AgentUI:
    """
    Live dashboard for one agent run.
    Call .update() from each graph node to reflect current state.
    """

    def __init__(self, task: str, trace_id: str, max_iterations: int, verbose: bool = False):
        self.task = task[:80] + ("..." if len(task) > 80 else "")
        self.trace_id = trace_id[:8]
        self.max_iterations = max_iterations
        self.verbose = verbose
        self._phase = "PLAN"
        self._iteration = 0
        self._detail_lines: list[str] = []
        self._live: Live | None = None

    def _make_panel(self) -> Panel:
        color = PHASE_COLORS.get(self._phase, "white")

        header = Table.grid(expand=True)
        header.add_column(ratio=3)
        header.add_column(justify="right", ratio=1)
        header.add_row(
            Text(f"R52 Coding Agent", style="bold white"),
            Text(
                f"iter {self._iteration}/{self.max_iterations}  id:{self.trace_id}",
                style="dim",
            ),
        )

        phase_line = Text(f"Phase: {self._phase}", style=f"bold {color}")
        task_line  = Text(f"Task:  {self.task}", style="italic")

        detail_text = "\n".join(self._detail_lines[-20:]) if self._detail_lines else ""

        body = Table.grid()
        body.add_row(phase_line)
        body.add_row(task_line)
        if detail_text:
            body.add_row(Text(""))
            body.add_row(Text(detail_text, style="dim"))

        return Panel(body, title=str(header), box=box.ROUNDED, border_style=color)

    def start(self) -> "AgentUI":
        self._live = Live(self._make_panel(), refresh_per_second=4, console=console)
        self._live.__enter__()
        return self

    def stop(self) -> None:
        if self._live:
            self._live.__exit__(None, None, None)
            self._live = None

    def update(
        self,
        phase: str | None = None,
        iteration: int | None = None,
        detail: str | None = None,
    ) -> None:
        if phase:
            self._phase = phase
        if iteration is not None:
            self._iteration = iteration
        if detail:
            for line in detail.splitlines():
                self._detail_lines.append(line)
        if self._live:
            self._live.update(self._make_panel())

    def print(self, msg: str, style: str = "") -> None:
        """Print a persistent message above the live display."""
        if self._live:
            self._live.console.print(msg, style=style)
        else:
            console.print(msg, style=style)

    def success(self, msg: str) -> None:
        self.update(phase="DONE")
        self.stop()
        console.print(f"\n[bold bright_green]✓ {msg}[/]")

    def failure(self, msg: str) -> None:
        self.update(phase="FAILED")
        self.stop()
        console.print(f"\n[bold bright_red]✗ {msg}[/]")


class QuietUI:
    """Minimal stdout-only UI for non-interactive / piped use."""

    def __init__(self, *a, **kw): pass
    def start(self): return self
    def stop(self): pass
    def update(self, phase=None, iteration=None, detail=None):
        if phase:
            print(f"[{phase}]" + (f" iter {iteration}" if iteration else ""))
        if detail:
            print(detail)
    def print(self, msg, style=""): print(msg)
    def success(self, msg): print(f"✓ {msg}")
    def failure(self, msg): print(f"✗ {msg}")
