"""
Structured JSONL logger.

Every agent run writes a JSONL file to ~/.r52agent/runs/<trace_id>.jsonl.
Each line is a JSON object with a 'type' field distinguishing event kinds:
  run_start, node_start, node_end, llm_call, build_result, run_result,
  validation_result, run_end

These logs are the primary observability primitive — they can be:
  - Replayed for debugging
  - Aggregated for eval metrics
  - Streamed to any log aggregator (Loki, Splunk, etc.)
"""

from __future__ import annotations

import json
import time
import datetime
from pathlib import Path
from typing import Any


RUNS_DIR = Path.home() / ".r52agent" / "runs"


class RunLogger:
    """
    Writes structured events to a JSONL file.
    One logger instance per agent run (keyed by trace_id).
    """

    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self._path = RUNS_DIR / f"{trace_id}.jsonl"
        self._file = open(self._path, "a", buffering=1)  # line-buffered

    def _write(self, event_type: str, data: dict[str, Any]) -> None:
        record = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "trace_id": self.trace_id,
            "type": event_type,
            **data,
        }
        self._file.write(json.dumps(record) + "\n")

    def run_start(self, task: str, repo: str, backend: str, model: str) -> None:
        self._write("run_start", {
            "task": task, "repo": repo, "backend": backend, "model": model,
        })

    def node_start(self, node: str, iteration: int) -> None:
        self._write("node_start", {"node": node, "iteration": iteration})

    def node_end(self, node: str, iteration: int, duration_s: float) -> None:
        self._write("node_end", {"node": node, "iteration": iteration, "duration_s": duration_s})

    def llm_call(
        self,
        node: str,
        model: str,
        backend: str,
        input_tokens: int,
        output_tokens: int,
        duration_s: float,
    ) -> None:
        self._write("llm_call", {
            "node": node, "model": model, "backend": backend,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "duration_s": duration_s,
        })

    def plan_result(self, iteration: int, build_system: str, is_new_project: bool,
                    files_to_create: list, files_to_modify: list,
                    implementation_steps: list, rationale: str) -> None:
        self._write("plan_result", {
            "iteration": iteration,
            "build_system": build_system,
            "is_new_project": is_new_project,
            "files_to_create": files_to_create,
            "files_to_modify": files_to_modify,
            "implementation_steps": implementation_steps,
            "rationale": rationale,
        })

    def diagnosis_result(self, iteration: int, text: str) -> None:
        self._write("diagnosis_result", {
            "iteration": iteration, "diagnosis": text[:1000],
        })

    def scout_probes(self, iteration: int, probes: list) -> None:
        full = [
            {
                "label": p.get("label", ""),
                "tool": p.get("tool", ""),
                "command": p.get("command"),
                "path": p.get("path"),
                "success": p.get("success", False),
                "output": p.get("output", "")[:2000],
            }
            for p in probes
        ]
        self._write("scout_probes", {"iteration": iteration, "probes": full})

    def diagnose_probes(self, iteration: int, probes: list) -> None:
        full = [
            {
                "label": p.get("label", ""),
                "tool": p.get("tool", ""),
                "command": p.get("command"),
                "path": p.get("path"),
                "success": p.get("success", False),
                "output": p.get("output", "")[:2000],
            }
            for p in probes
        ]
        self._write("diagnose_probes", {"iteration": iteration, "probes": full})

    def review_result(self, iteration: int, approved: bool, issues: list, rejection_count: int) -> None:
        self._write("review_result", {
            "iteration": iteration, "approved": approved,
            "issues": issues[:5], "rejection_count": rejection_count,
        })

    def scout_result(self, iteration: int, fields: int, verified: int, machine: str,
                     field_data: dict | None = None) -> None:
        record: dict = {
            "iteration": iteration, "total_fields": fields,
            "verified_fields": verified, "machine": machine,
        }
        if field_data:
            record["fields"] = {
                k: {"value": v.get("value"), "trust": v.get("trust")}
                for k, v in field_data.items()
            }
        self._write("scout_result", record)

    def build_result(self, iteration: int, success: bool, duration_s: float, stderr_snippet: str) -> None:
        self._write("build_result", {
            "iteration": iteration, "success": success,
            "duration_s": duration_s, "stderr_snippet": stderr_snippet[:500],
        })

    def run_result(self, iteration: int, success: bool, timed_out: bool, duration_s: float, stdout_snippet: str) -> None:
        self._write("run_result", {
            "iteration": iteration, "success": success, "timed_out": timed_out,
            "duration_s": duration_s, "stdout_snippet": stdout_snippet[:500],
        })

    def validation_result(self, iteration: int, passed: bool, detail: str) -> None:
        self._write("validation_result", {
            "iteration": iteration, "passed": passed, "detail": detail,
        })

    def run_end(self, status: str, iterations: int, total_s: float) -> None:
        self._write("run_end", {
            "status": status, "iterations": iterations, "total_s": total_s,
        })

    def close(self) -> None:
        self._file.close()

    @property
    def log_path(self) -> Path:
        return self._path


class NullLogger:
    """No-op logger used when logging is disabled."""
    def __getattr__(self, name: str):
        return lambda *a, **kw: None
    @property
    def log_path(self) -> Path:
        return Path("/dev/null")
