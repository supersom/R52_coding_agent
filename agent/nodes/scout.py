"""
SCOUT node — LLM-directed hardware model extraction.

Design principle: the LLM figures out how to investigate the hardware at runtime.
SCOUT provides two primitive tools — run_command and read_file — and gets out
of the way. The LLM uses its knowledge of the simulator to decide what to run:
  - QEMU: QMP via qemu-system-arm -qmp stdio, or grepping source files
  - FVP:  FVP binary --list-params, or reading DTS files
  - Renode: reading .repl platform description files
  - Real hardware: reading SVD files
No simulator-specific logic lives here. If a new simulator is introduced, SCOUT
adapts without code changes.

Flow:
  Phase 1 — The LLM reads the plan and simulator name and produces an
             InvestigationPlan: an ordered list of commands/files to probe.
  Phase 2 — Python executes each probe deterministically and collects outputs.
  Phase 3 — The LLM reads raw probe outputs and produces a HardwareModel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.probe_tools import run_command, read_file
from agent.state import AgentState
from backends.base import LLMBackend


# ---------------------------------------------------------------------------
# Trust levels
# ---------------------------------------------------------------------------

_TRUST_LABEL = {
    "runtime": "VERIFIED  (live probe)",
    "source":  "VERIFIED  (source/file)",
    "prior":   "UNVERIFIED (LLM prior)",
}


# ---------------------------------------------------------------------------
# Hardware model  — consumed by all downstream nodes
# ---------------------------------------------------------------------------

class HardwareField(BaseModel):
    value: str
    trust: str    # "runtime" | "source" | "prior"
    source: str   # which probe produced this, e.g. "memory_map" or "uart_irq_grep"
    evidence: str # the exact output line(s) this value was read from


class HardwareModel(BaseModel):
    machine: str
    fields: dict[str, HardwareField]  # key = "Peripheral.field"


# ---------------------------------------------------------------------------
# Phase 1 — investigation plan
# ---------------------------------------------------------------------------

class ProbeSpec(BaseModel):
    label: str = Field(description="Short identifier referenced in Phase 3, e.g. 'memory_map'.")
    purpose: str = Field(description="What hardware information this probe yields.")
    tool: Literal["run_command", "read_file"]
    command: str | None = Field(
        default=None,
        description="Shell command for run_command probes."
    )
    path: str | None = Field(
        default=None,
        description="File path for read_file probes."
    )


class InvestigationPlan(BaseModel):
    reasoning: str = Field(
        description="How you assessed the simulator and why you chose these probes."
    )
    probes: list[ProbeSpec]


# ---------------------------------------------------------------------------
# Phase 3 — synthesis
# ---------------------------------------------------------------------------

class SynthesisField(BaseModel):
    key: str = Field(description="'Peripheral.field', e.g. 'UART0.base', 'BRAM.top'.")
    value: str
    trust: Literal["runtime", "source", "prior"]
    source_label: str = Field(description="The probe label this came from.")
    evidence: str = Field(description="Exact line(s) from probe output.")


class HardwareModelSynthesis(BaseModel):
    fields: list[SynthesisField]


# ---------------------------------------------------------------------------
# Probe deny list — paths the LLM must not read via probes.
# Supports prefix matching (directory) and exact file matching.
# Paths are matched against the resolved absolute path so relative tricks
# like "../../docs" are still caught.
# ---------------------------------------------------------------------------

_PROBE_DENY: list[str] = [
    "docs/",    # project docs — human notes, may bias hardware model synthesis
    ".git/",    # git internals
]


def _is_denied(path: str) -> bool:
    resolved = str(Path(path).resolve())
    # Also check the raw path so relative paths like "docs/foo.md" match.
    for denied in _PROBE_DENY:
        denied_abs = str(Path(denied).resolve())
        if resolved.startswith(denied_abs) or path.startswith(denied):
            return True
    return False


def _execute_probe(probe: ProbeSpec) -> tuple[str, bool]:
    if probe.tool == "run_command" and probe.command:
        return run_command(probe.command)
    if probe.tool == "read_file" and probe.path:
        if _is_denied(probe.path):
            return f"(access denied: {probe.path})", False
        return read_file(probe.path)
    return "(probe misconfigured — missing command or path)", False


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_PHASE1_SYSTEM = """
You are a hardware investigator for bare-metal firmware development.

Your job: given a firmware implementation plan and target simulator, figure out
what probes to run to collect authoritative hardware information.

You have exactly two tools:

  run_command(command)
    Execute any read-only shell command and return its output.
    Use your knowledge of the simulator to choose the right commands.
    Examples of what you might generate depending on simulator:
      - QEMU machines: use the QEMU monitor or QMP protocol to query the
        live machine. You know how to do this.
      - ARM FVP: invoke the FVP binary with --list-params or similar flags.
      - dtc / device tree: run dtc to decompile a DTB file.
      - Grep: use grep to search simulator source or config files.

  read_file(path)
    Read a file and return its full contents.
    Use for static descriptions of hardware that don't need a running process:
    device tree source (.dts/.dtsi), SVD files, Renode .repl platform files,
    FVP JSON configs, QEMU source files, etc.

Strategy:
  - Use a live probe first (run_command on the simulator) to get authoritative
    runtime values — base addresses and region sizes come directly from the emulator.
  - Follow up with file reads or grep for values not visible in live output
    (interrupt IDs, register offsets within a peripheral, etc.).
  - Always probe for BOTH base addresses AND sizes of memory regions.
    stack_top = base + size. This is safety-critical — base alone is not enough.
"""

_PHASE1_USER = """
## Firmware implementation plan
{plan_text}

## Target simulator / machine
{simulator}

Produce an InvestigationPlan with the probes needed to fully characterise the
hardware this plan requires.
"""

_PHASE3_SYSTEM = """
You are synthesising hardware investigation results into a structured HardwareModel.

Read the probe outputs and extract every hardware value the firmware needs.

Trust levels:
  "runtime" — value came from a live process (run_command on the simulator).
              Highest trust. Use this for addresses and sizes from the live machine.
  "source"  — value came from reading a source file, DTS, SVD, or similar.
  "prior"   — not found in any probe. Use your own knowledge as last resort.
              Always mark this explicitly so downstream code treats it with caution.

For every memory region (RAM, ROM, TCM, SRAM, DDR, BRAM, etc.) include:
  Region.base  — first byte address
  Region.size  — total bytes
  Region.top   — base + size  ← this is the correct initial stack pointer value

Never use Region.base as a stack pointer value — `push` decrements SP before
writing, so SP = base writes to base-8, which is before the region → Data Abort.
"""

_PHASE3_USER = """
## Firmware implementation plan
{plan_text}

## Probe results
{probe_results}

Produce a HardwareModelSynthesis.
"""


def _format_plan(plan: dict) -> str:
    parts: list[str] = []
    for step in plan.get("implementation_steps", []):
        parts.append(f"- {step}")
    syms = plan.get("new_symbols", [])
    if syms:
        parts.append(f"New symbols: {', '.join(syms)}")
    rat = plan.get("rationale", "")
    if rat:
        parts.append(f"Rationale: {rat}")
    return "\n".join(parts) or "(no plan yet)"


def _format_probe_results(results: list[dict]) -> str:
    parts = []
    for r in results:
        status = "OK" if r["success"] else "FAILED"
        parts.append(
            f"=== {r['label']} [{r['tool']}] [{status}] ===\n"
            f"Purpose: {r['purpose']}\n"
            f"{r['output']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_scout(state: AgentState, backend: LLMBackend) -> AgentState:
    """
    Phase 1: LLM decides what to probe (simulator-agnostic).
    Phase 2: Execute probes deterministically.
    Phase 3: LLM synthesises raw results into HardwareModel.
    """
    plan = state.repo_context.get("plan", {})
    plan_text = _format_plan(plan)

    # Phase 1 — strategy
    inv_plan: InvestigationPlan = backend.complete_structured(
        system=_PHASE1_SYSTEM,
        user=_PHASE1_USER.format(
            plan_text=plan_text,
            simulator=state.simulator.value,
        ),
        response_model=InvestigationPlan,
        temperature=0.0,
    )

    # Phase 2 — execution
    probe_results: list[dict] = []
    for probe in inv_plan.probes:
        output, success = _execute_probe(probe)
        probe_results.append({
            "label": probe.label,
            "tool": probe.tool,
            "purpose": probe.purpose,
            "output": output[:8000],
            "success": success,
        })

    # Phase 3 — synthesis
    synthesis: HardwareModelSynthesis = backend.complete_structured(
        system=_PHASE3_SYSTEM,
        user=_PHASE3_USER.format(
            plan_text=plan_text,
            probe_results=_format_probe_results(probe_results),
        ),
        response_model=HardwareModelSynthesis,
        temperature=0.0,
    )

    fields: dict[str, HardwareField] = {
        sf.key: HardwareField(
            value=sf.value,
            trust=sf.trust,
            source=sf.source_label,
            evidence=sf.evidence,
        )
        for sf in synthesis.fields
    }

    hw_model = HardwareModel(machine=state.simulator.value, fields=fields)
    new_ctx = {
        **state.repo_context,
        "hardware_model": hw_model.model_dump(),
        "scout_probe_results": probe_results,
    }
    return state.model_copy(update={"repo_context": new_ctx})


def format_hardware_model(hw: dict[str, Any]) -> str:
    """Format the hardware model for injection into LLM prompts."""
    if not hw or not hw.get("fields"):
        return ""

    by_peripheral: dict[str, list[tuple[str, dict]]] = {}
    for key, field in hw["fields"].items():
        peripheral, _, field_name = key.partition(".")
        by_peripheral.setdefault(peripheral, []).append((field_name or key, field))

    lines = [f"Machine: {hw.get('machine', '?')}\n"]
    for peripheral, fields in sorted(by_peripheral.items()):
        lines.append(f"{peripheral}:")
        for field_name, field in fields:
            label = _TRUST_LABEL.get(field.get("trust", ""), field.get("trust", ""))
            lines.append(
                f"  {field_name:<22} {field.get('value', '?'):<18}"
                f"  [{label}  {field.get('source', '')}]"
            )
        lines.append("")

    return "\n".join(lines)
