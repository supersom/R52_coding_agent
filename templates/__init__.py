"""
Template manager — discovers, lists, and applies project templates.

Two-stage selection (as agreed):
  1. LLM suggests the best template based on task description
  2. User confirms or overrides via Rich interactive menu
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml
import questionary
from rich.console import Console
from rich.table import Table

from backends.base import LLMBackend


console = Console()

# Built-in templates live next to this file
BUILTIN_TEMPLATES_DIR = Path(__file__).parent

# User templates live in ~/.r52agent/templates/
USER_TEMPLATES_DIR = Path.home() / ".r52agent" / "templates"


def _load_templates() -> list[dict[str, Any]]:
    templates = []
    for search_dir in [BUILTIN_TEMPLATES_DIR, USER_TEMPLATES_DIR]:
        if not search_dir.exists():
            continue
        for manifest in search_dir.rglob("template.yaml"):
            with open(manifest) as f:
                data = yaml.safe_load(f)
            data["_dir"] = str(manifest.parent)
            templates.append(data)
    return templates


def suggest_template(task: str, backend: LLMBackend) -> str | None:
    """Ask the LLM to pick the most suitable template name."""
    templates = _load_templates()
    if not templates:
        return None

    from pydantic import BaseModel

    class TemplateSuggestion(BaseModel):
        name: str
        reason: str

    template_list = "\n".join(
        f"- {t['name']}: {t['description']} (use_when: {t.get('use_when', '')})"
        for t in templates
    )
    suggestion: TemplateSuggestion = backend.complete_structured(
        system="You are an ARM embedded systems expert.",
        user=(
            f"Given this task:\n{task}\n\n"
            f"Pick the most suitable project template from:\n{template_list}\n\n"
            f"Return the template name and a one-sentence reason."
        ),
        response_model=TemplateSuggestion,
    )
    return suggestion.name, suggestion.reason


def interactive_select(task: str, backend: LLMBackend) -> str | None:
    """
    Two-stage template selection:
      1. LLM suggests best match with reasoning
      2. User confirms or picks from list
    """
    templates = _load_templates()
    if not templates:
        return None

    # Stage 1: LLM suggestion
    suggestion_name, reason = suggest_template(task, backend)

    console.print(f"\n[bold cyan]Template Suggestion[/]")
    console.print(f"  Recommended: [bold]{suggestion_name}[/]")
    console.print(f"  Reason: {reason}\n")

    # Stage 2: User confirmation or override
    names = [t["name"] for t in templates]
    descriptions = {t["name"]: t["description"] for t in templates}

    # Build choice strings: "name — description"
    choices = [f"{n} — {descriptions.get(n, '')}" for n in names]
    default_choice = next(
        (c for c in choices if c.startswith(suggestion_name)), choices[0]
    )

    selected = questionary.select(
        "Confirm template (or choose another):",
        choices=choices,
        default=default_choice,
    ).ask()

    if selected is None:
        return None

    return selected.split(" — ")[0]


def apply_template(template_name: str, dest: str) -> list[str]:
    """Copy template files to dest directory. Returns list of copied files."""
    templates = _load_templates()
    tmpl = next((t for t in templates if t["name"] == template_name), None)
    if not tmpl:
        raise ValueError(f"Template not found: {template_name}")

    src = Path(tmpl["_dir"])
    dst = Path(dest)
    dst.mkdir(parents=True, exist_ok=True)

    copied = []
    for f in tmpl.get("files", []):
        src_file = src / f
        if src_file.exists():
            shutil.copy2(src_file, dst / f)
            copied.append(f)

    return copied
