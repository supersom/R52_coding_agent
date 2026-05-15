"""
Backend registry — maps backend name strings to factory functions.

Adding a new backend:
  1. Create backends/my_backend.py implementing LLMBackend protocol
  2. Add an entry to REGISTRY below
"""

from __future__ import annotations

from typing import Callable
from .base import LLMBackend


def _load_anthropic(model: str, **kw) -> LLMBackend:
    from .anthropic_api import AnthropicBackend
    return AnthropicBackend(model=model, **kw)

def _load_openai(model: str, **kw) -> LLMBackend:
    from .openai_api import OpenAIBackend
    return OpenAIBackend(model=model, **kw)

def _load_openrouter(model: str, **kw) -> LLMBackend:
    from .openrouter_api import OpenRouterBackend
    return OpenRouterBackend(model=model, **kw)

def _load_claude_cli(model: str, **kw) -> LLMBackend:
    from .claude_cli import ClaudeCliBackend
    return ClaudeCliBackend(model=model, **kw)

def _load_gemini_cli(model: str, **kw) -> LLMBackend:
    from .gemini_cli import GeminiCliBackend
    return GeminiCliBackend(model=model, **kw)

def _load_codex_cli(model: str, **kw) -> LLMBackend:
    from .codex_cli import CodexCliBackend
    return CodexCliBackend(model=model, **kw)


REGISTRY: dict[str, Callable[..., LLMBackend]] = {
    "anthropic-api": _load_anthropic,
    "openai-api":    _load_openai,
    "openrouter":    _load_openrouter,
    "claude-cli":    _load_claude_cli,
    "gemini-cli":    _load_gemini_cli,
    "codex-cli":     _load_codex_cli,
}

def _openrouter_default() -> str:
    from .openrouter_api import FREE_ROUTER
    return FREE_ROUTER


DEFAULT_MODELS: dict[str, str] = {
    "anthropic-api": "claude-sonnet-4-6",
    "openai-api":    "gpt-4o",
    # To use the auto-router:  --model openrouter/free
    # To use a paid model:     --model anthropic/claude-sonnet-4-6
    "openrouter":    _openrouter_default(),
    "claude-cli":    "claude-sonnet-4-6",
    "gemini-cli":    "gemini-2.5-pro",
    "codex-cli":     "codex-latest",
}


def get_backend(name: str, model: str | None = None, **kwargs) -> LLMBackend:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {list(REGISTRY)}"
        )
    resolved_model = model or DEFAULT_MODELS.get(name, "")
    return REGISTRY[name](model=resolved_model, **kwargs)
