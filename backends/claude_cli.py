"""
Claude CLI backend — invokes the `claude` CLI as a subprocess.

The CLI is passed a carefully structured prompt that asks it to respond
in JSON format, then ResponseParser extracts structured data from the output.

Trade-off vs API backend:
  + No API key management in this process (CLI handles auth)
  + Can use Claude's interactive features / slash commands if needed
  - Output parsing is heuristic, not guaranteed structured
  - Slower (subprocess spawn per call)
  - No token count reporting
"""

from __future__ import annotations

import subprocess
import shutil
from pydantic import BaseModel

from .base import BackendResponse, ResponseParser
from observability.tracer import record_llm_request, record_llm_response


class ClaudeCliBackend:
    name = "claude-cli"

    def __init__(self, model: str = "claude-sonnet-4-6", cli_path: str = "claude", **kwargs):
        self.model = model
        self.cli_path = cli_path
        if not shutil.which(cli_path):
            raise RuntimeError(
                f"Claude CLI not found at '{cli_path}'. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )

    def _invoke(self, prompt: str, timeout: int = 300) -> str:
        """Run `claude -p <prompt>` and return stdout."""
        result = subprocess.run(
            [self.cli_path, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Claude CLI failed (rc={result.returncode}): {result.stderr[:500]}"
            )
        return result.stdout

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> BackendResponse:
        # CLI doesn't accept separate system/user — merge them
        full_prompt = f"{system}\n\n---\n\n{user}"
        record_llm_request(system, user)
        content = self._invoke(full_prompt)
        record_llm_response(content, self.model)
        return BackendResponse(content=content, model=self.model)

    def complete_structured(
        self,
        system: str,
        user: str,
        response_model: type[BaseModel],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> BaseModel:
        # Augment user prompt to ask for JSON output
        json_instruction = (
            f"\n\nIMPORTANT: Respond with a single JSON object matching this schema:\n"
            f"{response_model.model_json_schema()}\n"
            f"Wrap the JSON in a ```json code fence."
        )
        full_prompt = f"{system}\n\n---\n\n{user}{json_instruction}"
        record_llm_request(system, user)
        content = self._invoke(full_prompt)
        record_llm_response(content, self.model)
        return ResponseParser.to_pydantic(content, response_model)
