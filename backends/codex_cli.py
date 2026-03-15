"""
OpenAI Codex CLI backend — invokes the `codex` CLI as a subprocess.
Requires OpenAI's codex CLI to be installed and authenticated.
"""

from __future__ import annotations

import subprocess
import shutil
from pydantic import BaseModel

from .base import BackendResponse, ResponseParser
from observability.tracer import record_llm_request, record_llm_response


class CodexCliBackend:
    name = "codex-cli"

    def __init__(self, model: str = "codex-latest", cli_path: str = "codex", **kwargs):
        self.model = model
        self.cli_path = cli_path
        if not shutil.which(cli_path):
            raise RuntimeError(
                f"Codex CLI not found at '{cli_path}'. "
                "Install from: https://github.com/openai/codex"
            )

    def _invoke(self, prompt: str, timeout: int = 300) -> str:
        result = subprocess.run(
            [self.cli_path, "-q", prompt],   # -q = quiet/non-interactive
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Codex CLI failed (rc={result.returncode}): {result.stderr[:500]}"
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
