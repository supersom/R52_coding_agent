"""OpenAI API backend (GPT-4o, o3, etc.) with instructor for structured output."""

from __future__ import annotations

import os
from pydantic import BaseModel
from openai import OpenAI
import instructor

from .base import BackendResponse
from observability.tracer import record_llm_request, record_llm_response


class OpenAIBackend:
    name = "openai-api"

    def __init__(self, model: str = "gpt-4o", **kwargs):
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY") or kwargs.get("api_key")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self._raw_client = OpenAI(api_key=api_key)
        self._client = instructor.from_openai(self._raw_client)

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> BackendResponse:
        record_llm_request(system, user)
        resp = self._raw_client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        in_tok = resp.usage.prompt_tokens if resp.usage else 0
        out_tok = resp.usage.completion_tokens if resp.usage else 0
        record_llm_response(content, resp.model, in_tok, out_tok)
        return BackendResponse(
            content=content,
            model=resp.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            raw=resp,
        )

    def complete_structured(
        self,
        system: str,
        user: str,
        response_model: type[BaseModel],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> BaseModel:
        record_llm_request(system, user)
        result = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_model=response_model,
        )
        record_llm_response(result.model_dump_json(), self.model)
        return result
