"""
Anthropic API backend using the Anthropic SDK + instructor for structured output.

instructor patches the Anthropic client to add .messages.create(..., response_model=...)
which automatically retries on validation failure — you get a valid Pydantic object or
an exception after max_retries.
"""

from __future__ import annotations

import os
from pydantic import BaseModel
import anthropic
import instructor

from .base import BackendResponse, ResponseParser
from observability.tracer import record_llm_request, record_llm_response


class AnthropicBackend:
    name = "anthropic-api"

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs):
        self.model = model
        api_key = os.environ.get("ANTHROPIC_API_KEY") or kwargs.get("api_key")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self._raw_client = anthropic.Anthropic(api_key=api_key)
        # instructor patches the client to support response_model= parameter
        self._client = instructor.from_anthropic(self._raw_client)

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> BackendResponse:
        record_llm_request(system, user)
        msg = self._raw_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        content = msg.content[0].text
        record_llm_response(content, msg.model,
                            msg.usage.input_tokens, msg.usage.output_tokens)
        return BackendResponse(
            content=content,
            model=msg.model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            raw=msg,
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
        result, completion = self._client.messages.create_with_completion(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            response_model=response_model,
        )
        record_llm_response(
            result.model_dump_json(),
            completion.model,
            completion.usage.input_tokens,
            completion.usage.output_tokens,
        )
        return result
