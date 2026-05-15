"""
OpenRouter backend — OpenAI-compatible API giving access to any model
(Gemini, Claude, Llama, Mistral, Qwen, etc.) through a single endpoint.

Default model: "openrouter/free" — auto-routes to whatever free provider
has capacity right now. Individual free models can be rate-limited; the
router handles fallback automatically.

To use a specific free model: --model qwen/qwen3-coder:free
To use a paid model:          --model anthropic/claude-sonnet-4-6

Reasoning models (e.g. DeepSeek-R1, step-3.5-flash) output long <think>
blocks before the actual response. We use a high max_tokens to accommodate
this and strip reasoning content before structured parsing.
"""

from __future__ import annotations

import os
import re
from pydantic import BaseModel
from openai import OpenAI, RateLimitError
import instructor
try:
    from instructor.core import InstructorRetryException
except ImportError:
    from instructor.exceptions import InstructorRetryException  # type: ignore[no-redef]

from .base import BackendResponse, ResponseParser
from observability.tracer import record_llm_request, record_llm_response


OPENROUTER_BASE = "https://openrouter.ai/api/v1"
FREE_ROUTER = "poolside/laguna-xs.2:free" # "qwen/qwen3-coder:free" # "openrouter/free"

# High token limit to accommodate reasoning models' thinking chains.
# Free models often think for 2k-6k tokens before the actual output.
DEFAULT_MAX_TOKENS = 32768


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> and <reasoning>...</reasoning> blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
    return text.strip()


class OpenRouterBackend:
    name = "openrouter"

    def __init__(self, model: str = FREE_ROUTER, **kwargs):
        self.model = model
        api_key = os.environ.get("OPENROUTER_API_KEY") or kwargs.get("api_key")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        self._raw_client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE,
            default_headers={
                "HTTP-Referer": "https://github.com/r52-agent",
                "X-Title": "R52 Coding Agent",
            },
        )
        # Use JSON mode instead of TOOLS mode.
        # Many OpenRouter models (including step-3.5-flash) return multiple tool
        # calls which instructor's TOOLS mode doesn't support. JSON mode embeds
        # the schema in the system prompt and parses the text response instead.
        self._client = instructor.from_openai(
            self._raw_client, mode=instructor.Mode.JSON
        )

    def _with_rate_limit_fallback(self, fn, **kwargs):
        """On RateLimitError for a specific model, retry with the free router."""
        try:
            return fn(**kwargs)
        except RateLimitError:
            if kwargs.get("model") == FREE_ROUTER:
                raise
            kwargs["model"] = FREE_ROUTER
            self.model = FREE_ROUTER
            return fn(**kwargs)

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> BackendResponse:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        def _call(model, **_):
            return self._raw_client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
            )

        record_llm_request(system, user)
        resp = self._with_rate_limit_fallback(_call, model=self.model)
        choice = resp.choices[0]
        content = _strip_reasoning(choice.message.content or "")
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
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> BaseModel:
        record_llm_request(system, user)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Strategy 1: instructor structured output (JSON mode)
        def _call_structured(model, **_):
            return self._client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
                response_model=response_model,
            )

        try:
            result = self._with_rate_limit_fallback(_call_structured, model=self.model)
            record_llm_response(result.model_dump_json(), self.model)
            return result
        except (InstructorRetryException, Exception) as e:
            # Strategy 2: plain completion + text parsing fallback.
            # Reasoning models may exhaust max_tokens on their thinking chain
            # before emitting JSON, causing instructor to fail. We fall back to
            # asking for JSON in the prompt and parsing the text response.
            json_prompt = (
                f"\n\nIMPORTANT: Respond with ONLY a JSON object matching this schema "
                f"(no explanation, no markdown, no thinking tags):\n"
                f"{response_model.model_json_schema()}"
            )
            augmented_messages = messages[:-1] + [
                {"role": "user", "content": messages[-1]["content"] + json_prompt}
            ]

            def _call_plain(model, **_):
                return self._raw_client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=augmented_messages,
                )

            resp = self._with_rate_limit_fallback(_call_plain, model=self.model)
            content = _strip_reasoning(resp.choices[0].message.content or "")
            in_tok = resp.usage.prompt_tokens if resp.usage else 0
            out_tok = resp.usage.completion_tokens if resp.usage else 0
            record_llm_response(content, resp.model, in_tok, out_tok)
            try:
                return ResponseParser.to_pydantic(content, response_model)
            except ValueError as parse_err:
                # Response was unparseable (e.g. truncated due to max_tokens).
                # Surface as InstructorRetryException so callers treat this as a
                # generation failure rather than an unhandled crash.
                total_tok = (
                    (resp.usage.prompt_tokens or 0) + (resp.usage.completion_tokens or 0)
                    if resp.usage else 0
                )
                raise InstructorRetryException(
                    str(parse_err),
                    last_completion=resp,
                    n_attempts=1,
                    messages=augmented_messages,
                    total_usage=total_tok,
                ) from parse_err
