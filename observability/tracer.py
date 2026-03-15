"""
OpenTelemetry tracing integration (optional).

When OTEL_ENDPOINT is set, traces are exported to that endpoint (e.g. Jaeger).
When not set, a no-op tracer is used — zero overhead, no configuration needed.

What you see in Jaeger per run:
  Root span: "r52-run"
  ├── plan       — LLM prompt sent, plan JSON received
  ├── generate   — LLM prompt sent, generated files (path + content)
  ├── review     — LLM prompt sent, review decision + issues
  ├── build      — compiler command, full stdout/stderr
  ├── run        — simulator command, full stdout/stderr
  ├── validate   — expected vs actual output
  └── patch      — failure summary sent to LLM, patched files received

Each span carries:
  Attributes — short key/value metadata (iteration, success, model, token counts)
  Events     — timestamped payloads for large data (prompts, responses, file contents)
               Visible in Jaeger under the "Logs" tab of each span.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator, Any

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# OTel attribute values are limited to 64KB by most backends.
# Truncate large strings to avoid silent drops.
_MAX_ATTR = 32_000


def _t(s: str) -> str:
    """Truncate a string to fit OTel attribute limits."""
    if len(s) > _MAX_ATTR:
        return s[:_MAX_ATTR] + f"\n... [truncated, {len(s)} chars total]"
    return s


def _setup_provider(endpoint: str) -> "TracerProvider | None":
    if not OTEL_AVAILABLE or not endpoint:
        return None
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        resource = Resource.create({"service.name": "r52-coding-agent"})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        return provider
    except Exception:
        return None


class SpanHandle:
    """
    Wraps an OTel span with convenience methods for recording
    LLM calls, generated files, build results, etc.
    """

    def __init__(self, span):
        self._span = span

    def set(self, key: str, value: Any) -> None:
        if self._span:
            self._span.set_attribute(key, str(value)[:_MAX_ATTR])

    def event(self, name: str, **fields) -> None:
        """
        Add a timestamped event to this span.
        In Jaeger: visible under the 'Logs' tab of the span.
        """
        if self._span:
            self._span.add_event(
                name,
                attributes={k: _t(str(v)) for k, v in fields.items()},
            )

    def llm_request(self, system: str, user: str, node: str) -> None:
        """Record what was sent to the LLM."""
        self.event("llm.request",
                   node=node,
                   system_prompt=system,
                   user_prompt=user)

    def llm_response(self, content: str, model: str,
                     input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record what the LLM returned."""
        self.event("llm.response",
                   model=model,
                   content=content,
                   input_tokens=str(input_tokens),
                   output_tokens=str(output_tokens))

    def generated_files(self, files: dict[str, str]) -> None:
        """Record every file the LLM generated."""
        for path, content in files.items():
            self.event("generated_file", path=path, content=content)

    def build_result(self, command: str, stdout: str, stderr: str,
                     success: bool, duration_s: float) -> None:
        self.event("build.result",
                   command=command,
                   success=str(success),
                   duration_s=str(duration_s),
                   stdout=stdout,
                   stderr=stderr)

    def run_result(self, stdout: str, stderr: str,
                   success: bool, timed_out: bool, duration_s: float) -> None:
        self.event("run.result",
                   success=str(success),
                   timed_out=str(timed_out),
                   duration_s=str(duration_s),
                   stdout=stdout,
                   stderr=stderr)

    def validation_result(self, passed: bool, expected: str,
                          actual: str, detail: str) -> None:
        self.event("validation.result",
                   passed=str(passed),
                   expected=expected,
                   actual=actual,
                   detail=detail)


class _NoopHandle:
    """No-op handle used when OTel is disabled."""
    def set(self, *a, **kw): pass
    def event(self, *a, **kw): pass
    def llm_request(self, *a, **kw): pass
    def llm_response(self, *a, **kw): pass
    def generated_files(self, *a, **kw): pass
    def build_result(self, *a, **kw): pass
    def run_result(self, *a, **kw): pass
    def validation_result(self, *a, **kw): pass


class AgentTracer:
    """
    Thin wrapper around the OTel tracer.
    Falls back to no-op when OTel is unavailable/unconfigured.
    """

    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        endpoint = os.environ.get("OTEL_ENDPOINT", "")
        self._provider = _setup_provider(endpoint)
        self._enabled = self._provider is not None
        self._tracer = trace.get_tracer("r52-agent") if (self._enabled and OTEL_AVAILABLE) else None

    @contextmanager
    def span(self, name: str, **attributes) -> Generator[SpanHandle, None, None]:
        if self._tracer is None:
            yield _NoopHandle()
            return
        with self._tracer.start_as_current_span(name) as otel_span:
            otel_span.set_attribute("trace_id", self.trace_id)
            for k, v in attributes.items():
                otel_span.set_attribute(k, str(v)[:_MAX_ATTR])
            yield SpanHandle(otel_span)

    def shutdown(self) -> None:
        if self._provider:
            self._provider.shutdown()


# ---------------------------------------------------------------------------
# Module-level helpers — call from any backend without passing a tracer
# ---------------------------------------------------------------------------

def record_llm_request(system: str, user: str, node: str = "") -> None:
    """
    Add an llm.request event to the *current* active OTel span.

    Backends call this just before sending the prompt to the LLM.
    Because LangGraph node wrappers open the span with `tracer.span(...)`,
    the span is already current on the thread when the backend is called,
    so no tracer reference needs to be threaded through.
    """
    if not OTEL_AVAILABLE:
        return
    span = trace.get_current_span()
    if span and span.is_recording():
        span.add_event("llm.request", attributes={
            "node": _t(node),
            "system_prompt": _t(system),
            "user_prompt": _t(user),
        })


def record_llm_response(
    content: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """
    Add an llm.response event to the *current* active OTel span.

    Backends call this immediately after receiving the LLM response.
    Visible in Jaeger under the 'Logs' tab of each node span.
    """
    if not OTEL_AVAILABLE:
        return
    span = trace.get_current_span()
    if span and span.is_recording():
        span.add_event("llm.response", attributes={
            "model": _t(model),
            "content": _t(content),
            "input_tokens": str(input_tokens),
            "output_tokens": str(output_tokens),
        })
