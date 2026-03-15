"""
LLM Backend abstraction.

Design pattern: Protocol (structural subtyping) rather than ABC.
This means any object with the right methods is a valid backend —
no inheritance required. This is the "duck typing" approach and
keeps backends completely decoupled from each other.

Two backend families:
  API backends   — direct SDK calls, support structured output via instructor
  CLI backends   — subprocess calls to claude/gemini/codex, output parsed from text
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
from pydantic import BaseModel


class BackendResponse(BaseModel):
    """Normalised response from any backend."""
    content: str              # raw text content
    model: str                # model name actually used
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None           # backend-specific raw response object


@runtime_checkable
class LLMBackend(Protocol):
    """
    Protocol that every backend must satisfy.
    Protocols are preferred over ABCs here because:
    - They enable structural typing (mypy/pyright verify conformance without inheritance)
    - They're easier to mock in tests
    - They don't pollute the class hierarchy
    """

    name: str      # e.g. "anthropic-api", "gemini-cli"
    model: str     # e.g. "claude-sonnet-4-6", "gemini-2.5-pro"

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> BackendResponse:
        """Synchronous completion. All backends implement this."""
        ...

    def complete_structured(
        self,
        system: str,
        user: str,
        response_model: type[BaseModel],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> BaseModel:
        """
        Return a validated Pydantic model.
        API backends use instructor for reliability.
        CLI backends parse the text response and validate.
        """
        ...


# ---------------------------------------------------------------------------
# Shared output parser (used by all CLI backends)
# ---------------------------------------------------------------------------

import json
import re


class ResponseParser:
    """
    Extract structured data from free-form LLM text output.
    CLI backends (claude/gemini/codex subprocess calls) return prose;
    this parser finds JSON fences or structured patterns within it.

    Parsing strategy (in order of preference):
      1. ```json ... ``` fenced block
      2. First { ... } JSON object spanning multiple lines
      3. Heuristic field extraction as last resort
    """

    @staticmethod
    def extract_json(text: str) -> dict | None:
        """Extract the first JSON object from text."""
        # Strategy 1: JSON code fence
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            try:
                return json.loads(fence.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 2: largest JSON object (greedy search from outermost braces)
        # Simple brace-matching to avoid grabbing only the first shallow object.
        start = text.find("{")
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break

        return None

    @staticmethod
    def extract_json_array(text: str) -> list | None:
        """Extract the first JSON array from text."""
        # Strategy 1: fenced code block containing an array
        fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if fence:
            try:
                return json.loads(fence.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 2: outermost bare array via bracket-matching
        start = text.find("[")
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break

        return None

    @staticmethod
    def extract_code_files(text: str) -> dict[str, str]:
        """
        Extract filename → code mappings from text containing fenced code blocks.
        Supports patterns like:
          ### path/to/file.c
          ```c
          ... code ...
          ```
        """
        files: dict[str, str] = {}

        # Pattern: heading with filepath followed by a code fence
        pattern = re.compile(
            r"(?:#{1,4}\s*|File:\s*|`?)([^\s`#\n]+\.[a-zA-Z]+)`?\n"
            r"```[a-zA-Z]*\n(.*?)```",
            re.DOTALL,
        )
        for match in pattern.finditer(text):
            path = match.group(1).strip()
            code = match.group(2)
            files[path] = code

        # Fallback: just extract any fenced blocks if no path headers found
        if not files:
            for i, fence in enumerate(re.finditer(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)):
                files[f"output_{i}.c"] = fence.group(1)

        return files

    @classmethod
    def to_pydantic(cls, text: str, model: type[BaseModel]) -> BaseModel:
        """
        Best-effort conversion of LLM text → Pydantic model.
        Used by CLI backends and as the OpenRouter JSON-mode fallback.

        Handles several malformed shapes that LLMs produce:
          - Correct: {"files": [...], "explanation": "..."}
          - Single file dict: {"path": "...", "content": "..."}  ← wrap in list
          - Array at top level: [{"path": "...", "content": "..."}, ...]
          - files key holds a single dict instead of a list
        """
        from r52_types import CodeGenerationOutput, GeneratedFile

        data = cls.extract_json(text)
        if data:
            try:
                return model.model_validate(data)
            except Exception:
                pass

            # Single GeneratedFile dict returned instead of CodeGenerationOutput
            if model is CodeGenerationOutput:
                if "path" in data and "content" in data:
                    return CodeGenerationOutput(
                        files=[GeneratedFile(path=data["path"],
                                             content=data["content"])],
                        explanation="",
                    )
                # files key exists but holds a single dict instead of a list
                raw_files = data.get("files")
                if isinstance(raw_files, dict) and "path" in raw_files:
                    return CodeGenerationOutput(
                        files=[GeneratedFile(path=raw_files["path"],
                                             content=raw_files.get("content", ""))],
                        explanation=data.get("explanation", ""),
                    )

        # Top-level JSON array: [{"path": ..., "content": ...}, ...]
        if model is CodeGenerationOutput:
            arr = cls.extract_json_array(text)
            if arr:
                files = [
                    GeneratedFile(path=item["path"], content=item["content"])
                    for item in arr
                    if isinstance(item, dict) and "path" in item and "content" in item
                ]
                if files:
                    return CodeGenerationOutput(files=files, explanation="")

            # Partial extraction: LLM hit max_tokens mid-JSON — pull whatever
            # complete GeneratedFile objects were emitted before truncation.
            partial = cls._extract_partial_files(text)
            if partial:
                return CodeGenerationOutput(files=partial, explanation="(truncated output)")

            # Last resort: pull code fences with filename headers
            file_dict = cls.extract_code_files(text)
            if file_dict:
                files = [GeneratedFile(path=p, content=c)
                         for p, c in file_dict.items()]
                return CodeGenerationOutput(files=files, explanation=text[:500])

        raise ValueError(f"Cannot parse response as {model.__name__}:\n{text[:300]}")

    @staticmethod
    def _extract_partial_files(text: str) -> list:
        """
        Extract GeneratedFile objects from truncated JSON.

        When the LLM hits max_tokens mid-output, the JSON is incomplete
        (e.g. ``{"files": [{"path": "a.c", "content": "..."}``).
        We use regex to pull out complete path+content pairs, since each
        completed file object has both keys before the truncation point.
        """
        from r52_types import GeneratedFile

        files = []
        # Match complete {"path": "...", "content": "..."} objects within the array.
        # The content value may span many lines, so we match non-greedy up to the
        # closing brace that follows the content string end.
        pattern = re.compile(
            r'\{\s*"path"\s*:\s*"([^"]+)"\s*,\s*"content"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
            re.DOTALL,
        )
        for m in pattern.finditer(text):
            path = m.group(1)
            # Unescape JSON string escape sequences
            try:
                content = json.loads(f'"{m.group(2)}"')
            except json.JSONDecodeError:
                content = m.group(2)
            files.append(GeneratedFile(path=path, content=content))
        return files
