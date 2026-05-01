"""Async LLM client wrapper with structured output and retries.

Default backend is Anthropic (Haiku/Sonnet) because the user's Gemini free
tier hit a daily-quota wall at 20 RPD. Backend selectable via HRO_BACKEND
in the env (`anthropic` or `gemini`).

Structured output:
  - Anthropic: tool-use with tool_choice forcing the tool. The Pydantic
    model_json_schema is the tool input_schema; the model is required to
    invoke the tool and we parse `tool_use.input` into the Pydantic model.
  - Gemini: native responseSchema (only when HRO_BACKEND=gemini).

Retries (per code/HANDOFF.md error matrix):
  - 429 / RateLimit: honor server-provided retry hint or exponential backoff,
    up to 5 retries.
  - 5xx / overloaded: jittered backoff, up to 3 retries.
  - schema validation failure: one retry with the validation error appended.
  - timeout: one retry, then surface.

A global RateLimiter (see hro.llm.rate_limiter) gates calls so we don't
trip rate limits in the first place.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
from typing import Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from hro.config import (
    ANTHROPIC_HAIKU,
    ANTHROPIC_SONNET,
    CODE_DIR,
    GEMINI_MODEL,
    LLM_BACKEND,
    REPO_ROOT,
)
from hro.llm.rate_limiter import get_limiter

# Load keys regardless of shell cwd (hackathon runners often invoke from repo root).
load_dotenv(REPO_ROOT / ".env")
load_dotenv(CODE_DIR / ".env")
load_dotenv()

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Wraps any backend error after retries are exhausted."""


# ---------------------------------------------------------------------------
# Public client. Backend-agnostic surface.
# ---------------------------------------------------------------------------


class LLMClient:
    """Backend-agnostic async client. Picks Anthropic or Gemini at construction
    time based on `HRO_BACKEND`.
    """

    def __init__(self, backend: str | None = None) -> None:
        backend = (backend or LLM_BACKEND).lower()
        self.backend = backend
        if backend == "anthropic":
            self._impl: _Backend = _AnthropicBackend()
        elif backend == "gemini":
            self._impl = _GeminiBackend()
        else:
            raise LLMError(f"Unknown HRO_BACKEND={backend!r}; expected 'anthropic' or 'gemini'.")
        # Default model for plain-text calls. Specialists/triage override per-call.
        self.model = self._impl.default_model

    async def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        thinking_level: str = "minimal",
        system_instruction: str | None = None,
        temperature: float = 0.0,
        model: str | None = None,
        cached_content: str | None = None,  # Gemini-only; ignored on Anthropic
    ) -> T:
        model_name = model or self._impl.default_model

        last_err: Exception | None = None
        attempts = 0
        max_transient = 3
        max_429 = 5
        used_schema_retry = False
        current_prompt = prompt

        while True:
            attempts += 1
            await get_limiter().acquire()
            try:
                obj = await self._impl.generate_structured(
                    prompt=current_prompt,
                    response_schema=response_schema,
                    thinking_level=thinking_level,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    model=model_name,
                    cached_content=cached_content,
                )
                return obj

            except ValidationError as ve:
                if used_schema_retry:
                    raise LLMError(f"Structured output failed schema twice: {ve}") from ve
                used_schema_retry = True
                current_prompt = (
                    f"{prompt}\n\n"
                    f"PREVIOUS RESPONSE FAILED SCHEMA VALIDATION:\n{ve}\n\n"
                    f"Return ONLY a JSON object that satisfies the schema. No commentary."
                )
                continue

            except _BackendRateLimit as rl:
                if attempts <= max_429:
                    delay = rl.retry_after if rl.retry_after else _backoff(attempts)
                    last_err = rl
                    await asyncio.sleep(min(delay + random.uniform(0.0, 0.5), 90.0))
                    continue
                raise LLMError(f"Rate limited after {attempts} attempts: {rl}") from rl

            except _BackendTransient as te:
                if attempts <= max_transient:
                    last_err = te
                    await asyncio.sleep(_backoff(attempts))
                    continue
                raise LLMError(f"Transient error after {attempts} attempts: {te}") from te

            except _BackendBadRequest as bre:
                # 4xx other than 429 — surface immediately.
                raise LLMError(f"Bad request: {bre}") from bre

    async def generate_text(
        self,
        prompt: str,
        thinking_level: str = "minimal",
        system_instruction: str | None = None,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str:
        await get_limiter().acquire()
        return await self._impl.generate_text(
            prompt=prompt,
            thinking_level=thinking_level,
            system_instruction=system_instruction,
            temperature=temperature,
            model=model or self._impl.default_model,
        )


# Process-singleton.
_default_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


# ---------------------------------------------------------------------------
# Backend-specific exceptions used as a stable abstraction layer.
# ---------------------------------------------------------------------------


class _BackendError(Exception):
    pass


class _BackendRateLimit(_BackendError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class _BackendTransient(_BackendError):
    """5xx, overloaded, timeout."""


class _BackendBadRequest(_BackendError):
    """4xx other than 429."""


# ---------------------------------------------------------------------------
# Backend protocol.
# ---------------------------------------------------------------------------


class _Backend:
    default_model: str

    async def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        thinking_level: str,
        system_instruction: str | None,
        temperature: float,
        model: str,
        cached_content: str | None,
    ) -> T:
        raise NotImplementedError

    async def generate_text(
        self,
        prompt: str,
        thinking_level: str,
        system_instruction: str | None,
        temperature: float,
        model: str,
    ) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Anthropic backend.
# ---------------------------------------------------------------------------


_ANTHROPIC_THINKING_BUDGET = {
    "minimal": 0,
    "low": 1024,
    "medium": 4096,
    "high": 16000,
}


def _supports_thinking(model: str) -> bool:
    """Haiku does NOT support extended thinking. Sonnet/Opus do."""
    return not model.startswith("claude-haiku")


class _AnthropicBackend(_Backend):
    default_model = ANTHROPIC_HAIKU

    def __init__(self) -> None:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY not set. Add it to .env or export it in your shell."
            )
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._anthropic = anthropic

    async def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        thinking_level: str,
        system_instruction: str | None,
        temperature: float,
        model: str,
        cached_content: str | None,  # ignored
    ) -> T:
        kwargs = self._build_messages_kwargs(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            model=model,
            thinking_level=thinking_level,
        )
        kwargs["tools"] = [
            {
                "name": "submit_decision",
                "description": "Submit the structured decision for this ticket.",
                "input_schema": response_schema.model_json_schema(),
            }
        ]
        kwargs["tool_choice"] = {"type": "tool", "name": "submit_decision"}

        try:
            resp = await self._client.messages.create(**kwargs)
        except self._anthropic.RateLimitError as rl:
            raise _BackendRateLimit(str(rl), retry_after=_anthropic_retry_after(rl)) from rl
        except (self._anthropic.APITimeoutError, self._anthropic.APIConnectionError, asyncio.TimeoutError) as te:
            raise _BackendTransient(str(te)) from te
        except self._anthropic.APIStatusError as se:
            code = getattr(se, "status_code", None)
            if code and 500 <= code < 600:
                raise _BackendTransient(str(se)) from se
            raise _BackendBadRequest(str(se)) from se

        # Extract the tool_use block.
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "submit_decision":
                return response_schema.model_validate(block.input)
        raise _BackendBadRequest(
            "Anthropic response had no tool_use block. "
            f"stop_reason={getattr(resp, 'stop_reason', '?')}"
        )

    async def generate_text(
        self,
        prompt: str,
        thinking_level: str,
        system_instruction: str | None,
        temperature: float,
        model: str,
    ) -> str:
        kwargs = self._build_messages_kwargs(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            model=model,
            thinking_level=thinking_level,
        )
        try:
            resp = await self._client.messages.create(**kwargs)
        except self._anthropic.RateLimitError as rl:
            raise _BackendRateLimit(str(rl), retry_after=_anthropic_retry_after(rl)) from rl
        except (self._anthropic.APITimeoutError, self._anthropic.APIConnectionError) as te:
            raise _BackendTransient(str(te)) from te
        except self._anthropic.APIStatusError as se:
            code = getattr(se, "status_code", None)
            if code and 500 <= code < 600:
                raise _BackendTransient(str(se)) from se
            raise _BackendBadRequest(str(se)) from se
        out: list[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                out.append(block.text)
        return "".join(out)

    @staticmethod
    def _build_messages_kwargs(
        prompt: str,
        system_instruction: str | None,
        temperature: float,
        model: str,
        thinking_level: str,
    ) -> dict:
        kwargs: dict = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if system_instruction:
            kwargs["system"] = system_instruction
        # Extended thinking is supported on Sonnet/Opus only. When enabled,
        # Anthropic requires temperature=1.
        if _supports_thinking(model):
            budget = _ANTHROPIC_THINKING_BUDGET.get(thinking_level.lower(), 0)
            if budget > 0:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                kwargs["temperature"] = 1.0
                kwargs["max_tokens"] = max(kwargs["max_tokens"], budget + 2048)
        return kwargs


def _anthropic_retry_after(err: Exception) -> float | None:
    """Best-effort extract Retry-After header from an Anthropic RateLimitError."""
    resp = getattr(err, "response", None)
    if resp is None:
        return None
    try:
        ra = resp.headers.get("retry-after")
        if ra is None:
            return None
        return float(ra)
    except (AttributeError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Gemini backend (fallback path; not the default).
# ---------------------------------------------------------------------------


_GEMINI_THINKING_BUDGETS = {"minimal": 0, "low": 512, "medium": 4096, "high": 16384}


class _GeminiBackend(_Backend):
    default_model = GEMINI_MODEL

    def __init__(self) -> None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMError("GEMINI_API_KEY not set.")
        self._client = genai.Client(api_key=api_key)
        from google.genai import errors as genai_errors
        from google.genai import types

        self._genai = genai
        self._errors = genai_errors
        self._types = types

    async def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        thinking_level: str,
        system_instruction: str | None,
        temperature: float,
        model: str,
        cached_content: str | None,
    ) -> T:
        types = self._types
        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=42,
            responseMimeType="application/json",
            responseSchema=response_schema,
            thinkingConfig=types.ThinkingConfig(
                thinkingBudget=_GEMINI_THINKING_BUDGETS.get(thinking_level.lower(), 0)
            ),
            systemInstruction=system_instruction,
            cachedContent=cached_content,
        )
        try:
            resp = await self._client.aio.models.generate_content(
                model=model, contents=prompt, config=config
            )
        except self._errors.ClientError as ce:
            code = getattr(ce, "code", None) or _infer_status(ce)
            if code == 429:
                raise _BackendRateLimit(str(ce), retry_after=_retry_delay_seconds(str(ce))) from ce
            raise _BackendBadRequest(str(ce)) from ce
        except self._errors.ServerError as se:
            raise _BackendTransient(str(se)) from se

        if resp.parsed is not None and isinstance(resp.parsed, response_schema):
            return resp.parsed
        if resp.text:
            return response_schema.model_validate_json(resp.text)
        raise _BackendBadRequest("Gemini returned an empty response.")

    async def generate_text(
        self,
        prompt: str,
        thinking_level: str,
        system_instruction: str | None,
        temperature: float,
        model: str,
    ) -> str:
        types = self._types
        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=42,
            thinkingConfig=types.ThinkingConfig(
                thinkingBudget=_GEMINI_THINKING_BUDGETS.get(thinking_level.lower(), 0)
            ),
            systemInstruction=system_instruction,
        )
        resp = await self._client.aio.models.generate_content(
            model=model, contents=prompt, config=config
        )
        return resp.text or ""


def _backoff(attempt: int) -> float:
    base = 0.5 * (2 ** (attempt - 1))
    return min(base + random.uniform(0.0, 0.5), 8.0)


def _infer_status(err: Exception) -> int | None:
    msg = str(err)
    for code in (429, 500, 502, 503, 504):
        if str(code) in msg:
            return code
    return None


_RETRY_DELAY_RE = re.compile(r"'retryDelay'\s*:\s*'(\d+(?:\.\d+)?)s'")


def _retry_delay_seconds(message: str) -> float | None:
    m = _RETRY_DELAY_RE.search(message)
    if not m:
        return None
    try:
        return min(float(m.group(1)), 60.0)
    except ValueError:
        return None
