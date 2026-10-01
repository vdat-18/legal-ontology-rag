"""Shared non-streaming HTTP transport for generated BAML requests."""

from __future__ import annotations

import asyncio
import http.client
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from http import HTTPStatus
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from baml_py import ClientRegistry, HTTPRequest
from baml_py.logging import set_log_level
from loguru import logger

if TYPE_CHECKING:
    from legal_rag.settings import Settings

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


@dataclass(slots=True)
class LLMUsage:
    """Thread-safe token accounting for the LLM calls made in one evaluation context."""

    call_count: int = 0
    usage_available_call_count: int = 0
    usage_unavailable_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    models: dict[str, int] = field(default_factory=dict)
    partial: bool = False
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def record(self, response: Mapping[str, Any]) -> None:
        """Add standard OpenAI-compatible usage fields from one successful response."""
        model = response.get("model")
        model_name = model if isinstance(model, str) and model else "unknown"
        usage = response.get("usage")
        with self._lock:
            self.call_count += 1
            self.models[model_name] = self.models.get(model_name, 0) + 1
            if not isinstance(usage, Mapping):
                self.usage_unavailable_call_count += 1
                self.partial = True
                return

            input_tokens = _optional_token_count(usage.get("prompt_tokens", usage.get("input_tokens")))
            output_tokens = _optional_token_count(usage.get("completion_tokens", usage.get("output_tokens")))
            total_tokens = _optional_token_count(usage.get("total_tokens"))
            if input_tokens is None and output_tokens is None and total_tokens is None:
                self.usage_unavailable_call_count += 1
                self.partial = True
                return
            if input_tokens is None or output_tokens is None:
                self.partial = True
            if total_tokens is None and input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
            if total_tokens is None:
                self.partial = True

            self.usage_available_call_count += 1
            self.input_tokens += input_tokens or 0
            self.output_tokens += output_tokens or 0
            self.total_tokens += total_tokens or 0

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible usage summary."""
        with self._lock:
            return {
                "call_count": self.call_count,
                "usage_available_call_count": self.usage_available_call_count,
                "usage_unavailable_call_count": self.usage_unavailable_call_count,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "models": dict(sorted(self.models.items())),
                "partial": self.partial,
            }


_LLM_USAGE: ContextVar[LLMUsage | None] = ContextVar("legal_rag_llm_usage", default=None)


@contextmanager
def capture_llm_usage() -> Iterator[LLMUsage]:
    """Collect token usage for provider calls made in this async/thread context."""
    usage = LLMUsage()
    token = _LLM_USAGE.set(usage)
    try:
        yield usage
    finally:
        _LLM_USAGE.reset(token)


class BamlTransportError(ValueError):
    """Raised when a BAML provider request fails or returns an oversized body."""

    @classmethod
    def response_too_large(cls, max_response_bytes: int) -> BamlTransportError:
        """Build an error for a response exceeding its configured byte limit."""
        return cls(f"BAML provider response exceeded {max_response_bytes} bytes")


class BamlTruncatedResponseError(BamlTransportError):
    """Raised when the provider stopped at the token limit, leaving unparseable output."""

    @classmethod
    def hit_token_limit(cls, output_tokens: int | None) -> BamlTruncatedResponseError:
        """Build an error naming the limit that cut the response off."""
        used = f" after {output_tokens} output tokens" if output_tokens else ""
        return cls(
            f"BAML provider stopped at its token limit{used}, so the response is incomplete. "
            "Raise the call's max output tokens or ask the prompt for a shorter answer."
        )


class BamlRateLimitError(BamlTransportError):
    """Raised when the provider rejects a request due to quota limits."""

    @classmethod
    def provider_rate_limited(cls, details: str | None = None) -> BamlRateLimitError:
        """Build a retryable error containing the provider's response details."""
        message = "BAML call was rate limited by the provider"
        return cls(f"{message}: {details}" if details else message)


def build_client_registry(
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_output_tokens: int,
    temperature: float,
    timeout_seconds: float,
    reasoning_effort: ReasoningEffort | None = None,
) -> ClientRegistry:
    """Build a BAML client registry bound to one runtime OpenAI-compatible provider."""
    registry = ClientRegistry()
    options: dict[str, Any] = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "supports_streaming": False,
        "http": {"request_timeout_ms": int(timeout_seconds * 1000)},
    }
    if reasoning_effort is not None:
        options["reasoning_effort"] = reasoning_effort
    registry.add_llm_client(
        "RuntimeLLMProvider",
        "openai-generic",
        options,
    )
    registry.set_primary("RuntimeLLMProvider")
    return registry


class BamlCaller:
    """One bounded BAML call site: a pinned provider plus the limits every call to it shares.

    Each typed client owns its own ``b.request.X`` / ``b.parse.X`` pair, so the generated types
    stay at the call site; only the provider registry and the transport bounds are shared here,
    which is the part every client would otherwise re-declare identically.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_output_tokens: int,
        temperature: float,
        timeout_seconds: float,
        max_response_bytes: int,
        extra_request_body: Mapping[str, Any] | None = None,
    ) -> None:
        """Pin one runtime provider and the per-call execution bounds."""
        set_log_level("OFF")
        self.options: dict[str, Any] = {
            "client_registry": build_client_registry(
                base_url=base_url,
                api_key=api_key,
                model=model,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )
        }
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._extra_request_body = extra_request_body

    @classmethod
    def from_settings(
        cls, settings: Settings, *, max_output_tokens: int, temperature: float, max_response_bytes: int
    ) -> BamlCaller:
        """Build a caller from the shared LLM connection settings and one call's own bounds."""
        return cls(
            extra_request_body=settings.llm_extra_request_body,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            max_response_bytes=max_response_bytes,
        )

    def execute(self, request: HTTPRequest) -> str:
        """Run one bounded request and return the provider's raw assistant content."""
        return execute_baml_request(request, self._timeout_seconds, self._max_response_bytes, self._extra_request_body)

    async def execute_async(self, request: HTTPRequest) -> str:
        """Run one bounded request off the event loop, since the transport is blocking."""
        return await asyncio.to_thread(self.execute, request)


def execute_baml_request(
    request: HTTPRequest,
    timeout_seconds: float,
    max_response_bytes: int,
    extra_request_body: Mapping[str, Any] | None = None,
) -> str:
    """Execute one non-streaming OpenAI-compatible request and return assistant content.

    *extra_request_body* carries provider-specific fields that are not part of the
    OpenAI-compatible schema. It is supplied from configuration rather than inferred, so a
    field only one provider understands is never sent to a different one.
    """
    parsed_url = urlsplit(request.url)
    connection_type = http.client.HTTPSConnection if parsed_url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed_url.hostname, parsed_url.port, timeout=timeout_seconds)
    request_path = parsed_url.path or "/"
    if parsed_url.query:
        request_path = f"{request_path}?{parsed_url.query}"
    headers = {name: value for name, value in request.headers.items() if name.lower() != "baml-original-url"}
    request_body = dict(extra_request_body or {})
    request_body.update(request.body.json())
    request_body["stream"] = False
    request_body["response_format"] = {"type": "json_object"}
    try:
        connection.request(
            request.method,
            request_path,
            body=json.dumps(request_body).encode("utf-8"),
            headers=headers,
        )
        response = connection.getresponse()
        response_status = response.status
        response_reason = response.reason
        response_bytes = response.read(max_response_bytes + 1)
    finally:
        connection.close()
    if len(response_bytes) > max_response_bytes:
        raise BamlTransportError.response_too_large(max_response_bytes)
    response_text = response_bytes.decode("utf-8", errors="replace")
    if response_status == HTTPStatus.TOO_MANY_REQUESTS:
        raise BamlRateLimitError.provider_rate_limited(response_text)
    if response_status != HTTPStatus.OK:
        message = f"{response_status} {response_reason}: {response_text}"
        raise http.client.HTTPException(message)
    return _openai_response_content(response_bytes)


def _openai_response_content(response_bytes: bytes) -> str:
    """Return assistant content from one OpenAI-compatible JSON response."""
    response = json.loads(response_bytes)
    provider_error = response.get("error")
    if provider_error is not None:
        if isinstance(provider_error, dict) and provider_error.get("code") == HTTPStatus.TOO_MANY_REQUESTS:
            raise BamlRateLimitError.provider_rate_limited(json.dumps(provider_error, ensure_ascii=False))
        raise BamlTransportError(json.dumps(provider_error, ensure_ascii=False))
    _log_token_usage(response)
    choice = response["choices"][0]
    # A truncated completion parses as an empty value further down the stack, which reads as a
    # coercion bug rather than the budget problem it is.
    if choice.get("finish_reason") == "length":
        raise BamlTruncatedResponseError.hit_token_limit((response.get("usage") or {}).get("completion_tokens"))
    return choice["message"]["content"]


def _log_token_usage(response: dict[str, Any]) -> None:
    """Log standard OpenAI-compatible token usage for one successful LLM response."""
    collector = _LLM_USAGE.get()
    if collector is not None:
        collector.record(response)
    usage = response.get("usage")
    model = response.get("model", "unknown")
    if not isinstance(usage, dict):
        logger.warning("LLM token usage unavailable: model={}", model)
        return
    logger.info(
        "LLM token usage: model={} input_tokens={} output_tokens={} total_tokens={}",
        model,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )


def record_openai_usage(response: Mapping[str, Any]) -> None:
    """Record one non-BAML OpenAI-compatible response in the active usage context."""
    collector = _LLM_USAGE.get()
    if collector is not None:
        collector.record(response)


def _optional_token_count(value: Any) -> int | None:
    """Return a non-negative provider token count or ``None`` when unavailable."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
