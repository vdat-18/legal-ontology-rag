"""Shared non-streaming HTTP transport for generated BAML requests."""

from __future__ import annotations

import asyncio
import http.client
import json
from http import HTTPStatus
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from baml_py import ClientRegistry, HTTPRequest
from baml_py.logging import set_log_level

if TYPE_CHECKING:
    from legal_rag.settings import Settings


class BamlTransportError(ValueError):
    """Raised when a BAML provider request fails or returns an oversized body."""

    @classmethod
    def response_too_large(cls, max_response_bytes: int) -> BamlTransportError:
        """Build an error for a response exceeding its configured byte limit."""
        return cls(f"BAML provider response exceeded {max_response_bytes} bytes")


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
) -> ClientRegistry:
    """Build a BAML client registry bound to one runtime OpenAI-compatible provider."""
    registry = ClientRegistry()
    registry.add_llm_client(
        "RuntimeLLMProvider",
        "openai-generic",
        {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "supports_streaming": False,
            "http": {"request_timeout_ms": int(timeout_seconds * 1000)},
        },
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

    @classmethod
    def from_settings(
        cls, settings: Settings, *, max_output_tokens: int, temperature: float, max_response_bytes: int
    ) -> BamlCaller:
        """Build a caller from the shared LLM connection settings and one call's own bounds."""
        return cls(
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
        return execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)

    async def execute_async(self, request: HTTPRequest) -> str:
        """Run one bounded request off the event loop, since the transport is blocking."""
        return await asyncio.to_thread(self.execute, request)


def execute_baml_request(request: HTTPRequest, timeout_seconds: float, max_response_bytes: int) -> str:
    """Execute one non-streaming OpenAI-compatible request and return assistant content."""
    parsed_url = urlsplit(request.url)
    connection_type = http.client.HTTPSConnection if parsed_url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed_url.hostname, parsed_url.port, timeout=timeout_seconds)
    request_path = parsed_url.path or "/"
    if parsed_url.query:
        request_path = f"{request_path}?{parsed_url.query}"
    headers = {name: value for name, value in request.headers.items() if name.lower() != "baml-original-url"}
    request_body = request.body.json()
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
    return response["choices"][0]["message"]["content"]
