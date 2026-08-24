"""Tests for the shared non-streaming BAML HTTP transport."""

from __future__ import annotations

import http.client
import json
from typing import ClassVar

import pytest
from baml_py import Image
from loguru import logger

from legal_rag import baml_transport
from legal_rag.baml_client.sync_client import b
from legal_rag.baml_transport import BamlRateLimitError, BamlTransportError, build_client_registry


def _request(*, reasoning_effort: baml_transport.ReasoningEffort | None = None) -> object:
    """Build one generated BAML HTTP request for transport-level tests."""
    registry = build_client_registry(
        base_url="http://localhost:20128/v1",
        api_key="provider-test-key",
        model="provider/test-model",
        max_output_tokens=2048,
        temperature=0,
        timeout_seconds=30,
        reasoning_effort=reasoning_effort,
    )
    return b.request.CaptionImage(
        [],
        Image.from_base64("image/png", "aW1hZ2U="),
        baml_options={"client_registry": registry},
    )


def test_runtime_registry_sends_configured_reasoning_effort() -> None:
    """The generated request carries the configured OpenAI reasoning effort."""
    request = _request(reasoning_effort="medium")

    assert request.body.json()["reasoning_effort"] == "medium"


def test_non_stream_transport_forces_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider receives explicit non-stream JSON response settings."""
    captured_body: dict[str, object] = {}

    class JsonResponse:
        """Return one bounded OpenAI-compatible JSON response."""

        status = 200
        reason = "OK"

        def read(self, amount: int) -> bytes:
            assert amount == 1025
            return json.dumps({"choices": [{"message": {"content": '{"description_vi":"Biển báo"}'}}]}).encode()

    class JsonConnection:
        """Capture the request body without external I/O."""

        def __init__(self, host: str, port: int | None, timeout: float) -> None:
            return None

        def request(self, method: str, path: str, *, body: bytes, headers: dict[str, str]) -> None:
            captured_body.update(json.loads(body))

        def getresponse(self) -> JsonResponse:
            return JsonResponse()

        def close(self) -> None:
            return None

    request = _request()
    monkeypatch.setattr(baml_transport.http.client, "HTTPConnection", JsonConnection)

    content = baml_transport.execute_baml_request(request, 30, 1024)

    assert captured_body["stream"] is False
    assert captured_body["response_format"] == {"type": "json_object"}
    assert content == '{"description_vi":"Biển báo"}'


def test_non_stream_transport_exposes_http_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-successful HTTP responses preserve provider status and body."""

    class ErrorResponse:
        """Return one provider failure response."""

        status = 503
        reason = "Service Unavailable"

        def read(self, amount: int) -> bytes:
            assert amount == 1025
            return b'{"error":{"message":"Upstream model unavailable"}}'

    class ErrorConnection:
        """Provide the failed response without external I/O."""

        def __init__(self, host: str, port: int | None, timeout: float) -> None:
            return None

        def request(self, method: str, path: str, *, body: bytes, headers: dict[str, str]) -> None:
            return None

        def getresponse(self) -> ErrorResponse:
            return ErrorResponse()

        def close(self) -> None:
            return None

    request = _request()
    monkeypatch.setattr(baml_transport.http.client, "HTTPConnection", ErrorConnection)

    with pytest.raises(http.client.HTTPException) as error:
        baml_transport.execute_baml_request(request, 30, 1024)

    assert str(error.value) == '503 Service Unavailable: {"error":{"message":"Upstream model unavailable"}}'


def test_execute_baml_request_raises_on_oversized_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A response body exceeding the configured limit is rejected before parsing."""

    class OversizedResponse:
        """Return more bytes than the configured limit allows."""

        status = 200
        reason = "OK"

        def read(self, amount: int) -> bytes:
            return b"x" * amount

    class OversizedConnection:
        """Provide the oversized response without external I/O."""

        def __init__(self, host: str, port: int | None, timeout: float) -> None:
            return None

        def request(self, method: str, path: str, *, body: bytes, headers: dict[str, str]) -> None:
            return None

        def getresponse(self) -> OversizedResponse:
            return OversizedResponse()

        def close(self) -> None:
            return None

    request = _request()
    monkeypatch.setattr(baml_transport.http.client, "HTTPConnection", OversizedConnection)

    with pytest.raises(BamlTransportError, match="exceeded 4 bytes"):
        baml_transport.execute_baml_request(request, 30, 4)


@pytest.mark.parametrize(
    ("response_bytes", "error_type"),
    [
        (b"not-json", json.JSONDecodeError),
        (b"[]", AttributeError),
        (b"{}", KeyError),
        # a string in place of the choices list is indexed then probed for finish_reason
        (b'{"choices":"invalid"}', AttributeError),
    ],
)
def test_openai_response_exposes_native_schema_errors(
    response_bytes: bytes,
    error_type: type[Exception],
) -> None:
    """Malformed provider envelopes expose their native parsing errors."""
    with pytest.raises(error_type):
        baml_transport._openai_response_content(response_bytes)


def test_openai_response_exposes_embedded_rate_limit() -> None:
    """An HTTP 200 response containing error code 429 remains retryable."""
    response_bytes = b'{"error":{"code":429}}'

    with pytest.raises(BamlRateLimitError, match="rate limited"):
        baml_transport._openai_response_content(response_bytes)


def test_openai_response_exposes_full_provider_error() -> None:
    """Provider error details remain visible to the caller."""
    response_bytes = json.dumps(
        {
            "error": {
                "code": "upstream_error",
                "type": "server_error",
                "message": "Upstream model unavailable",
            }
        }
    ).encode()

    with pytest.raises(BamlTransportError) as error:
        baml_transport._openai_response_content(response_bytes)

    assert '"code": "upstream_error"' in str(error.value)
    assert '"type": "server_error"' in str(error.value)
    assert '"message": "Upstream model unavailable"' in str(error.value)


def test_openai_response_logs_token_usage() -> None:
    """Successful provider responses expose their standard token counts in logs."""
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    response_bytes = json.dumps(
        {
            "model": "provider/test-model",
            "usage": {"prompt_tokens": 30_000, "completion_tokens": 5_000, "total_tokens": 35_000},
            "choices": [{"message": {"content": "{}"}}],
        }
    ).encode()

    try:
        content = baml_transport._openai_response_content(response_bytes)
    finally:
        logger.remove(sink_id)

    assert content == "{}"
    assert any(
        "model=provider/test-model input_tokens=30000 output_tokens=5000 total_tokens=35000" in message
        for message in messages
    )


def test_extra_request_body_is_merged_into_the_request() -> None:
    """A provider field outside the OpenAI schema reaches the wire when configured."""
    sent: dict[str, object] = {}

    class _FakeBody:
        def json(self) -> dict[str, object]:
            return {"model": "m", "messages": []}

    class _FakeRequest:
        url = "http://provider.test/v1/chat/completions"
        method = "POST"
        headers: ClassVar[dict[str, str]] = {"Content-Type": "application/json"}
        body = _FakeBody()

    class _FakeResponse:
        status = 200
        reason = "OK"

        def read(self, _size: int) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    class _FakeConnection:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def request(self, _method: str, _path: str, body: bytes, headers: dict[str, str]) -> None:
            sent.update(json.loads(body))

        def getresponse(self) -> _FakeResponse:
            return _FakeResponse()

        def close(self) -> None:
            pass

    original = http.client.HTTPConnection
    http.client.HTTPConnection = _FakeConnection  # type: ignore[misc]
    try:
        baml_transport.execute_baml_request(_FakeRequest(), 30, 1_048_576, {"thinking": {"type": "disabled"}})
    finally:
        http.client.HTTPConnection = original  # type: ignore[misc]

    assert sent["thinking"] == {"type": "disabled"}
    assert sent["stream"] is False


def test_no_extra_request_body_leaves_the_request_untouched() -> None:
    """Without configuration nothing provider-specific is added, so other providers still work."""
    sent: dict[str, object] = {}

    class _FakeBody:
        def json(self) -> dict[str, object]:
            return {"model": "m", "messages": []}

    class _FakeRequest:
        url = "http://provider.test/v1/chat/completions"
        method = "POST"
        headers: ClassVar[dict[str, str]] = {"Content-Type": "application/json"}
        body = _FakeBody()

    class _FakeResponse:
        status = 200
        reason = "OK"

        def read(self, _size: int) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    class _FakeConnection:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def request(self, _method: str, _path: str, body: bytes, headers: dict[str, str]) -> None:
            sent.update(json.loads(body))

        def getresponse(self) -> _FakeResponse:
            return _FakeResponse()

        def close(self) -> None:
            pass

    original = http.client.HTTPConnection
    http.client.HTTPConnection = _FakeConnection  # type: ignore[misc]
    try:
        baml_transport.execute_baml_request(_FakeRequest(), 30, 1_048_576)
    finally:
        http.client.HTTPConnection = original  # type: ignore[misc]

    assert "thinking" not in sent


def test_truncated_completion_reports_the_token_limit_not_a_coercion_error() -> None:
    """A response cut off at the limit must say so; it otherwise parses as an empty value."""
    response = json.dumps(
        {
            "choices": [{"finish_reason": "length", "message": {"content": '{"answer_vi": "mot phan'}}],
            "usage": {"completion_tokens": 8192},
        }
    ).encode()

    with pytest.raises(baml_transport.BamlTruncatedResponseError, match="8192 output tokens"):
        baml_transport._openai_response_content(response)


def test_a_completed_response_is_returned_unchanged() -> None:
    """Only a length-truncated response is rejected, not every finish_reason."""
    response = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}).encode()

    assert baml_transport._openai_response_content(response) == "ok"
