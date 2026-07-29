"""Tests for the shared non-streaming BAML HTTP transport."""

from __future__ import annotations

import http.client
import json

import pytest
from baml_py import Image

from legal_rag import baml_transport
from legal_rag.baml_client.sync_client import b
from legal_rag.baml_transport import BamlRateLimitError, BamlTransportError, build_client_registry


def _request() -> object:
    """Build one generated BAML HTTP request for transport-level tests."""
    registry = build_client_registry(
        base_url="http://localhost:20128/v1",
        api_key="provider-test-key",
        model="provider/test-model",
        max_output_tokens=2048,
        temperature=0,
        timeout_seconds=30,
    )
    return b.request.CaptionImage(
        [],
        Image.from_base64("image/png", "aW1hZ2U="),
        baml_options={"client_registry": registry},
    )


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
        (b'{"choices":"invalid"}', TypeError),
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
