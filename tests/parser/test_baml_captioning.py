"""Tests for the generated BAML image-caption contract and runtime client."""

from __future__ import annotations

import json

import pytest
from baml_py import Image

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import StructuralContextEntry, StructuralContextRelation
from legal_rag.parser import image_captioner
from legal_rag.parser.image_captioner import (
    BamlCaptionClient,
    ImageCaptionRateLimitError,
)
from legal_rag.parser.image_context import ImageContext
from legal_rag.settings import Settings


def _settings() -> Settings:
    """Build runtime settings for an offline BAML request."""
    return Settings(
        llm_base_url="http://localhost:20128/v1",
        llm_api_key="provider-test-key",
        llm_model="provider/test-model",
        llm_timeout_seconds=30,
        image_caption_max_concurrency=1,
        image_caption_max_output_tokens=2048,
        image_caption_max_context_characters=4000,
        image_caption_temperature=0,
    )


def test_baml_contract_builds_provider_multimodal_request() -> None:
    """Generated BAML sends typed context and base64 images to the configured provider."""
    client = BamlCaptionClient(_settings())

    request = b.request.CaptionImage(
        [
            StructuralContextEntry(
                source_ref="word/document.xml#/body/1",
                relation=StructuralContextRelation.SameTableRow,
                distance=0,
                text="Ô tô tải tự đổ",
            )
        ],
        Image.from_base64("image/png", "aW1hZ2U="),
        baml_options={"client_registry": client._registry},
    )
    body = request.body.json()

    assert request.method == "POST"
    assert request.url == "http://localhost:20128/v1/chat/completions"
    assert body["model"] == "provider/test-model"
    assert body["max_tokens"] == 2048
    assert body["temperature"] == 0
    content = body["messages"][0]["content"]
    assert any(part.get("image_url", {}).get("url") == "data:image/png;base64,aW1hZ2U=" for part in content)
    prompt_text = "\n".join(part["text"] for part in content if part["type"] == "text")
    assert "Ô tô tải tự đổ" in prompt_text
    assert "SameTableRow" in prompt_text
    assert "distance=0" in prompt_text
    assert "description_vi" in prompt_text
    assert "visible_text" not in prompt_text


def test_baml_client_exposes_provider_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 429 remains a retryable parser error."""

    def raise_rate_limit(*args: object, **kwargs: object) -> str:
        raise ImageCaptionRateLimitError.provider_rate_limited()

    client = BamlCaptionClient(_settings())
    monkeypatch.setattr(image_captioner, "_execute_baml_request", raise_rate_limit)

    with pytest.raises(ImageCaptionRateLimitError, match="rate limited"):
        client.caption(image_bytes=b"image", mime_type="image/png", context=ImageContext(()))


def test_baml_client_returns_typed_non_stream_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generated non-stream client returns the typed BAML contract."""
    client = BamlCaptionClient(_settings())
    monkeypatch.setattr(
        image_captioner,
        "_execute_baml_request",
        lambda *args, **kwargs: '{"description_vi":"Biển báo"}',
    )

    result = client.caption(image_bytes=b"image", mime_type="image/png", context=ImageContext(()))

    assert result.description_vi == "Biển báo"


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

    client = BamlCaptionClient(_settings())
    request = b.request.CaptionImage(
        [],
        Image.from_base64("image/png", "aW1hZ2U="),
        baml_options={"client_registry": client._registry},
    )
    monkeypatch.setattr(image_captioner.http.client, "HTTPConnection", JsonConnection)

    content = image_captioner._execute_baml_request(request, 30, 1024)

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

    client = BamlCaptionClient(_settings())
    request = b.request.CaptionImage(
        [],
        Image.from_base64("image/png", "aW1hZ2U="),
        baml_options={"client_registry": client._registry},
    )
    monkeypatch.setattr(image_captioner.http.client, "HTTPConnection", ErrorConnection)

    with pytest.raises(image_captioner.http.client.HTTPException) as error:
        image_captioner._execute_baml_request(request, 30, 1024)

    assert str(error.value) == '503 Service Unavailable: {"error":{"message":"Upstream model unavailable"}}'


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
        image_captioner._openai_response_content(response_bytes)


def test_openai_response_exposes_embedded_rate_limit() -> None:
    """An HTTP 200 response containing error code 429 remains retryable."""
    response_bytes = b'{"error":{"code":429}}'

    with pytest.raises(ImageCaptionRateLimitError, match="rate limited"):
        image_captioner._openai_response_content(response_bytes)


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

    with pytest.raises(image_captioner.ImageCaptionError) as error:
        image_captioner._openai_response_content(response_bytes)

    assert '"code": "upstream_error"' in str(error.value)
    assert '"type": "server_error"' in str(error.value)
    assert '"message": "Upstream model unavailable"' in str(error.value)
