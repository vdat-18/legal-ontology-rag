"""Tests for the generated BAML image-caption contract and runtime client."""

from __future__ import annotations

import pytest
from baml_py import Image

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import StructuralContextEntry, StructuralContextRelation
from legal_rag.baml_transport import BamlRateLimitError
from legal_rag.parser import image_captioner
from legal_rag.parser.image_captioner import (
    BamlCaptionClient,
    ImageCaptionError,
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


def test_baml_client_translates_rate_limit_from_shared_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rate limit from the shared BAML transport becomes an image-caption error."""

    def raise_rate_limit(*args: object, **kwargs: object) -> str:
        raise BamlRateLimitError.provider_rate_limited()

    client = BamlCaptionClient(_settings())
    monkeypatch.setattr(image_captioner, "execute_baml_request", raise_rate_limit)

    with pytest.raises(ImageCaptionRateLimitError, match="rate limited"):
        client.caption(image_bytes=b"image", mime_type="image/png", context=ImageContext(()))


def test_baml_client_translates_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-rate-limit transport failure becomes a plain image-caption error."""
    from legal_rag.baml_transport import BamlTransportError

    def raise_transport_error(*args: object, **kwargs: object) -> str:
        raise BamlTransportError("boom")

    client = BamlCaptionClient(_settings())
    monkeypatch.setattr(image_captioner, "execute_baml_request", raise_transport_error)

    with pytest.raises(ImageCaptionError, match="boom"):
        client.caption(image_bytes=b"image", mime_type="image/png", context=ImageContext(()))


def test_baml_client_returns_typed_non_stream_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generated non-stream client returns the typed BAML contract."""
    client = BamlCaptionClient(_settings())
    monkeypatch.setattr(
        image_captioner,
        "execute_baml_request",
        lambda *args, **kwargs: '{"description_vi":"Biển báo"}',
    )

    result = client.caption(image_bytes=b"image", mime_type="image/png", context=ImageContext(()))

    assert result.description_vi == "Biển báo"
