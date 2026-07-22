"""Tests for the generated BAML image-caption contract and runtime client."""

from __future__ import annotations

from baml_py import Image

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import StructuralContextEntry
from legal_rag.parser.config import ParserSettings
from legal_rag.parser.image_captioner import BamlCaptionClient


def _settings() -> ParserSettings:
    """Build runtime settings for an offline BAML request."""
    return ParserSettings(
        base_url="http://localhost:20128/v1",
        api_key="router-test-key",
        model="gemini/test-model",
        timeout_seconds=30,
        max_concurrency=1,
        max_output_tokens=2048,
        max_context_characters=4000,
        temperature=0,
    )


def test_baml_contract_builds_9router_multimodal_request() -> None:
    """Generated BAML sends typed context and base64 images to 9Router."""
    client = BamlCaptionClient(_settings())

    request = b.request.CaptionImage(
        [StructuralContextEntry(source_ref="word/document.xml#/body/1", text="Ô tô tải tự đổ")],
        Image.from_base64("image/png", "aW1hZ2U="),
        baml_options={"client_registry": client._registry},
    )
    body = request.body.json()

    assert request.method == "POST"
    assert request.url == "http://localhost:20128/v1/chat/completions"
    assert body["model"] == "gemini/test-model"
    assert body["max_tokens"] == 2048
    assert body["temperature"] == 0
    content = body["messages"][0]["content"]
    assert any(part.get("image_url", {}).get("url") == "data:image/png;base64,aW1hZ2U=" for part in content)
    prompt_text = "\n".join(part["text"] for part in content if part["type"] == "text")
    assert "Ô tô tải tự đổ" in prompt_text
    assert "description_vi" in prompt_text
    assert "visible_text" in prompt_text
