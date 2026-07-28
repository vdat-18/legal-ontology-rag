"""Offline tests for contextual image caption generation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from legal_rag.parser import image_captioner
from legal_rag.parser.document_source import KIND_PARAGRAPH, DocumentSource, DocumentSourceUnit
from legal_rag.parser.image_captioner import (
    ImageCaptionError,
    ImageCaptionRateLimitError,
    caption_images,
)
from legal_rag.parser.image_context import ImageContext
from legal_rag.parser.images import DocxImage, DocxImageExtraction
from legal_rag.parser.models import CaptionModelResponse
from legal_rag.settings import LlmProvider, Settings


class FakeCaptionClient:
    """Capture multimodal requests without network access."""

    def __init__(self) -> None:
        """Initialize captured image and context payloads."""
        self.images: list[tuple[bytes, str]] = []
        self.contexts: list[ImageContext] = []

    def caption(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        context: ImageContext,
    ) -> CaptionModelResponse:
        """Return a valid response derived only from request order."""
        self.images.append((image_bytes, mime_type))
        self.contexts.append(context)
        return CaptionModelResponse(description_vi=f"Mô tả {len(self.images)}")


class RateLimitedCaptionClient:
    """Fail repeatedly with provider rate limits before returning a caption."""

    def __init__(self) -> None:
        """Initialize the request counter."""
        self.call_count = 0

    def caption(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        context: ImageContext,
    ) -> CaptionModelResponse:
        """Return a caption after two simulated HTTP 429 responses."""
        self.call_count += 1
        if self.call_count <= 2:
            raise ImageCaptionRateLimitError.provider_rate_limited()
        return CaptionModelResponse(description_vi="Mô tả sau retry")


class AlwaysRateLimitedCaptionClient:
    """Return a provider rate limit for every caption request."""

    def __init__(self) -> None:
        """Initialize the request counter."""
        self.call_count = 0

    def caption(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        context: ImageContext,
    ) -> CaptionModelResponse:
        """Raise a rate-limit error for every request."""
        self.call_count += 1
        raise ImageCaptionRateLimitError.provider_rate_limited()


def _settings() -> Settings:
    """Build bounded settings without reading the process environment."""
    return Settings(
        llm_base_url="http://localhost:20128/v1",
        llm_api_key="test-provider-key",
        llm_model="provider/test-model",
        llm_provider=LlmProvider.OPENAI,
        llm_timeout_seconds=30,
        image_caption_max_concurrency=1,
        image_caption_max_output_tokens=256,
        image_caption_max_context_characters=1000,
        image_caption_temperature=0,
    )


def _image(*, sequence_index: int, mime_type: str = "image/png", media_bytes: bytes = b"image") -> DocxImage:
    """Build one top-level paragraph image occurrence."""
    parent_ref = f"word/document.xml#/body/{sequence_index}"
    return DocxImage(
        schema_version="1.0.0",
        image_id=f"image-{sequence_index}",
        source_name="source.docx",
        source_sha256="source-hash",
        sequence_index=sequence_index,
        body_index=sequence_index,
        source_ref=f"{parent_ref}/run/0/image/0",
        parent_source_ref=parent_ref,
        run_index=0,
        image_index=0,
        relationship_id=f"rId{sequence_index}",
        package_path=f"word/media/image{sequence_index}.png",
        media_sha256=f"media-{sequence_index}",
        mime_type=mime_type,
        asset_name=f"media-{sequence_index}.png",
        media_bytes=media_bytes,
    )


def _source(images: tuple[DocxImage, ...]) -> DocumentSource:
    """Build matching paragraph context for image occurrences."""
    units = tuple(
        DocumentSourceUnit(
            schema_version="1.1.0",
            unit_id=f"unit-{index}",
            source_name="source.docx",
            source_sha256="source-hash",
            sequence_index=index,
            body_index=image.body_index,
            source_ref=image.parent_source_ref,
            kind=KIND_PARAGRAPH,
            payload_sha256=f"payload-{index}",
            text=f"Ngữ cảnh {index}",
            runs=(),
            rows=None,
        )
        for index, image in enumerate(images)
    )
    return DocumentSource("source.docx", "source-hash", units)


def test_settings_use_code_defaults_for_image_captioning() -> None:
    """Image-caption execution limits do not depend on environment variables."""
    environment = {
        "LLM_BASE_URL": "http://localhost:20128/v1/",
        "LLM_API_KEY": "provider-key",
        "LLM_MODEL": "provider/model",
        "LLM_PROVIDER": "openai",
        "LLM_TIMEOUT_SECONDS": "60",
    }

    settings = Settings.from_environment(environment)

    assert settings.llm_base_url == "http://localhost:20128/v1"
    assert settings.image_caption_max_concurrency == 4
    assert settings.image_caption_max_output_tokens == 2048
    assert settings.image_caption_max_context_characters == 4000
    assert settings.image_caption_temperature == 0
    assert settings.image_caption_surrounding_context_units == 3
    assert settings.image_caption_max_rate_limit_retries == 3
    assert settings.image_caption_rate_limit_retry_delay_seconds == 60
    assert settings.image_caption_max_response_bytes == 1_048_576


def test_caption_docx_images_preserves_order_and_supplies_context() -> None:
    """Concurrent work preserves image order and sends bounded structural context."""
    images = (_image(sequence_index=0, media_bytes=b"first"), _image(sequence_index=1, media_bytes=b"second"))
    extraction = DocxImageExtraction("source.docx", "source-hash", images)
    client = FakeCaptionClient()

    captions = caption_images(extraction, _source(images), _settings(), client)

    assert [caption.image_id for caption in captions] == ["image-0", "image-1"]
    assert [caption.description_vi for caption in captions] == ["Mô tả 1", "Mô tả 2"]
    assert [caption.context_source_refs for caption in captions] == [
        ("word/document.xml#/body/0", "word/document.xml#/body/1"),
        ("word/document.xml#/body/1", "word/document.xml#/body/0"),
    ]
    assert [[entry.text for entry in context.entries] for context in client.contexts] == [
        ["Ngữ cảnh 0", "Ngữ cảnh 1"],
        ["Ngữ cảnh 1", "Ngữ cảnh 0"],
    ]
    assert client.images == [(b"first", "image/png"), (b"second", "image/png")]


def test_caption_retries_until_provider_recovers() -> None:
    """Repeated HTTP 429 responses pause until the same image succeeds."""
    image = _image(sequence_index=0)
    extraction = DocxImageExtraction("source.docx", "source-hash", (image,))
    client = RateLimitedCaptionClient()
    delays: list[float] = []

    captions = caption_images(extraction, _source((image,)), _settings(), client, sleep=delays.append)

    assert client.call_count == 3
    assert delays == [60, 60]
    assert captions[0].description_vi == "Mô tả sau retry"


def test_caption_stops_after_three_rate_limit_retries() -> None:
    """Persistent HTTP 429 responses fail after the configured retry limit."""
    image = _image(sequence_index=0)
    extraction = DocxImageExtraction("source.docx", "source-hash", (image,))
    client = AlwaysRateLimitedCaptionClient()
    delays: list[float] = []

    with pytest.raises(ImageCaptionRateLimitError, match="rate limited"):
        caption_images(extraction, _source((image,)), _settings(), client, sleep=delays.append)

    assert client.call_count == 4
    assert delays == [60, 60, 60]


def test_legacy_media_requires_or_uses_explicit_renderer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Legacy office media is never sent without rasterization."""
    image = _image(sequence_index=0, mime_type="image/x-emf", media_bytes=b"emf")
    extraction = DocxImageExtraction("source.docx", "source-hash", (image,))
    source = _source((image,))

    with pytest.raises(ImageCaptionError, match="renderer-executable"):
        caption_images(extraction, source, _settings(), FakeCaptionClient())

    renderer = tmp_path / "soffice.com"
    renderer.write_bytes(b"renderer")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "capture_output" not in kwargs
        assert "stdout" not in kwargs
        assert "stderr" not in kwargs
        output_directory = Path(command[command.index("--outdir") + 1])
        input_path = Path(command[-1])
        output_directory.joinpath(f"{input_path.stem}.png").write_bytes(b"png")
        return subprocess.CompletedProcess(command, 0, "converted", "")

    monkeypatch.setattr(image_captioner.subprocess, "run", fake_run)
    client = FakeCaptionClient()

    captions = caption_images(
        extraction,
        source,
        _settings(),
        client,
        renderer_executable=renderer,
    )

    assert captions[0].caption_mime_type == "image/png"
    assert client.images[0] == (b"png", "image/png")
