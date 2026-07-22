"""Offline tests for contextual image caption generation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from legal_rag.parser import image_captioner
from legal_rag.parser.config import ParserConfigurationError, ParserSettings
from legal_rag.parser.document_source import KIND_PARAGRAPH, DocumentSource, DocumentSourceUnit
from legal_rag.parser.image_captioner import (
    ImageCaptionError,
    caption_images,
)
from legal_rag.parser.image_context import ImageContext
from legal_rag.parser.images import DocxImage, DocxImageExtraction
from legal_rag.parser.models import CaptionModelResponse


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
        return CaptionModelResponse(
            description_vi=f"Mô tả {len(self.images)}",
            visible_text=None,
        )


def _settings() -> ParserSettings:
    """Build bounded settings without reading the process environment."""
    return ParserSettings(
        base_url="http://localhost:20128/v1",
        api_key="test-router-key",
        model="gemini/test-model",
        timeout_seconds=30,
        max_concurrency=1,
        max_output_tokens=256,
        max_context_characters=1000,
        temperature=0,
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
        alt_text=None,
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
    return DocumentSource("source.docx", "source-hash", units, len(units), 0, 0, (), ())


def test_settings_require_typed_environment_values() -> None:
    """Service and context limits come from typed configuration."""
    environment = {
        "LLM_BASE_URL": "http://localhost:20128/v1/",
        "LLM_API_KEY": "router-key",
        "LLM_MODEL": "gemini/model",
        "LLM_TIMEOUT_SECONDS": "60",
        "IMAGE_CAPTION_MAX_CONCURRENCY": "4",
        "IMAGE_CAPTION_MAX_OUTPUT_TOKENS": "2048",
        "IMAGE_CAPTION_MAX_CONTEXT_CHARACTERS": "4000",
        "IMAGE_CAPTION_TEMPERATURE": "0",
    }

    settings = ParserSettings.from_environment(environment)

    assert settings.base_url == "http://localhost:20128/v1"
    assert settings.max_concurrency == 4
    assert settings.max_output_tokens == 2048
    assert settings.max_context_characters == 4000


def test_settings_reject_missing_limits() -> None:
    """Incomplete execution configuration fails before creating an API client."""
    with pytest.raises(ParserConfigurationError, match="LLM_TIMEOUT_SECONDS"):
        ParserSettings.from_environment(
            {
                "LLM_BASE_URL": "http://localhost:20128/v1",
                "LLM_API_KEY": "router-key",
                "LLM_MODEL": "gemini/model",
                "IMAGE_CAPTION_TEMPERATURE": "0",
            }
        )


def test_caption_docx_images_preserves_order_and_supplies_context() -> None:
    """Concurrent work preserves image order and sends bounded structural context."""
    images = (_image(sequence_index=0, media_bytes=b"first"), _image(sequence_index=1, media_bytes=b"second"))
    extraction = DocxImageExtraction("source.docx", "source-hash", images)
    client = FakeCaptionClient()

    captions = caption_images(extraction, _source(images), _settings(), client)

    assert [caption.image_id for caption in captions] == ["image-0", "image-1"]
    assert [caption.description_vi for caption in captions] == ["Mô tả 1", "Mô tả 2"]
    assert [caption.context_source_refs for caption in captions] == [
        ("word/document.xml#/body/0",),
        ("word/document.xml#/body/1",),
    ]
    assert [context.entries[0].text for context in client.contexts] == ["Ngữ cảnh 0", "Ngữ cảnh 1"]
    assert client.images == [(b"first", "image/png"), (b"second", "image/png")]


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
