"""Tests for the complete enriched parser pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.parser import pipeline
from legal_rag.parser.config import ParserSettings
from legal_rag.parser.document_source import KIND_PARAGRAPH, DocumentSource, DocumentSourceUnit
from legal_rag.parser.image_context import ImageContext
from legal_rag.parser.images import DocxImage, DocxImageExtraction
from legal_rag.parser.models import CaptionModelResponse
from legal_rag.parser.pipeline import ParserPipeline, ParserPipelineError
from legal_rag.parser.writer import write_parsed_document


class FakeCaptionClient:
    """Return a deterministic typed caption without external I/O."""

    def __init__(self) -> None:
        """Capture structural contexts supplied by the pipeline."""
        self.contexts: list[ImageContext] = []

    def caption(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        context: ImageContext,
    ) -> CaptionModelResponse:
        """Build a caption from validated request fields."""
        assert image_bytes == b"image"
        assert mime_type == "image/png"
        self.contexts.append(context)
        return CaptionModelResponse("Biển báo giao thông.", None)


def _settings() -> ParserSettings:
    """Build parser settings without reading the environment."""
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


def _source() -> DocumentSource:
    """Build one paragraph source unit."""
    unit = DocumentSourceUnit(
        schema_version="1.1.0",
        unit_id="unit-0",
        source_name="source.docx",
        source_sha256="source-hash",
        sequence_index=0,
        body_index=0,
        source_ref="word/document.xml#/body/0",
        kind=KIND_PARAGRAPH,
        payload_sha256="payload-hash",
        text="Biển báo tại vị trí này.",
        runs=(),
        rows=None,
    )
    return DocumentSource("source.docx", "source-hash", (unit,), 1, 0, 0, (), ())


def _images(*, source_sha256: str = "source-hash") -> DocxImageExtraction:
    """Build one image occurrence attached to the paragraph source unit."""
    image = DocxImage(
        schema_version="1.0.0",
        image_id="image-0",
        source_name="source.docx",
        source_sha256=source_sha256,
        sequence_index=0,
        body_index=0,
        source_ref="word/document.xml#/body/0/run/0/image/0",
        parent_source_ref="word/document.xml#/body/0",
        run_index=0,
        image_index=0,
        relationship_id="rId1",
        package_path="word/media/image1.png",
        media_sha256="media-hash",
        mime_type="image/png",
        asset_name="media-hash.png",
        alt_text=None,
        media_bytes=b"image",
    )
    return DocxImageExtraction("source.docx", source_sha256, (image,))


def test_pipeline_attaches_caption_to_final_source_unit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One pipeline call produces one enriched JSONL output without a caption sidecar."""
    monkeypatch.setattr(pipeline, "extract_document_source", lambda path: _source())
    monkeypatch.setattr(pipeline, "extract_docx_images", lambda path: _images())
    client = FakeCaptionClient()
    parser = ParserPipeline(_settings(), client)

    document = parser.parse(tmp_path / "source.docx")
    output_path = tmp_path / "parsed.jsonl"
    write_parsed_document(output_path, document)

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert document.image_count == 1
    assert records == [unit.to_dict() for unit in document.units]
    assert records[0]["schema_version"] == "1.0.0"
    assert records[0]["source_schema_version"] == "1.1.0"
    assert records[0]["images"][0]["description_vi"] == "Biển báo giao thông."
    assert records[0]["images"][0]["body_index"] == 0
    assert client.contexts[0].source_refs == ("word/document.xml#/body/0",)


def test_pipeline_preserves_an_image_only_paragraph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An image-only paragraph becomes an explicit final record at its native position."""
    empty_source = DocumentSource("source.docx", "source-hash", (), 0, 0, 0, (), ())
    monkeypatch.setattr(pipeline, "extract_document_source", lambda path: empty_source)
    monkeypatch.setattr(pipeline, "extract_docx_images", lambda path: _images())
    client = FakeCaptionClient()

    document = ParserPipeline(_settings(), client).parse(tmp_path / "source.docx")

    assert document.units[0].to_dict()["kind"] == "image"
    assert document.units[0].to_dict()["source_ref"] == "word/document.xml#/body/0"
    assert document.units[0].to_dict()["images"][0]["image_id"] == "image-0"
    assert client.contexts[0].entries == ()


def test_pipeline_rejects_mismatched_extraction_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Parser stages cannot combine provenance from different source bytes."""
    monkeypatch.setattr(pipeline, "extract_document_source", lambda path: _source())
    monkeypatch.setattr(pipeline, "extract_docx_images", lambda path: _images(source_sha256="other-hash"))
    parser = ParserPipeline(_settings(), FakeCaptionClient())

    with pytest.raises(ParserPipelineError, match="different source bytes"):
        parser.parse(tmp_path / "source.docx")


def test_pipeline_rejects_a_document_without_parseable_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An empty extraction fails instead of writing an empty success artifact."""
    empty_source = DocumentSource("source.docx", "source-hash", (), 0, 0, 0, (), ())
    empty_images = DocxImageExtraction("source.docx", "source-hash", ())
    monkeypatch.setattr(pipeline, "extract_document_source", lambda path: empty_source)
    monkeypatch.setattr(pipeline, "extract_docx_images", lambda path: empty_images)

    with pytest.raises(ParserPipelineError, match="no text, tables, or images"):
        ParserPipeline(_settings(), FakeCaptionClient()).parse(tmp_path / "source.docx")
