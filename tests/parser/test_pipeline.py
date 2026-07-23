"""Tests for the complete enriched parser pipeline."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from legal_rag.parser import pipeline
from legal_rag.parser.attachments import DocxAttachmentExtraction
from legal_rag.parser.document_source import KIND_PARAGRAPH, DocumentSource, DocumentSourceUnit
from legal_rag.parser.image_context import ImageContext
from legal_rag.parser.images import DocxImage, DocxImageExtraction
from legal_rag.parser.models import CaptionModelResponse
from legal_rag.parser.pipeline import ParserPipeline, ParserPipelineError
from legal_rag.parser.writer import write_parsed_document
from legal_rag.settings import Settings


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
        return CaptionModelResponse("Biển báo giao thông.")


def _settings() -> Settings:
    """Build parser settings without reading the environment."""
    return Settings(
        llm_base_url="http://localhost:20128/v1",
        llm_api_key="test-provider-key",
        llm_model="provider/test-model",
        llm_timeout_seconds=30,
        image_caption_max_concurrency=1,
        image_caption_max_output_tokens=256,
        image_caption_max_context_characters=1000,
        image_caption_temperature=0,
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
    return DocumentSource("source.docx", "source-hash", (unit,))


def _images() -> DocxImageExtraction:
    """Build one image occurrence attached to the paragraph source unit."""
    image = DocxImage(
        schema_version="1.0.0",
        image_id="image-0",
        source_name="source.docx",
        source_sha256="source-hash",
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
        media_bytes=b"image",
    )
    return DocxImageExtraction("source.docx", "source-hash", (image,))


def test_native_extraction_reads_docx_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Paragraph, table, and image stages share one loaded DOCX package."""
    input_path = tmp_path / "source.docx"
    with zipfile.ZipFile(input_path, mode="w") as archive:
        archive.writestr(
            "word/document.xml",
            """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:body><w:p><w:r><w:t>Nội dung</w:t></w:r></w:p></w:body>
            </w:document>""",
        )
        archive.writestr(
            "[Content_Types].xml",
            """<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
            <Default Extension="xml" ContentType="application/xml"/>
            </Types>""",
        )
    original_read_bytes = Path.read_bytes
    read_count = 0

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal read_count
        read_count += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)

    source, images, attachments = pipeline._extract_parser_inputs(input_path)

    assert read_count == 1
    assert [unit.text for unit in source.units] == ["Nội dung"]
    assert images.images == ()
    assert attachments.attachments == ()


def test_pipeline_attaches_caption_to_final_source_unit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One pipeline call produces one enriched JSONL output without a caption sidecar."""
    empty_attachments = DocxAttachmentExtraction("source.docx", "source-hash", ())
    monkeypatch.setattr(pipeline, "_extract_parser_inputs", lambda path: (_source(), _images(), empty_attachments))
    client = FakeCaptionClient()
    parser = ParserPipeline(_settings(), client)

    document = parser.parse(tmp_path / "source.docx")
    output_path = tmp_path / "parsed.jsonl"
    write_parsed_document(output_path, document)

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert document.image_count == 1
    assert records == [unit.to_dict() for unit in document.units]
    assert records[0]["schema_version"] == "2.0.0"
    assert records[0]["source_schema_version"] == "1.1.0"
    assert records[0]["images"][0]["description_vi"] == "Biển báo giao thông."
    assert records[0]["images"][0]["schema_version"] == "2.0.0"
    assert "visible_text" not in records[0]["images"][0]
    assert records[0]["images"][0]["body_index"] == 0
    assert client.contexts[0].source_refs == ("word/document.xml#/body/0",)


def test_pipeline_preserves_an_image_only_paragraph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An image-only paragraph becomes an explicit final record at its native position."""
    empty_source = DocumentSource("source.docx", "source-hash", ())
    empty_attachments = DocxAttachmentExtraction("source.docx", "source-hash", ())
    monkeypatch.setattr(
        pipeline,
        "_extract_parser_inputs",
        lambda path: (empty_source, _images(), empty_attachments),
    )
    client = FakeCaptionClient()

    document = ParserPipeline(_settings(), client).parse(tmp_path / "source.docx")

    assert document.units[0].to_dict()["kind"] == "image"
    assert document.units[0].to_dict()["source_ref"] == "word/document.xml#/body/0"
    assert document.units[0].to_dict()["images"][0]["image_id"] == "image-0"
    assert client.contexts[0].entries == ()


def test_pipeline_rejects_a_document_without_parseable_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An empty extraction fails instead of writing an empty success artifact."""
    empty_source = DocumentSource("source.docx", "source-hash", ())
    empty_images = DocxImageExtraction("source.docx", "source-hash", ())
    empty_attachments = DocxAttachmentExtraction("source.docx", "source-hash", ())
    monkeypatch.setattr(
        pipeline,
        "_extract_parser_inputs",
        lambda path: (empty_source, empty_images, empty_attachments),
    )

    with pytest.raises(ParserPipelineError, match="no text, tables, images, or attachments"):
        ParserPipeline(_settings(), FakeCaptionClient()).parse(tmp_path / "source.docx")
