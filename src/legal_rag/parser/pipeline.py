"""Orchestration for the complete document parser pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.parser.attachment_documents import ingest_attachment_documents
from legal_rag.parser.attachments import DocxAttachmentExtraction, extract_package_attachments
from legal_rag.parser.document_source import DocumentSource, extract_package_document_source
from legal_rag.parser.docx_package import DocxPackage
from legal_rag.parser.image_captioner import CaptionClient, caption_images
from legal_rag.parser.images import DocxImageExtraction, extract_package_images
from legal_rag.parser.models import ParsedDocument
from legal_rag.settings import Settings


class ParserPipelineError(ValueError):
    """Raised when the parser cannot produce a valid output document."""

    @classmethod
    def no_content(cls) -> ParserPipelineError:
        """Build an error for a document without parseable content."""
        return cls("Document parser found no text, tables, images, or attachments")


@dataclass(frozen=True, slots=True)
class ParserPipeline:
    """Parse and enrich one document into the final persisted model."""

    settings: Settings
    caption_client: CaptionClient

    def parse(
        self,
        input_path: Path,
        *,
        renderer_executable: Path | None = None,
    ) -> ParsedDocument:
        """Run native extraction and image captioning without intermediate files."""
        return self._parse(input_path, renderer_executable=renderer_executable, ingest_children=True)

    def _parse(
        self,
        input_path: Path,
        *,
        renderer_executable: Path | None,
        ingest_children: bool,
    ) -> ParsedDocument:
        """Parse one DOCX, optionally ingesting one attachment-child level."""
        source, images, attachments = _extract_parser_inputs(input_path)
        if not source.units and not images.images and not attachments.attachments:
            raise ParserPipelineError.no_content()
        captions = caption_images(
            images,
            source,
            self.settings,
            self.caption_client,
            renderer_executable=renderer_executable,
        )
        child_documents = ()
        if ingest_children:
            child_documents = ingest_attachment_documents(
                attachments.attachments,
                self.settings,
                lambda child_path: self._parse(
                    child_path,
                    renderer_executable=renderer_executable,
                    ingest_children=False,
                ),
                renderer_executable=renderer_executable,
            )
        return ParsedDocument.from_source(
            source,
            captions,
            attachments.attachments,
            child_documents,
        )


def _extract_parser_inputs(
    input_path: Path,
) -> tuple[DocumentSource, DocxImageExtraction, DocxAttachmentExtraction]:
    """Extract all native parser inputs from one shared DOCX package."""
    with DocxPackage.open(input_path) as package:
        return (
            extract_package_document_source(package),
            extract_package_images(package),
            extract_package_attachments(package),
        )
