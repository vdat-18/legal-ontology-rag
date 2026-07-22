"""Orchestration for the complete document parser pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.parser.config import ParserSettings
from legal_rag.parser.document_source import extract_document_source
from legal_rag.parser.image_captioner import CaptionClient, caption_images
from legal_rag.parser.images import extract_docx_images
from legal_rag.parser.models import ParsedDocument


class ParserPipelineError(ValueError):
    """Raised when parser stages disagree about source provenance."""

    @classmethod
    def source_mismatch(cls) -> ParserPipelineError:
        """Build an error for extraction stages reading different source bytes."""
        return cls("Document and image extraction stages read different source bytes")

    @classmethod
    def no_content(cls) -> ParserPipelineError:
        """Build an error for a document without parseable text, tables, or images."""
        return cls("Document parser found no text, tables, or images")


@dataclass(frozen=True, slots=True)
class ParserPipeline:
    """Parse and enrich one document into the final persisted model."""

    settings: ParserSettings
    caption_client: CaptionClient

    def parse(
        self,
        input_path: Path,
        *,
        renderer_executable: Path | None = None,
    ) -> ParsedDocument:
        """Run native extraction and image captioning without intermediate files."""
        source = extract_document_source(input_path)
        images = extract_docx_images(input_path)
        if images.source_sha256 != source.source_sha256:
            raise ParserPipelineError.source_mismatch()
        if not source.units and not images.images:
            raise ParserPipelineError.no_content()
        captions = caption_images(
            images,
            source,
            self.settings,
            self.caption_client,
            renderer_executable=renderer_executable,
        )
        return ParsedDocument.from_source(source, captions)
