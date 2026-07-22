"""Command-line interface for ground-truth source extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from legal_rag.document_source import extract_document_source, write_document_source
from legal_rag.docx_paragraphs import extract_docx_paragraphs, write_docx_paragraphs
from legal_rag.docx_tables import extract_docx_tables, write_docx_tables
from legal_rag.image_captioning import (
    BamlCaptionClient,
    ImageCaptionSettings,
    caption_docx_document,
    write_image_captions,
)
from legal_rag.normalization import select_content_blocks, write_normalized_source_blocks
from legal_rag.pdf_sidecar import PdfSidecar, extract_pdf_blocks, render_docx_pdf, write_pdf_sidecar
from legal_rag.source_blocks import extract_source_blocks, write_source_blocks


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Extract deterministic Docling source blocks.")
    parser.add_argument("input_path", type=Path, help="Source document path")
    parser.add_argument("output_path", type=Path, help="Destination JSONL path")
    parser.add_argument("--normalized-output", type=Path, help="JSONL path excluding empty text items")
    parser.add_argument("--normalization-manifest", type=Path, help="JSON path recording skipped empty text items")
    parser.add_argument("--paragraph-output", type=Path, help="JSONL path for native DOCX paragraph units")
    parser.add_argument("--paragraph-manifest", type=Path, help="JSON path for the paragraph extraction audit")
    parser.add_argument("--table-output", type=Path, help="JSONL path for native DOCX table units")
    parser.add_argument("--table-manifest", type=Path, help="JSON path for the table extraction audit")
    parser.add_argument("--pdf-output", type=Path, help="Rendered PDF sidecar path")
    parser.add_argument("--pdf-block-output", type=Path, help="JSONL path for page-aware PDF blocks")
    parser.add_argument("--pdf-manifest", type=Path, help="JSON path for DOCX-to-PDF provenance")
    parser.add_argument("--renderer-executable", type=Path, help="LibreOffice console executable path")
    parser.add_argument("--render-timeout-seconds", type=float, help="Positive renderer process timeout")
    return parser


def main() -> None:
    """Extract one source document into JSONL source blocks."""
    parser = _parser()
    arguments = parser.parse_args()
    if (arguments.normalized_output is None) != (arguments.normalization_manifest is None):
        parser.error("--normalized-output and --normalization-manifest must be provided together")
    if (arguments.paragraph_output is None) != (arguments.paragraph_manifest is None):
        parser.error("--paragraph-output and --paragraph-manifest must be provided together")
    if (arguments.table_output is None) != (arguments.table_manifest is None):
        parser.error("--table-output and --table-manifest must be provided together")
    pdf_arguments = (
        arguments.pdf_output,
        arguments.pdf_block_output,
        arguments.pdf_manifest,
        arguments.renderer_executable,
        arguments.render_timeout_seconds,
    )
    if any(value is not None for value in pdf_arguments) and any(value is None for value in pdf_arguments):
        parser.error(
            "--pdf-output, --pdf-block-output, --pdf-manifest, --renderer-executable, "
            "and --render-timeout-seconds must be provided together"
        )

    blocks = extract_source_blocks(arguments.input_path)
    write_source_blocks(arguments.output_path, blocks)
    logger.info(
        "Extracted {} source blocks from {} to {}",
        len(blocks),
        arguments.input_path.name,
        arguments.output_path,
    )
    if arguments.normalized_output is not None and arguments.normalization_manifest is not None:
        selection = select_content_blocks(blocks)
        write_normalized_source_blocks(
            arguments.normalized_output,
            arguments.normalization_manifest,
            selection,
        )
        logger.info(
            "Wrote {} normalized blocks; skipped {} empty text items",
            len(selection.blocks),
            len(selection.skipped_empty_text_refs),
        )
    if arguments.paragraph_output is not None and arguments.paragraph_manifest is not None:
        paragraphs = extract_docx_paragraphs(arguments.input_path)
        write_docx_paragraphs(arguments.paragraph_output, arguments.paragraph_manifest, paragraphs)
        logger.info(
            "Wrote {} native DOCX paragraph units; skipped {} empty paragraphs",
            len(paragraphs.units),
            len(paragraphs.skipped_empty_paragraph_refs),
        )
    if arguments.table_output is not None and arguments.table_manifest is not None:
        tables = extract_docx_tables(arguments.input_path)
        write_docx_tables(arguments.table_output, arguments.table_manifest, tables)
        logger.info(
            "Wrote {} native DOCX table units with {} cells",
            len(tables.units),
            tables.cell_count,
        )
    if all(value is not None for value in pdf_arguments):
        rendered = render_docx_pdf(
            arguments.input_path,
            arguments.pdf_output,
            renderer_executable=arguments.renderer_executable,
            timeout_seconds=arguments.render_timeout_seconds,
        )
        parsed = extract_pdf_blocks(arguments.pdf_output)
        write_pdf_sidecar(
            arguments.pdf_block_output,
            arguments.pdf_manifest,
            PdfSidecar(rendered=rendered, parsed=parsed),
        )
        logger.info(
            "Wrote {} page-aware PDF blocks across {} pages",
            len(parsed.blocks),
            parsed.page_count,
        )


def _document_source_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the merged source view."""
    parser = argparse.ArgumentParser(
        description="Merge native DOCX paragraphs and tables into one ordered source view."
    )
    parser.add_argument("input_path", type=Path, help="Source DOCX path")
    parser.add_argument("output_path", type=Path, help="Destination JSONL path for merged source units")
    parser.add_argument("--manifest", type=Path, help="Optional JSON path for an audit manifest")
    return parser


def parse_document_main() -> None:
    """Extract one DOCX into a single native-order source view."""
    parser = _document_source_parser()
    arguments = parser.parse_args()

    source = extract_document_source(arguments.input_path)
    write_document_source(arguments.output_path, source, manifest_path=arguments.manifest)
    logger.info(
        "Wrote {} source units ({} paragraphs, {} tables) from {} to {}",
        len(source.units),
        source.paragraph_count,
        source.table_count,
        arguments.input_path.name,
        arguments.output_path,
    )


def _image_caption_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for direct DOCX image captioning."""
    parser = argparse.ArgumentParser(description="Caption one DOCX through the configured multimodal LLM API.")
    parser.add_argument("input_path", type=Path, help="Source DOCX path")
    parser.add_argument("output_path", type=Path, help="Destination JSONL path for validated captions")
    parser.add_argument(
        "--renderer-executable",
        type=Path,
        help="LibreOffice console executable used to rasterize legacy EMF/WMF assets",
    )
    return parser


def caption_document_images_main() -> None:
    """Extract and caption one DOCX through 9Router without intermediate files."""
    from dotenv import find_dotenv, load_dotenv

    arguments = _image_caption_parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = ImageCaptionSettings.from_environment()
    client = BamlCaptionClient(settings)
    captions = caption_docx_document(
        arguments.input_path,
        settings,
        client,
        renderer_executable=arguments.renderer_executable,
    )
    write_image_captions(arguments.output_path, captions)
    logger.info(
        "Wrote {} validated image captions for {} using requested model {}",
        len(captions),
        arguments.input_path.name,
        settings.model,
    )


if __name__ == "__main__":
    main()
