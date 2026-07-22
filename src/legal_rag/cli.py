"""Command-line interface for ground-truth source extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


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
    from legal_rag.parser.normalization import select_content_blocks, write_normalized_source_blocks
    from legal_rag.parser.paragraphs import extract_docx_paragraphs, write_docx_paragraphs
    from legal_rag.parser.pdf_sidecar import PdfSidecar, extract_pdf_blocks, render_docx_pdf, write_pdf_sidecar
    from legal_rag.parser.source_blocks import extract_source_blocks, write_source_blocks
    from legal_rag.parser.tables import extract_docx_tables, write_docx_tables

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


def _document_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the enriched document pipeline."""
    parser = argparse.ArgumentParser(description="Parse and enrich one DOCX into the final source view.")
    parser.add_argument("input_path", type=Path, help="Source DOCX path")
    parser.add_argument("output_path", type=Path, help="Destination JSONL path for enriched source units")
    parser.add_argument(
        "--renderer-executable",
        type=Path,
        help="LibreOffice console executable used to rasterize legacy EMF/WMF assets",
    )
    return parser


def parse_document_main() -> None:
    """Run the complete parser pipeline and write one enriched output."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.parser.config import ParserSettings
    from legal_rag.parser.image_captioner import BamlCaptionClient
    from legal_rag.parser.pipeline import ParserPipeline
    from legal_rag.parser.writer import write_parsed_document

    arguments = _document_parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = ParserSettings.from_environment()
    pipeline = ParserPipeline(settings, BamlCaptionClient(settings))
    document = pipeline.parse(
        arguments.input_path,
        renderer_executable=arguments.renderer_executable,
    )
    write_parsed_document(arguments.output_path, document)
    logger.info(
        "Wrote {} enriched source units with {} image captions from {} to {}",
        len(document.units),
        document.image_count,
        arguments.input_path.name,
        arguments.output_path,
    )


if __name__ == "__main__":
    main()
