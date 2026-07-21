"""Command-line interface for ground-truth source extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from legal_rag.docx_paragraphs import extract_docx_paragraphs, write_docx_paragraphs
from legal_rag.normalization import select_content_blocks, write_normalized_source_blocks
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
    return parser


def main() -> None:
    """Extract one source document into JSONL source blocks."""
    parser = _parser()
    arguments = parser.parse_args()
    if (arguments.normalized_output is None) != (arguments.normalization_manifest is None):
        parser.error("--normalized-output and --normalization-manifest must be provided together")
    if (arguments.paragraph_output is None) != (arguments.paragraph_manifest is None):
        parser.error("--paragraph-output and --paragraph-manifest must be provided together")

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


if __name__ == "__main__":
    main()
