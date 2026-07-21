"""Command-line interface for ground-truth source extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from legal_rag.source_blocks import extract_source_blocks, write_source_blocks


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Extract deterministic Docling source blocks.")
    parser.add_argument("input_path", type=Path, help="Source document path")
    parser.add_argument("output_path", type=Path, help="Destination JSONL path")
    return parser


def main() -> None:
    """Extract one source document into JSONL source blocks."""
    arguments = _parser().parse_args()
    blocks = extract_source_blocks(arguments.input_path)
    write_source_blocks(arguments.output_path, blocks)
    logger.info(
        "Extracted {} source blocks from {} to {}",
        len(blocks),
        arguments.input_path.name,
        arguments.output_path,
    )


if __name__ == "__main__":
    main()
