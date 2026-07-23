"""Command-line interface for the enriched document parser."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the parser-pipeline command-line interface."""
    parser = argparse.ArgumentParser(description="Parse and enrich one DOCX into the final source view.")
    parser.add_argument("input_path", type=Path, help="Source DOCX path")
    parser.add_argument("output_path", type=Path, help="Destination JSONL path for enriched source units")
    parser.add_argument(
        "--renderer-executable",
        type=Path,
        help="LibreOffice console executable used to rasterize legacy EMF/WMF assets",
    )
    return parser


def main() -> None:
    """Run the complete parser pipeline and write one enriched output."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.parser.image_captioner import BamlCaptionClient
    from legal_rag.parser.pipeline import ParserPipeline
    from legal_rag.parser.writer import write_parsed_document
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    pipeline = ParserPipeline(settings, BamlCaptionClient(settings))
    document = pipeline.parse(
        arguments.input_path,
        renderer_executable=arguments.renderer_executable,
    )
    write_parsed_document(arguments.output_path, document)
    logger.info(
        "Wrote {} enriched source units with {} image captions and {} attachments from {} to {}",
        len(document.units),
        document.image_count,
        document.attachment_count,
        arguments.input_path.name,
        arguments.output_path,
    )


if __name__ == "__main__":
    main()
