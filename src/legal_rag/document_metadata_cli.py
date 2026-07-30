"""Command-line interface for resolving one document's number and effective dates."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the document-metadata command-line interface."""
    parser = argparse.ArgumentParser(description="Resolve one document's number and effective dates from its chunks.")
    parser.add_argument("chunks_path", type=Path, help="Chunk JSONL path for one document")
    parser.add_argument("metadata_output_path", type=Path, help="Destination document-metadata JSONL path")
    parser.add_argument("--synthesis-config", type=Path, required=True, help="Versioned synthesis config JSON path")
    return parser


def main() -> None:
    """Resolve one document's metadata with a single language-model call and persist it."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.document_metadata import BamlDocumentMetadataClient, resolve_document_metadata
    from legal_rag.benchmark_synthesis.writer import read_document_metadata, write_document_metadata
    from legal_rag.chunker.writer import read_chunks
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    synthesis_settings = SynthesisSettings.from_path(arguments.synthesis_config)

    if arguments.metadata_output_path.exists():
        record = read_document_metadata(arguments.metadata_output_path)
        logger.info("Document metadata already complete for {}, skipping LLM call", record.document_number)
        return

    chunks = read_chunks(arguments.chunks_path)
    client = BamlDocumentMetadataClient(settings, synthesis_settings)
    record = resolve_document_metadata(
        chunks,
        client,
        opening_chunk_count=synthesis_settings.document_metadata_opening_chunk_count,
        closing_chunk_count=synthesis_settings.document_metadata_closing_chunk_count,
        max_retries=synthesis_settings.llm_max_retries,
    )
    write_document_metadata(arguments.metadata_output_path, record)
    logger.info(
        "Resolved document {} (valid_from={}, valid_to={}), wrote {}",
        record.document_number,
        record.valid_from,
        record.valid_to,
        arguments.metadata_output_path,
    )


if __name__ == "__main__":
    main()
