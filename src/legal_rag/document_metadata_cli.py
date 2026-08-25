"""Command-line interface for resolving one document's number and effective dates."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the document-metadata command-line interface."""
    parser = argparse.ArgumentParser(description="Resolve one document's number and effective dates from its chunks.")
    parser.add_argument("chunks_path", type=Path, help="Chunk JSONL path for one document")
    parser.add_argument("screening_path", type=Path, help="Chunk-screening JSONL path from legal-screen-chunks")
    parser.add_argument("metadata_output_path", type=Path, help="Destination document-metadata JSONL path")
    parser.add_argument("--synthesis-config", type=Path, required=True, help="Versioned synthesis config JSON path")
    return parser


def main() -> None:
    """Resolve one document's metadata with a single language-model call and persist it."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.document_metadata import BamlDocumentMetadataClient
    from legal_rag.benchmark_synthesis.document_preparation import prepare_document_metadata
    from legal_rag.benchmark_synthesis.stage_cache import (
        build_chunk_screening_fingerprint,
        build_document_metadata_fingerprint,
        initialize_stage_cache,
        require_chunk_screening_cache,
        stage_cache_path,
    )
    from legal_rag.benchmark_synthesis.writer import read_chunk_screening
    from legal_rag.chunker.writer import read_chunks
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    synthesis_settings = SynthesisSettings.from_path(arguments.synthesis_config)

    chunks = read_chunks(arguments.chunks_path)
    screening_records = read_chunk_screening(arguments.screening_path)
    screening_fingerprint = require_chunk_screening_cache(
        stage_cache_path(arguments.screening_path),
        chunks,
        screening_records,
        build_chunk_screening_fingerprint(
            chunks,
            synthesis_settings,
            llm_base_url=settings.llm_base_url,
            llm_model=settings.llm_model,
        ),
    )
    fingerprint_path = stage_cache_path(arguments.metadata_output_path)
    initialize_stage_cache(
        fingerprint_path,
        (arguments.metadata_output_path,),
        build_document_metadata_fingerprint(
            chunks,
            synthesis_settings,
            screening_run_id=screening_fingerprint.run_id,
            llm_base_url=settings.llm_base_url,
            llm_model=settings.llm_model,
        ),
    )
    was_complete = arguments.metadata_output_path.exists()
    record = prepare_document_metadata(
        chunks,
        screening_records,
        arguments.metadata_output_path,
        lambda: BamlDocumentMetadataClient(settings, synthesis_settings),
        synthesis_settings,
    )
    if was_complete:
        logger.info("Document metadata already complete for {}, skipping LLM call", record.document_number)
        return
    logger.info(
        "Resolved document {} (valid_from={}, valid_to={}), wrote {}",
        record.document_number,
        record.valid_from,
        record.valid_to,
        arguments.metadata_output_path,
    )


if __name__ == "__main__":
    main()
