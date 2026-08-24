"""Command-line interface for sequential screening and metadata preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the document-preparation command-line interface."""
    parser = argparse.ArgumentParser(description="Prepare screening before document metadata.")
    parser.add_argument("chunks_path", type=Path, help="Chunk JSONL path for one document")
    parser.add_argument("screening_output_path", type=Path, help="Destination chunk-screening JSONL path")
    parser.add_argument("metadata_output_path", type=Path, help="Destination document-metadata JSONL path")
    parser.add_argument("--synthesis-config", type=Path, required=True, help="Versioned synthesis config JSON path")
    return parser


def main() -> None:
    """Prepare chunk screening and then document metadata."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.chunk_screening import BamlChunkScreeningClient
    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.document_metadata import BamlDocumentMetadataClient
    from legal_rag.benchmark_synthesis.document_preparation import prepare_document_artifacts
    from legal_rag.benchmark_synthesis.stage_cache import (
        build_chunk_screening_fingerprint,
        build_document_metadata_fingerprint,
        initialize_stage_cache,
        stage_cache_path,
    )
    from legal_rag.chunker.writer import read_chunks
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    synthesis_settings = SynthesisSettings.from_path(arguments.synthesis_config)
    chunks = read_chunks(arguments.chunks_path)

    initialize_stage_cache(
        stage_cache_path(arguments.screening_output_path),
        (arguments.screening_output_path,),
        build_chunk_screening_fingerprint(
            chunks,
            synthesis_settings,
            llm_base_url=settings.llm_base_url,
            llm_model=settings.llm_model,
        ),
    )
    initialize_stage_cache(
        stage_cache_path(arguments.metadata_output_path),
        (arguments.metadata_output_path,),
        build_document_metadata_fingerprint(
            chunks,
            synthesis_settings,
            llm_base_url=settings.llm_base_url,
            llm_model=settings.llm_model,
        ),
    )

    records, metadata = prepare_document_artifacts(
        chunks,
        arguments.screening_output_path,
        arguments.metadata_output_path,
        lambda: BamlChunkScreeningClient(settings, synthesis_settings),
        lambda: BamlDocumentMetadataClient(settings, synthesis_settings),
        synthesis_settings,
    )
    accepted_count = sum(1 for record in records if record.has_penalty_content)
    logger.info(
        "Prepared document {}: {} accepted chunks, {} excluded chunks; wrote {} and {}",
        metadata.document_number,
        accepted_count,
        len(records) - accepted_count,
        arguments.screening_output_path,
        arguments.metadata_output_path,
    )


if __name__ == "__main__":
    main()
