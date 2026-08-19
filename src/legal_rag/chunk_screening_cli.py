"""Command-line interface for the chunk-screening gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the chunk-screening command-line interface."""
    parser = argparse.ArgumentParser(description="Screen one document's chunks for penalty content.")
    parser.add_argument("chunks_path", type=Path, help="Chunk JSONL path for one document")
    parser.add_argument("screening_output_path", type=Path, help="Destination chunk-screening JSONL path")
    parser.add_argument("--synthesis-config", type=Path, required=True, help="Versioned synthesis config JSON path")
    return parser


def main() -> None:
    """Screen chunks not yet in the output file, writing progress after every batch so it survives being interrupted."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.chunk_screening import BamlChunkScreeningClient
    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.document_preparation import prepare_chunk_screening
    from legal_rag.benchmark_synthesis.stage_cache import (
        build_chunk_screening_fingerprint,
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
    fingerprint_path = stage_cache_path(arguments.screening_output_path)
    initialize_stage_cache(
        fingerprint_path,
        (arguments.screening_output_path,),
        build_chunk_screening_fingerprint(
            chunks,
            synthesis_settings,
            llm_base_url=settings.llm_base_url,
            llm_model=settings.llm_model,
        ),
    )
    records = prepare_chunk_screening(
        chunks,
        arguments.screening_output_path,
        lambda: BamlChunkScreeningClient(settings, synthesis_settings),
        synthesis_settings,
    )
    accepted_count = sum(1 for record in records if record.has_penalty_content)
    logger.info(
        "Chunk screening gate: {} accepted, {} excluded, {} still unscreened, wrote {}",
        accepted_count,
        len(records) - accepted_count,
        len(chunks) - len(records),
        arguments.screening_output_path,
    )


if __name__ == "__main__":
    main()
