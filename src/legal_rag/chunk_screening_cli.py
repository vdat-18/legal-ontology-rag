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
    parser.add_argument("--rescreen-chunk-id", action="append", default=[], help="Chunk id to screen again")
    parser.add_argument("--rescreen-chunk-ids", type=Path, help="JSON file containing chunk ids to screen again")
    return parser


def main() -> None:
    """Screen chunks not yet in the output file, writing progress after every batch so it survives being interrupted."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.chunk_screening import CHUNK_SCREENING_PROMPT_VERSION, BamlChunkScreeningClient
    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.document_preparation import prepare_chunk_screening
    from legal_rag.benchmark_synthesis.screening_selection import (
        load_rescreen_chunk_ids,
        validate_selective_screening,
    )
    from legal_rag.benchmark_synthesis.stage_cache import (
        build_chunk_screening_artifact_fingerprint,
        build_chunk_screening_fingerprint,
        initialize_stage_cache,
        read_stage_cache_fingerprint,
        require_chunk_screening_cache,
        stage_cache_path,
        write_stage_cache_fingerprint,
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
    fingerprint_path = stage_cache_path(arguments.screening_output_path)
    execution_fingerprint = build_chunk_screening_fingerprint(
        chunks,
        synthesis_settings,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
    )
    rescreen_chunk_ids = load_rescreen_chunk_ids(arguments.rescreen_chunk_id, arguments.rescreen_chunk_ids)
    existing_fingerprint = None
    if rescreen_chunk_ids:
        if not arguments.screening_output_path.exists() or not fingerprint_path.exists():
            msg = "Selective re-screening requires an existing screening output and run fingerprint"
            raise ValueError(msg)
        validate_selective_screening(
            chunks,
            read_chunk_screening(arguments.screening_output_path),
            rescreen_chunk_ids,
        )
        existing_fingerprint = read_stage_cache_fingerprint(fingerprint_path)
        write_stage_cache_fingerprint(fingerprint_path, execution_fingerprint)
    elif arguments.screening_output_path.exists() and fingerprint_path.exists():
        existing_records = read_chunk_screening(arguments.screening_output_path)
        if all(record.screening_run_id for record in existing_records):
            require_chunk_screening_cache(fingerprint_path, chunks, existing_records, execution_fingerprint)
        else:
            initialize_stage_cache(fingerprint_path, (arguments.screening_output_path,), execution_fingerprint)
        existing_fingerprint = read_stage_cache_fingerprint(fingerprint_path)
    else:
        initialize_stage_cache(fingerprint_path, (arguments.screening_output_path,), execution_fingerprint)

    records = prepare_chunk_screening(
        chunks,
        arguments.screening_output_path,
        lambda: BamlChunkScreeningClient(settings, synthesis_settings),
        synthesis_settings,
        rescreen_chunk_ids=rescreen_chunk_ids,
        prompt_version=CHUNK_SCREENING_PROMPT_VERSION,
        model=settings.llm_model,
        screening_run_id=execution_fingerprint.run_id,
        existing_prompt_version=existing_fingerprint.prompt_version if existing_fingerprint else None,
        existing_model=existing_fingerprint.model if existing_fingerprint else None,
        existing_screening_run_id=existing_fingerprint.run_id if existing_fingerprint else None,
    )
    artifact_fingerprint = build_chunk_screening_artifact_fingerprint(chunks, records)
    write_stage_cache_fingerprint(fingerprint_path, artifact_fingerprint)
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
