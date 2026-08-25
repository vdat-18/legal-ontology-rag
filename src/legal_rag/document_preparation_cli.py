"""Command-line interface for sequential screening and metadata preparation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the document-preparation command-line interface."""
    parser = argparse.ArgumentParser(description="Prepare screening before document metadata.")
    parser.add_argument("chunks_path", type=Path, help="Chunk JSONL path for one document")
    parser.add_argument("screening_output_path", type=Path, help="Destination chunk-screening JSONL path")
    parser.add_argument("metadata_output_path", type=Path, help="Destination document-metadata JSONL path")
    parser.add_argument("--synthesis-config", type=Path, required=True, help="Versioned synthesis config JSON path")
    parser.add_argument("--rescreen-chunk-id", action="append", default=[], help="Chunk id to screen again")
    parser.add_argument("--rescreen-chunk-ids", type=Path, help="JSON file containing chunk ids to screen again")
    return parser


def main() -> None:
    """Prepare chunk screening and then document metadata."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.chunk_screening import (
        CHUNK_SCREENING_PROMPT_VERSION,
        BamlChunkScreeningClient,
        ChunkScreeningRecord,
    )
    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.document_metadata import BamlDocumentMetadataClient
    from legal_rag.benchmark_synthesis.document_preparation import prepare_document_artifacts
    from legal_rag.benchmark_synthesis.screening_selection import (
        load_rescreen_chunk_ids,
        validate_selective_screening,
    )
    from legal_rag.benchmark_synthesis.stage_cache import (
        build_chunk_screening_artifact_fingerprint,
        build_chunk_screening_fingerprint,
        build_document_metadata_fingerprint,
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

    execution_fingerprint = build_chunk_screening_fingerprint(
        chunks,
        synthesis_settings,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
    )
    screening_fingerprint_path = stage_cache_path(arguments.screening_output_path)
    rescreen_chunk_ids = load_rescreen_chunk_ids(arguments.rescreen_chunk_id, arguments.rescreen_chunk_ids)
    existing_fingerprint = None
    existing_records_by_id: dict[str, ChunkScreeningRecord] = {}
    if rescreen_chunk_ids:
        if not arguments.screening_output_path.exists() or not screening_fingerprint_path.exists():
            msg = "Selective re-screening requires an existing screening output and run fingerprint"
            raise ValueError(msg)
        existing_records = read_chunk_screening(arguments.screening_output_path)
        validate_selective_screening(chunks, existing_records, rescreen_chunk_ids)
        existing_records_by_id = {record.chunk_id: record for record in existing_records}
        existing_fingerprint = read_stage_cache_fingerprint(screening_fingerprint_path)
        write_stage_cache_fingerprint(screening_fingerprint_path, execution_fingerprint)
    elif arguments.screening_output_path.exists() and screening_fingerprint_path.exists():
        existing_records = read_chunk_screening(arguments.screening_output_path)
        if all(record.screening_run_id for record in existing_records):
            require_chunk_screening_cache(
                screening_fingerprint_path,
                chunks,
                existing_records,
                execution_fingerprint,
            )
        else:
            initialize_stage_cache(
                screening_fingerprint_path,
                (arguments.screening_output_path,),
                execution_fingerprint,
            )
        existing_fingerprint = read_stage_cache_fingerprint(screening_fingerprint_path)
    else:
        initialize_stage_cache(
            screening_fingerprint_path,
            (arguments.screening_output_path,),
            execution_fingerprint,
        )

    def _finalize_screening(records: Sequence[ChunkScreeningRecord]) -> None:
        artifact_fingerprint = build_chunk_screening_artifact_fingerprint(chunks, records)
        write_stage_cache_fingerprint(screening_fingerprint_path, artifact_fingerprint)
        metadata_fingerprint = build_document_metadata_fingerprint(
            chunks,
            synthesis_settings,
            screening_run_id=artifact_fingerprint.run_id,
            llm_base_url=settings.llm_base_url,
            llm_model=settings.llm_model,
        )
        records_by_id = {record.chunk_id: record for record in records}
        metadata_selection_changed = any(
            existing_records_by_id.get(chunk_id) is None
            or existing_records_by_id[chunk_id].has_document_metadata != records_by_id[chunk_id].has_document_metadata
            for chunk_id in rescreen_chunk_ids
        )
        metadata_fingerprint_path = stage_cache_path(arguments.metadata_output_path)
        if rescreen_chunk_ids and arguments.metadata_output_path.exists() and not metadata_selection_changed:
            write_stage_cache_fingerprint(metadata_fingerprint_path, metadata_fingerprint)
            logger.info("Step 1B metadata candidates unchanged; reusing existing metadata")
            return
        if rescreen_chunk_ids and arguments.metadata_output_path.exists():
            arguments.metadata_output_path.unlink()
        initialize_stage_cache(
            metadata_fingerprint_path,
            (arguments.metadata_output_path,),
            metadata_fingerprint,
        )

    records, metadata = prepare_document_artifacts(
        chunks,
        arguments.screening_output_path,
        arguments.metadata_output_path,
        lambda: BamlChunkScreeningClient(settings, synthesis_settings),
        lambda: BamlDocumentMetadataClient(settings, synthesis_settings),
        synthesis_settings,
        rescreen_chunk_ids=rescreen_chunk_ids,
        prompt_version=CHUNK_SCREENING_PROMPT_VERSION,
        model=settings.llm_model,
        screening_run_id=execution_fingerprint.run_id,
        existing_prompt_version=existing_fingerprint.prompt_version if existing_fingerprint else None,
        existing_model=existing_fingerprint.model if existing_fingerprint else None,
        existing_screening_run_id=existing_fingerprint.run_id if existing_fingerprint else None,
        on_screening_complete=_finalize_screening,
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
