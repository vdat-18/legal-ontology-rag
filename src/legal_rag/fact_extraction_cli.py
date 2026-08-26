"""Command-line interface for legal-fact extraction from one document's chunks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the fact-extraction command-line interface."""
    parser = argparse.ArgumentParser(description="Extract legal facts from one document's screened chunks.")
    parser.add_argument("chunks_path", type=Path, help="Chunk JSONL path for one document")
    parser.add_argument("screening_path", type=Path, help="Chunk-screening JSONL path from legal-screen-chunks")
    parser.add_argument(
        "metadata_path", type=Path, help="Document-metadata JSONL path from legal-resolve-document-metadata"
    )
    parser.add_argument("facts_output_path", type=Path, help="Destination legal-facts JSONL path")
    parser.add_argument("progress_output_path", type=Path, help="Destination extraction-progress JSONL path")
    parser.add_argument("--synthesis-config", type=Path, required=True, help="Versioned synthesis config JSON path")
    parser.add_argument("--reextract-chunk-id", action="append", default=[], help="Chunk id to extract again")
    parser.add_argument("--reextract-chunk-ids", type=Path, help="JSON file containing chunk ids to extract again")
    return parser


def main() -> None:
    """Extract legal facts from chunks not yet processed, writing progress after every batch."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.fact_extraction import (
        FACT_EXTRACTION_PROMPT_VERSION,
        BamlFactExtractionClient,
        ExtractionProgressRecord,
        extract_facts_from_document,
    )
    from legal_rag.benchmark_synthesis.fact_extraction_resume import prepare_extraction_resume
    from legal_rag.benchmark_synthesis.fact_extraction_selection import load_reextract_chunk_ids
    from legal_rag.benchmark_synthesis.models import LegalFact
    from legal_rag.benchmark_synthesis.stage_cache import (
        build_chunk_screening_fingerprint,
        build_document_metadata_fingerprint,
        build_fact_extraction_artifact_fingerprint,
        build_fact_extraction_fingerprint,
        require_chunk_screening_cache,
        require_stage_cache,
        stage_cache_path,
        write_stage_cache_fingerprint,
    )
    from legal_rag.benchmark_synthesis.writer import (
        read_chunk_screening,
        read_document_metadata,
        read_extraction_progress,
        read_legal_facts,
        write_extraction_progress,
        write_legal_facts,
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
    metadata_fingerprint = require_stage_cache(
        stage_cache_path(arguments.metadata_path),
        build_document_metadata_fingerprint(
            chunks,
            synthesis_settings,
            screening_run_id=screening_fingerprint.run_id,
            llm_base_url=settings.llm_base_url,
            llm_model=settings.llm_model,
        ),
    )
    accepted_chunk_ids = {record.chunk_id for record in screening_records if record.has_penalty_content}
    accepted_chunks = tuple(chunk for chunk in chunks if chunk.chunk_id in accepted_chunk_ids)
    metadata = read_document_metadata(arguments.metadata_path)
    accepted_chunk_id_order = tuple(chunk.chunk_id for chunk in accepted_chunks)
    execution_fingerprint = build_fact_extraction_fingerprint(
        chunks,
        accepted_chunk_id_order,
        metadata,
        synthesis_settings,
        screening_run_id=screening_fingerprint.run_id,
        metadata_run_id=metadata_fingerprint.run_id,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
    )
    existing_facts = read_legal_facts(arguments.facts_output_path) if arguments.facts_output_path.exists() else ()
    existing_progress = (
        read_extraction_progress(arguments.progress_output_path) if arguments.progress_output_path.exists() else ()
    )
    fingerprint_path = stage_cache_path(arguments.facts_output_path)
    reextract_chunk_ids = load_reextract_chunk_ids(
        arguments.reextract_chunk_id,
        arguments.reextract_chunk_ids,
    )
    existing_facts, existing_progress = prepare_extraction_resume(
        fingerprint_path=fingerprint_path,
        facts_path=arguments.facts_output_path,
        progress_path=arguments.progress_output_path,
        chunks=chunks,
        accepted_chunks=accepted_chunks,
        metadata=metadata,
        existing_facts=existing_facts,
        existing_progress=existing_progress,
        execution_fingerprint=execution_fingerprint,
        reextract_chunk_ids=reextract_chunk_ids,
    )

    facts_by_id: dict[str, LegalFact] = {}

    def _merge_facts(facts: Sequence[LegalFact]) -> None:
        """Merge content-addressed facts without duplicating identical identities."""
        for fact in facts:
            facts_by_id.setdefault(fact.fact_id, fact)

    _merge_facts(existing_facts)
    progress_by_chunk_id = {record.chunk_id: record for record in existing_progress}
    processed_chunk_ids = set(progress_by_chunk_id)
    remaining_targets = tuple(chunk for chunk in accepted_chunks if chunk.chunk_id not in processed_chunk_ids)
    logger.info(
        "Loaded {} accepted chunks of {} from {}; {} already extracted, {} remaining",
        len(accepted_chunks),
        len(chunks),
        arguments.screening_path,
        len(accepted_chunks) - len(remaining_targets),
        len(remaining_targets),
    )

    def _persist_batch(batch_results: Sequence[tuple[str, tuple[LegalFact, ...]]]) -> None:
        """Replace each target's prior facts before checkpointing the completed batch."""
        target_chunk_ids = {chunk_id for chunk_id, _ in batch_results}
        stale_fact_ids = [fact_id for fact_id, fact in facts_by_id.items() if fact.source_chunk_id in target_chunk_ids]
        for fact_id in stale_fact_ids:
            del facts_by_id[fact_id]
        for _, target_facts in batch_results:
            _merge_facts(target_facts)
        write_legal_facts(
            arguments.facts_output_path, tuple(sorted(facts_by_id.values(), key=lambda fact: fact.fact_id))
        )
        for chunk_id, _ in batch_results:
            processed_chunk_ids.add(chunk_id)
            progress_by_chunk_id[chunk_id] = ExtractionProgressRecord(
                chunk_id=chunk_id,
                prompt_version=FACT_EXTRACTION_PROMPT_VERSION,
                model=settings.llm_model,
                extraction_run_id=execution_fingerprint.run_id,
            )
        write_extraction_progress(
            arguments.progress_output_path,
            tuple(
                progress_by_chunk_id[chunk.chunk_id]
                for chunk in accepted_chunks
                if chunk.chunk_id in progress_by_chunk_id
            ),
        )
        logger.debug(
            "Persisted {} fact(s) for {} target chunk(s)",
            sum(len(target_facts) for _, target_facts in batch_results),
            len(batch_results),
        )

    if remaining_targets:
        extraction_client = BamlFactExtractionClient(settings, synthesis_settings)
        extract_facts_from_document(
            remaining_targets,
            chunks,
            extraction_client,
            metadata,
            max_preceding_chunks=synthesis_settings.fact_extraction_max_preceding_chunks,
            max_following_chunks=synthesis_settings.fact_extraction_max_following_chunks,
            batch_size=synthesis_settings.fact_extraction_batch_size,
            max_concurrency=synthesis_settings.fact_extraction_max_concurrency,
            extraction_target_ids=accepted_chunk_ids,
            max_retries=synthesis_settings.llm_max_retries,
            retry_delay_seconds=synthesis_settings.llm_retry_delay_seconds,
            on_batch_processed=_persist_batch,
        )

    facts = tuple(sorted(facts_by_id.values(), key=lambda fact: fact.fact_id))
    write_legal_facts(arguments.facts_output_path, facts)
    write_extraction_progress(
        arguments.progress_output_path,
        tuple(
            progress_by_chunk_id[chunk.chunk_id]
            for chunk in accepted_chunks
            if chunk.chunk_id in progress_by_chunk_id
        ),
    )
    progress = tuple(
        progress_by_chunk_id[chunk.chunk_id]
        for chunk in accepted_chunks
        if chunk.chunk_id in progress_by_chunk_id
    )
    if len(progress) == len(accepted_chunks):
        artifact_fingerprint = build_fact_extraction_artifact_fingerprint(
            chunks,
            accepted_chunk_id_order,
            metadata,
            facts,
            progress,
        )
        write_stage_cache_fingerprint(fingerprint_path, artifact_fingerprint)
    processed_chunk_count = len(processed_chunk_ids & accepted_chunk_ids)
    logger.info(
        "Extracted {} legal facts from {}/{} accepted chunks ({} still unprocessed), wrote {}",
        len(facts),
        processed_chunk_count,
        len(accepted_chunks),
        len(accepted_chunks) - processed_chunk_count,
        arguments.facts_output_path,
    )


if __name__ == "__main__":
    main()
