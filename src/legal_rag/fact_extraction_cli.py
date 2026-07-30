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
    parser.add_argument("facts_output_path", type=Path, help="Destination legal-facts JSONL path")
    return parser


def main() -> None:
    """Extract legal facts from chunks not yet processed, writing progress after every target."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.fact_extraction import BamlFactExtractionClient, extract_facts_from_document
    from legal_rag.benchmark_synthesis.models import LegalFact
    from legal_rag.benchmark_synthesis.writer import read_chunk_screening, read_legal_facts, write_legal_facts
    from legal_rag.chunker.writer import read_chunks
    from legal_rag.settings import EvaluationSettings, Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    evaluation_settings = EvaluationSettings.from_environment()

    chunks = read_chunks(arguments.chunks_path)
    screening_records = read_chunk_screening(arguments.screening_path)
    accepted_chunk_ids = {record.chunk_id for record in screening_records if record.has_penalty_content}
    accepted_chunks = tuple(chunk for chunk in chunks if chunk.chunk_id in accepted_chunk_ids)

    existing_facts = read_legal_facts(arguments.facts_output_path) if arguments.facts_output_path.exists() else ()
    facts_by_id = {fact.fact_id: fact for fact in existing_facts}
    processed_chunk_ids = {fact.source_chunk_id for fact in existing_facts if fact.source_chunk_id is not None}
    remaining_targets = tuple(chunk for chunk in accepted_chunks if chunk.chunk_id not in processed_chunk_ids)
    logger.info(
        "Loaded {} accepted chunks of {} from {}; {} already extracted, {} remaining",
        len(accepted_chunks),
        len(chunks),
        arguments.screening_path,
        len(accepted_chunks) - len(remaining_targets),
        len(remaining_targets),
    )

    def _persist_target(chunk_id: str, target_facts: Sequence[LegalFact]) -> None:
        """Merge one target's extracted facts into the running results and rewrite the output file."""
        facts_by_id.update({fact.fact_id: fact for fact in target_facts})
        processed_chunk_ids.add(chunk_id)
        write_legal_facts(arguments.facts_output_path, tuple(facts_by_id.values()))
        logger.debug("Persisted {} fact(s) for chunk {}", len(target_facts), chunk_id)

    if remaining_targets:
        extraction_client = BamlFactExtractionClient(settings, evaluation_settings)
        extract_facts_from_document(remaining_targets, chunks, extraction_client, on_target_processed=_persist_target)

    facts = tuple(facts_by_id.values())
    write_legal_facts(arguments.facts_output_path, facts)
    extracted_chunk_count = len(processed_chunk_ids & accepted_chunk_ids)
    logger.info(
        "Extracted {} legal facts from {}/{} accepted chunks ({} still unextracted), wrote {}",
        len(facts),
        extracted_chunk_count,
        len(accepted_chunks),
        len(accepted_chunks) - extracted_chunk_count,
        arguments.facts_output_path,
    )


if __name__ == "__main__":
    main()
