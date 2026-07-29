"""Command-line interface for legal-fact extraction from one document's chunks."""

from __future__ import annotations

import argparse
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
    """Extract legal facts from the chunks accepted by chunk screening."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.fact_extraction import BamlFactExtractionClient, extract_facts_from_document
    from legal_rag.benchmark_synthesis.writer import read_chunk_screening, write_legal_facts
    from legal_rag.chunker.writer import read_chunks
    from legal_rag.embedding.encoder import EmbeddingEncoder
    from legal_rag.embedding.profiles import load_configured_embedding_profile
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
    logger.info("Loaded {} accepted chunks of {} from {}", len(accepted_chunks), len(chunks), arguments.screening_path)

    profile = load_configured_embedding_profile(evaluation_settings.embedding_profile)
    encoder = EmbeddingEncoder.from_profile(profile)
    extraction_client = BamlFactExtractionClient(settings, evaluation_settings)
    facts = extract_facts_from_document(accepted_chunks, extraction_client, encoder)
    logger.info("Extracted {} legal facts", len(facts))

    write_legal_facts(arguments.facts_output_path, facts)
    logger.info("Wrote {} facts to {}", len(facts), arguments.facts_output_path)


if __name__ == "__main__":
    main()
