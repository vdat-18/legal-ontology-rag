"""Command-line interface for legal-fact extraction from one document's chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the fact-extraction command-line interface."""
    parser = argparse.ArgumentParser(description="Extract legal facts from one document's chunk JSONL.")
    parser.add_argument("chunks_path", type=Path, help="Chunk JSONL path for one document")
    parser.add_argument("facts_output_path", type=Path, help="Destination legal-facts JSONL path")
    parser.add_argument("ledger_path", type=Path, help="Fact-ledger JSONL path, read if present and rewritten")
    return parser


def main() -> None:
    """Run chunk-quality filtering, fact extraction, and ledger reconciliation for one document."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.chunk_quality import BamlChunkQualityClient, filter_chunks_by_quality
    from legal_rag.benchmark_synthesis.fact_extraction import BamlFactExtractionClient, extract_facts_from_document
    from legal_rag.benchmark_synthesis.fact_ledger import queue_for_expert_review, reconcile_ledger, verify_spans
    from legal_rag.benchmark_synthesis.writer import read_fact_ledger, write_fact_ledger, write_legal_facts
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
    quality_client = BamlChunkQualityClient(settings, evaluation_settings)
    accepted_chunks, excluded = filter_chunks_by_quality(
        chunks, quality_client, threshold=evaluation_settings.chunk_quality_threshold
    )
    logger.info(
        "Chunk quality gate: {} accepted, {} excluded from {}",
        len(accepted_chunks),
        len(excluded),
        arguments.chunks_path.name,
    )

    profile = load_configured_embedding_profile(evaluation_settings.embedding_profile)
    encoder = EmbeddingEncoder.from_profile(profile)
    extraction_client = BamlFactExtractionClient(settings, evaluation_settings)
    facts, rejections = extract_facts_from_document(accepted_chunks, extraction_client, encoder)
    logger.info("Extracted {} legal facts, rejected {} candidates", len(facts), len(rejections))

    write_legal_facts(arguments.facts_output_path, facts)

    facts_by_id = {fact.fact_id: fact for fact in facts}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in accepted_chunks}
    existing_entries = {entry.fact_id: entry for entry in read_fact_ledger(arguments.ledger_path)}
    reconciled = reconcile_ledger(facts, existing_entries)

    advanced_entries = []
    for entry in reconciled:
        if entry.status != "extracted":
            advanced_entries.append(entry)
            continue
        verified = verify_spans(
            entry, facts_by_id[entry.fact_id], chunks_by_id, threshold=evaluation_settings.span_match_threshold
        )
        advanced_entries.append(queue_for_expert_review(verified) if verified.status == "span_verified" else verified)

    write_fact_ledger(arguments.ledger_path, advanced_entries)
    needs_review = sum(1 for entry in advanced_entries if entry.status == "needs_expert_review")
    logger.info(
        "Wrote {} facts to {} and {} ledger entries to {} ({} awaiting expert review)",
        len(facts),
        arguments.facts_output_path,
        len(advanced_entries),
        arguments.ledger_path,
        needs_review,
    )


if __name__ == "__main__":
    main()
