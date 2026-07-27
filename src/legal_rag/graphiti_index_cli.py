"""Command-line interface for indexing chunked JSONL data into the Graphiti knowledge graph."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the Graphiti indexing command-line interface."""
    parser = argparse.ArgumentParser(
        description="Embed and index chunked JSONL data into the Graphiti temporal knowledge graph."
    )
    parser.add_argument("chunks_dir", type=Path, help="Directory containing one or more *.chunks.jsonl files")
    parser.add_argument("--corpus-config", type=Path, required=True, help="Ground-truth corpus config JSON path")
    return parser


async def _run(chunks_dir: Path, corpus_config_path: Path) -> None:
    """Load chunks and corpus metadata, then index every chunk as one Graphiti episode.

    Builds one Graphiti client per configured LLM API key (``LLM_API_KEY`` plus any
    ``LLM_API_KEY_2..N``) and indexes concurrently through a shared work queue when more than
    one key is available, so a client done with a short document keeps helping instead of
    sitting idle — see ``graphiti_rag.indexer.index_chunks_concurrent``.
    """
    from legal_rag.embedding.profiles import load_configured_embedding_profile
    from legal_rag.graphiti_rag.client import build_graphiti, load_llm_api_keys
    from legal_rag.graphiti_rag.effective_date import BamlEffectiveDateClient
    from legal_rag.graphiti_rag.embedder import QwenGraphitiEmbedder
    from legal_rag.graphiti_rag.indexer import (
        build_neo4j_driver,
        index_chunks,
        index_chunks_concurrent,
        load_corpus_metadata,
    )
    from legal_rag.naive_rag.indexer import load_chunks
    from legal_rag.settings import GraphitiSettings, Settings

    settings = Settings.from_environment()
    graphiti_settings = GraphitiSettings.from_environment()

    chunks = load_chunks(chunks_dir)
    corpus_metadata = load_corpus_metadata(corpus_config_path)
    profile = load_configured_embedding_profile(graphiti_settings.embedding_profile)
    embedder = QwenGraphitiEmbedder.from_profile(profile)
    date_extraction_client = BamlEffectiveDateClient(settings, graphiti_settings)

    api_keys = load_llm_api_keys()
    graphiti_clients = [
        build_graphiti(settings, graphiti_settings, profile, api_key=key, embedder=embedder) for key in api_keys
    ]
    neo4j_driver = build_neo4j_driver(graphiti_settings)
    try:
        await graphiti_clients[0].build_indices_and_constraints()
        if len(graphiti_clients) > 1:
            logger.info("Indexing with {} concurrent LLM API keys", len(graphiti_clients))
            summary = await index_chunks_concurrent(
                chunks, corpus_metadata, graphiti_clients, neo4j_driver, date_extraction_client
            )
        else:
            summary = await index_chunks(
                chunks, corpus_metadata, graphiti_clients[0], neo4j_driver, date_extraction_client
            )
    finally:
        for graphiti_client in graphiti_clients:
            await graphiti_client.close()
        await neo4j_driver.close()

    logger.info(
        "Indexed {} episodes into group '{}' ({} skipped: {} no-metadata, {} no-date, {} already-indexed, {} error)",
        summary.indexed,
        corpus_metadata.group_id,
        summary.skipped_no_metadata + summary.skipped_no_date + summary.skipped_already_indexed + summary.skipped_error,
        summary.skipped_no_metadata,
        summary.skipped_no_date,
        summary.skipped_already_indexed,
        summary.skipped_error,
    )


def main() -> None:
    """Load chunks, embed them with the configured profile, and index them into Graphiti."""
    from dotenv import find_dotenv, load_dotenv

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    asyncio.run(_run(arguments.chunks_dir, arguments.corpus_config))


if __name__ == "__main__":
    main()
