"""Command-line interface for indexing chunked JSONL data into the Graphiti knowledge graph."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from legal_rag.graphiti_rag.indexer import IndexingSummary

_CLEANUP_ERROR_MESSAGE = "Graphiti cleanup failed"


def _parser() -> argparse.ArgumentParser:
    """Build the Graphiti indexing command-line interface."""
    parser = argparse.ArgumentParser(
        description="Embed and index chunked JSONL data into the Graphiti temporal knowledge graph."
    )
    parser.add_argument("chunks_dir", type=Path, help="Directory containing one or more *.chunks.jsonl files")
    parser.add_argument("--corpus-config", type=Path, required=True, help="Ground-truth corpus config JSON path")
    return parser


async def _run(chunks_dir: Path, corpus_config_path: Path) -> IndexingSummary:
    """Load chunks and corpus metadata, then index every chunk as one Graphiti episode.

    Builds one Graphiti client per configured LLM API key (``LLM_API_KEY`` plus any
    ``LLM_API_KEY_2..N``) and indexes concurrently through a shared work queue when more than
    one key is available, so a client done with a short document keeps helping instead of
    sitting idle — see ``graphiti_rag.indexer.index_chunks_concurrent``.
    """
    from legal_rag.datastores.neo4j import build_neo4j_driver
    from legal_rag.embedding.profiles import load_configured_embedding_profile
    from legal_rag.graphiti_rag.client import build_graphiti, load_llm_api_keys
    from legal_rag.graphiti_rag.effective_date import BamlEffectiveDateClient
    from legal_rag.graphiti_rag.embedder import QwenGraphitiEmbedder
    from legal_rag.graphiti_rag.indexer import (
        index_chunks,
        index_chunks_concurrent,
        load_corpus_metadata,
    )
    from legal_rag.naive_rag.indexer import load_chunks
    from legal_rag.settings import GraphitiSettings, Neo4jSettings, Settings

    settings = Settings.from_environment()
    neo4j_settings = Neo4jSettings.from_environment()
    graphiti_settings = GraphitiSettings.from_environment()

    chunks = load_chunks(chunks_dir)
    corpus_metadata = load_corpus_metadata(corpus_config_path)
    profile = load_configured_embedding_profile(graphiti_settings.embedding_profile)
    embedder = QwenGraphitiEmbedder.from_profile(profile)
    date_extraction_client = BamlEffectiveDateClient(settings, graphiti_settings)

    api_keys = load_llm_api_keys()
    graphiti_clients = []
    neo4j_driver = None
    try:
        neo4j_driver = build_neo4j_driver(neo4j_settings)
        for key in api_keys:
            # Append immediately so a later construction failure cannot orphan earlier clients.
            graphiti_client = build_graphiti(
                settings,
                neo4j_settings,
                graphiti_settings,
                profile,
                api_key=key,
                embedder=embedder,
            )
            graphiti_clients.append(graphiti_client)
        await graphiti_clients[0].build_indices_and_constraints()
        window_token_budget = graphiti_settings.effective_date_window_token_budget
        if len(graphiti_clients) > 1:
            logger.info("Indexing with {} concurrent LLM API keys", len(graphiti_clients))
            summary = await index_chunks_concurrent(
                chunks, corpus_metadata, graphiti_clients, neo4j_driver, date_extraction_client, window_token_budget
            )
        else:
            summary = await index_chunks(
                chunks, corpus_metadata, graphiti_clients[0], neo4j_driver, date_extraction_client, window_token_budget
            )
    finally:
        close_operations = [graphiti_client.close() for graphiti_client in graphiti_clients]
        if neo4j_driver is not None:
            close_operations.append(neo4j_driver.close())
        cleanup_results = await asyncio.gather(*close_operations, return_exceptions=True)
        cleanup_errors = [result for result in cleanup_results if isinstance(result, BaseException)]
        active_error = sys.exception()
        if active_error is not None:
            for cleanup_error in cleanup_errors:
                active_error.add_note(f"Cleanup failed: {cleanup_error!r}")
        elif cleanup_errors:
            raise BaseExceptionGroup(_CLEANUP_ERROR_MESSAGE, cleanup_errors)

    logger.info(
        "Indexed {} episodes into group '{}' ({} skipped: {} no-metadata, {} no-date, {} already-indexed)",
        summary.indexed,
        corpus_metadata.group_id,
        summary.skipped_no_metadata + summary.skipped_no_date + summary.skipped_already_indexed,
        summary.skipped_no_metadata,
        summary.skipped_no_date,
        summary.skipped_already_indexed,
    )
    return summary


def main() -> None:
    """Load chunks, embed them with the configured profile, and index them into Graphiti.

    A failed episode (bad API key, bad Neo4j credentials, an LLM/Graphiti error, ...) raises out
    of the run, so the process exits non-zero and a broken run is never mistaken for a successful
    one by the caller/CI.
    """
    from dotenv import find_dotenv, load_dotenv

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    asyncio.run(_run(arguments.chunks_dir, arguments.corpus_config))


if __name__ == "__main__":
    main()
