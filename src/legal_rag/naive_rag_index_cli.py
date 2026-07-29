"""Command-line interface for indexing chunked JSONL data into the naive RAG baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the naive RAG indexing command-line interface."""
    parser = argparse.ArgumentParser(
        description="Embed and index chunked JSONL data into Milvus for the naive RAG baseline."
    )
    parser.add_argument("chunks_dir", type=Path, help="Directory containing one or more *.chunks.jsonl files")
    return parser


def main() -> None:
    """Load chunks, embed them with the configured profile, and index into Milvus."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.datastores.milvus import build_milvus_client
    from legal_rag.embedding.encoder import EmbeddingEncoder
    from legal_rag.embedding.profiles import load_configured_embedding_profile
    from legal_rag.naive_rag.indexer import index_chunks, load_chunks
    from legal_rag.settings import MilvusSettings, NaiveRagSettings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    milvus_settings = MilvusSettings.from_environment()
    naive_rag_settings = NaiveRagSettings.from_environment()

    chunks = load_chunks(arguments.chunks_dir)
    profile = load_configured_embedding_profile(naive_rag_settings.embedding_profile)
    encoder = EmbeddingEncoder.from_profile(profile)
    milvus_client = build_milvus_client(milvus_settings)

    total_indexed = index_chunks(chunks, encoder, milvus_client, naive_rag_settings)
    logger.info(
        "Naive RAG index ready with {} rows in collection '{}' at {}",
        total_indexed,
        naive_rag_settings.milvus_collection_name,
        milvus_settings.uri,
    )


if __name__ == "__main__":
    main()
