"""Command-line interface for querying the naive RAG baseline pipeline."""

from __future__ import annotations

import argparse

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the naive RAG query command-line interface."""
    parser = argparse.ArgumentParser(description="Answer one question using the naive RAG baseline pipeline.")
    parser.add_argument("query", type=str, help="Natural-language question in Vietnamese")
    parser.add_argument("--top-k", type=int, default=None, help="Override the configured number of retrieved chunks")
    return parser


def main() -> None:
    """Run the naive RAG pipeline for one query and log the answer and its sources."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.naive_rag.pipeline import NaiveRAGPipeline
    from legal_rag.settings import MilvusSettings, NaiveRagSettings, Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    milvus_settings = MilvusSettings.from_environment()
    naive_rag_settings = NaiveRagSettings.from_environment()

    pipeline = NaiveRAGPipeline(settings, milvus_settings, naive_rag_settings)
    result = pipeline.run(arguments.query, top_k=arguments.top_k)

    logger.info("Answer: {}", result.answer)
    for source in result.sources:
        logger.info("Source: {} ({}, score={:.4f})", source.chunk_id, source.source_name, source.score)


if __name__ == "__main__":
    main()
