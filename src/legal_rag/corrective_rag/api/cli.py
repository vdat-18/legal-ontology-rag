"""Command-line interface for the Corrective RAG pipeline.

Supports indexing processed chunks and executing CRAG queries.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from loguru import logger

from legal_rag.corrective_rag.generate.config import RAGConfig
from legal_rag.corrective_rag.generate.pipeline import CorrectiveRAGPipeline
from legal_rag.corrective_rag.retrieval.database import LegalVectorDB
from legal_rag.corrective_rag.retrieval.embeddings import EmbeddingModel


def handle_index(args: argparse.Namespace) -> None:
    """Index chunk JSONL documents into the Milvus vector database."""
    logger.info("Initializing indexing sequence...")
    try:
        config = RAGConfig.from_env()
        embedder = EmbeddingModel(config.embedding_profile)
        db = LegalVectorDB(config, embedder)

        db.index_processed_data(processed_dir=args.data_dir)

        db.close()
        embedder.close()
        logger.info("Indexing sequence completed successfully.")
    except Exception as e:
        logger.error(f"Indexing sequence failed: {e}")
        sys.exit(1)


def handle_query(args: argparse.Namespace) -> None:
    """Query the Corrective RAG pipeline and output the response."""
    logger.info("Querying Corrective RAG.")
    try:
        config = RAGConfig.from_env()
        pipeline = CorrectiveRAGPipeline(config)
        result = pipeline.run(args.query, limit=args.limit)
        _write_query_result(result)
        pipeline.close()
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        sys.exit(1)


def _write_query_result(result: dict[str, Any]) -> None:
    """Write a compact query result to stdout."""
    lines = [
        "",
        "=" * 60,
        "QUESTION:",
        str(result["query"]),
    ]
    if result["reformulated_query"]:
        lines.extend(["", f"REFORMULATED QUERY: {result['reformulated_query']}"])

    lines.extend(
        [
            "",
            f"PIPELINE STATUS: {result['status']}",
            f"CRAG ACTION: {result['crag_action']}",
            f"KNOWLEDGE USED: {result['relevant_count']}",
            f"INTERNAL: {result['internal_knowledge_count']} | WEB: {result['external_knowledge_count']}",
            "",
            "ANSWER:",
            str(result["answer"]),
            "=" * 60,
            "",
        ]
    )
    sys.stdout.write("\n".join(lines))


def main() -> None:
    """Define parsers and orchestrate command-line options."""
    parser = argparse.ArgumentParser(
        description="Corrective RAG (CRAG) system CLI for Vietnamese road traffic regulations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")

    index_parser = subparsers.add_parser("index", help="Index processed document chunks into Milvus")
    index_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/groundtruth",
        help="Directory containing processed *.chunks.jsonl files",
    )

    query_parser = subparsers.add_parser("query", help="Run a query through the CRAG pipeline")
    query_parser.add_argument("query", type=str, help="Search query string")
    query_parser.add_argument("--limit", type=int, default=5, help="Maximum chunk documents to retrieve")

    args = parser.parse_args()

    if args.command == "index":
        handle_index(args)
    elif args.command == "query":
        handle_query(args)


if __name__ == "__main__":
    main()
