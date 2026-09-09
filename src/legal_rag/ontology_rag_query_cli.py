"""Command-line interface for querying the Ontology-enhanced RAG pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the ontology RAG query CLI."""
    parser = argparse.ArgumentParser(description="Answer one question using the ontology-enhanced RAG pipeline.")
    parser.add_argument("query", type=str, help="Natural-language question in Vietnamese")
    parser.add_argument("--top-k", type=int, default=None, help="Override the configured number of retrieved chunks")
    parser.add_argument(
        "--query-date", type=str, default=None, help="Query date in YYYY-MM-DD format (defaults to today)"
    )
    parser.add_argument(
        "--ontology",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to ontology TTL files",
    )
    return parser


def main() -> None:
    """Run the ontology RAG pipeline and log answer and penalty rules."""
    from datetime import datetime

    from dotenv import find_dotenv, load_dotenv

    from legal_rag.ontology_rag.pipeline import OntologyRAGPipeline
    from legal_rag.settings import MilvusSettings, NaiveRagSettings, Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    settings = Settings.from_environment()
    milvus_settings = MilvusSettings.from_environment()
    naive_rag_settings = NaiveRagSettings.from_environment()
    ontology_paths = [Path(p) for p in arguments.ontology]

    query_date = None
    if arguments.query_date:
        try:
            query_date = datetime.fromisoformat(arguments.query_date).date()
        except ValueError:
            logger.error("Invalid query_date format. Use YYYY-MM-DD.")
            return

    pipeline = OntologyRAGPipeline(settings, milvus_settings, naive_rag_settings, ontology_paths)
    try:
        result = pipeline.run(arguments.query, query_date=query_date, top_k=arguments.top_k)

        logger.info("Answer: {}", result.answer)
        logger.info("Valid rules: {}", len(result.rules))

        for rule in result.rules:
            parts = []
            if rule.monetary_penalty_min and rule.monetary_penalty_max:
                parts.append(f"Phạt {rule.monetary_penalty_min}-{rule.monetary_penalty_max} {rule.currency or 'VND'}")
            if rule.point_value:
                parts.append(f"trừ {rule.point_value} điểm")
            consequence = ", ".join(parts) if parts else "Không có thông tin phạt"

            logger.info("Rule: {} - {}", rule.conduct_label_vi, consequence)
            logger.info("  Source: {} ({})", rule.provision_path, rule.document_number)

        for source in result.sources[:5]:
            logger.info("Source: {} ({}, score={:.4f})", source.chunk_id, source.source_name, source.score)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
