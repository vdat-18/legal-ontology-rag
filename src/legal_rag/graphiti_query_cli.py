"""Command-line interface for querying the Graphiti knowledge graph RAG pipeline."""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the Graphiti query command-line interface."""
    parser = argparse.ArgumentParser(description="Answer one question using the Graphiti knowledge graph pipeline.")
    parser.add_argument("query", type=str, help="Natural-language question in Vietnamese")
    parser.add_argument("--group-id", type=str, required=True, help="Graphiti group id to search within")
    parser.add_argument("--max-facts", type=int, default=None, help="Override the configured number of context facts")
    return parser


async def _run(query: str, group_id: str, max_facts: int | None) -> None:
    """Run the Graphiti RAG pipeline for one query and log the answer and its sources."""
    from legal_rag.graphiti_rag.pipeline import GraphitiRAGPipeline
    from legal_rag.settings import GraphitiSettings, Settings

    settings = Settings.from_environment()
    graphiti_settings = GraphitiSettings.from_environment()

    pipeline = GraphitiRAGPipeline(settings, graphiti_settings, group_id)
    try:
        result = await pipeline.run(query, max_facts=max_facts)
    finally:
        await pipeline.close()

    logger.info("Answer: {}", result.answer)
    for source in result.sources:
        logger.info("Fact [{}]: {}", source.source_name, source.text)


def main() -> None:
    """Load settings, run the Graphiti pipeline for one query, and log the answer and its sources."""
    from dotenv import find_dotenv, load_dotenv

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    asyncio.run(_run(arguments.query, arguments.group_id, arguments.max_facts))


if __name__ == "__main__":
    main()
