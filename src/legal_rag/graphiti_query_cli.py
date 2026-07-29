"""Command-line interface for querying the Graphiti knowledge graph RAG pipeline."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from loguru import logger


def _positive_int(raw: str) -> int:
    """Parse *raw* as an integer of at least 1, for argparse options that count things."""
    value = int(raw)
    if value < 1:
        msg = f"expected a positive integer, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _parser() -> argparse.ArgumentParser:
    """Build the Graphiti query command-line interface."""
    parser = argparse.ArgumentParser(description="Answer one question using the Graphiti knowledge graph pipeline.")
    parser.add_argument("query", type=str, help="Natural-language question in Vietnamese")
    parser.add_argument("--group-id", type=str, required=True, help="Graphiti group id to search within")
    parser.add_argument(
        "--corpus-config",
        type=Path,
        default=None,
        help="Ground-truth corpus config JSON path, used to flag citations outside the indexed corpus",
    )
    parser.add_argument(
        "--max-facts",
        type=_positive_int,
        default=None,
        help="Override the configured number of context facts (positive integer)",
    )
    return parser


async def _run(query: str, group_id: str, corpus_config_path: Path | None, max_facts: int | None) -> None:
    """Run the Graphiti RAG pipeline for one query and log the answer, sources, and any caveats."""
    from legal_rag.graphiti_rag.indexer import load_corpus_metadata
    from legal_rag.graphiti_rag.pipeline import GraphitiRAGPipeline
    from legal_rag.settings import GraphitiSettings, Settings

    settings = Settings.from_environment()
    graphiti_settings = GraphitiSettings.from_environment()

    known_document_numbers: frozenset[str] | None = None
    if corpus_config_path is not None:
        corpus_metadata = load_corpus_metadata(corpus_config_path)
        known_document_numbers = frozenset(document.document_number for document in corpus_metadata.documents.values())

    pipeline = GraphitiRAGPipeline(settings, graphiti_settings, group_id, known_document_numbers=known_document_numbers)
    try:
        result = await pipeline.run(query, max_facts=max_facts)
    finally:
        await pipeline.close()

    logger.info("Answer: {}", result.answer)
    for fact in result.sources:
        citations = ", ".join(f"{source.source_name}#{source.chunk_id}" for source in fact.sources)
        logger.info("Fact [{}]: {}", citations, fact.fact)
    if result.unindexed_citations:
        logger.warning(
            "Caveat: this answer cites document(s) not fully indexed in this corpus: {} — "
            "information from those documents may be incomplete or missing",
            ", ".join(result.unindexed_citations),
        )


def main() -> None:
    """Load settings, run the Graphiti pipeline for one query, and log the answer and its sources."""
    from dotenv import find_dotenv, load_dotenv

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    asyncio.run(_run(arguments.query, arguments.group_id, arguments.corpus_config, arguments.max_facts))


if __name__ == "__main__":
    main()
