"""Command-line interface for rebinding persisted fact evidence without LLM calls."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the fact-evidence rebinding command-line interface."""
    parser = argparse.ArgumentParser(description="Split non-contiguous fact citations into exact source spans.")
    parser.add_argument("facts_path", type=Path, help="Current legal-facts JSONL path")
    parser.add_argument("chunks_path", type=Path, help="Source chunks JSONL path for the same document")
    return parser


def main() -> None:
    """Rewrite one fact file with exact evidence spans and recomputed content hashes."""
    from legal_rag.benchmark_synthesis.evidence_binding import rebuild_fact_with_bound_evidence
    from legal_rag.benchmark_synthesis.writer import read_legal_facts, write_legal_facts
    from legal_rag.chunker.writer import read_chunks

    arguments = _parser().parse_args()
    facts = read_legal_facts(arguments.facts_path)
    chunk_text_by_id = {chunk.chunk_id: chunk.text for chunk in read_chunks(arguments.chunks_path)}
    rebound = tuple(rebuild_fact_with_bound_evidence(fact, chunk_text_by_id) for fact in facts)
    changed_count = sum(before != after for before, after in zip(facts, rebound, strict=True))
    write_legal_facts(arguments.facts_path, rebound)
    logger.info("Rebound evidence for {} of {} facts in {}", changed_count, len(facts), arguments.facts_path)


if __name__ == "__main__":
    main()
