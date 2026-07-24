"""Command-line interface for deterministic parser-output chunking."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from legal_rag.chunker.pipeline import chunk_source_file
from legal_rag.chunker.writer import write_chunks
from legal_rag.embedding.profiles import load_configured_embedding_profile
from legal_rag.embedding.tokenizer import EmbeddingTokenizer
from legal_rag.settings import ChunkerSettings


def _parser() -> argparse.ArgumentParser:
    """Build the chunker command-line interface."""
    parser = argparse.ArgumentParser(description="Chunk one parser source JSONL for retrieval.")
    parser.add_argument("input_path", type=Path, help="Parser source JSONL path")
    parser.add_argument("output_path", type=Path, help="Destination chunk JSONL path")
    return parser


def main() -> None:
    """Run the complete chunking pipeline and atomically write its output."""
    from dotenv import find_dotenv, load_dotenv

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = ChunkerSettings.from_environment()
    profile = load_configured_embedding_profile(settings.embedding_profile)
    tokenizer = EmbeddingTokenizer.from_profile(profile)
    chunks = chunk_source_file(arguments.input_path, tokenizer, settings)
    write_chunks(arguments.output_path, chunks)

    token_counts = [chunk.text_token_count for chunk in chunks]
    logger.info(
        "Wrote {} chunks across {} scopes from {} to {}; token min/mean/max: {}/{:.1f}/{}",
        len(chunks),
        len({chunk.scope_id for chunk in chunks}),
        arguments.input_path.name,
        arguments.output_path,
        min(token_counts, default=0),
        sum(token_counts) / len(token_counts) if token_counts else 0,
        max(token_counts, default=0),
    )


if __name__ == "__main__":
    main()
