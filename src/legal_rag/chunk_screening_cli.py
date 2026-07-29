"""Command-line interface for the chunk-screening gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the chunk-screening command-line interface."""
    parser = argparse.ArgumentParser(description="Screen one document's chunks for penalty content.")
    parser.add_argument("chunks_path", type=Path, help="Chunk JSONL path for one document")
    parser.add_argument("screening_output_path", type=Path, help="Destination chunk-screening JSONL path")
    return parser


def main() -> None:
    """Screen chunks not yet in the output file, keeping prior verdicts on resume."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.chunk_screening import BamlChunkScreeningClient, screen_chunks
    from legal_rag.benchmark_synthesis.writer import read_chunk_screening, write_chunk_screening
    from legal_rag.chunker.writer import read_chunks
    from legal_rag.settings import EvaluationSettings, Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    evaluation_settings = EvaluationSettings.from_environment()

    chunks = read_chunks(arguments.chunks_path)
    existing_records = (
        read_chunk_screening(arguments.screening_output_path) if arguments.screening_output_path.exists() else ()
    )
    records_by_chunk_id = {record.chunk_id: record for record in existing_records}
    chunks_to_screen = tuple(chunk for chunk in chunks if chunk.chunk_id not in records_by_chunk_id)

    if chunks_to_screen:
        screening_client = BamlChunkScreeningClient(settings, evaluation_settings)
        new_records = screen_chunks(chunks_to_screen, screening_client)
        records_by_chunk_id.update({record.chunk_id: record for record in new_records})

    records = tuple(records_by_chunk_id[chunk.chunk_id] for chunk in chunks if chunk.chunk_id in records_by_chunk_id)
    write_chunk_screening(arguments.screening_output_path, records)
    accepted_count = sum(1 for record in records if record.has_penalty_content)
    logger.info(
        "Chunk screening gate: {} accepted, {} excluded, {} still unscreened, wrote {}",
        accepted_count,
        len(records) - accepted_count,
        len(chunks) - len(records),
        arguments.screening_output_path,
    )


if __name__ == "__main__":
    main()
