"""Command-line interface backfilling extraction progress for a document already fully extracted."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the extraction-progress backfill command-line interface."""
    parser = argparse.ArgumentParser(
        description="Record every accepted chunk of one document as already processed by fact extraction."
    )
    parser.add_argument("screening_path", type=Path, help="Chunk-screening JSONL path from legal-screen-chunks")
    parser.add_argument("progress_output_path", type=Path, help="Destination extraction-progress JSONL path")
    parser.add_argument(
        "--skipped-chunk-id",
        action="append",
        default=[],
        help="Chunk id fact extraction skipped and must retry; repeat per chunk",
    )
    return parser


def main() -> None:
    """Mark one document's accepted chunks as processed, for facts extracted before progress was recorded."""
    from legal_rag.benchmark_synthesis.fact_extraction import ExtractionProgressRecord
    from legal_rag.benchmark_synthesis.writer import read_chunk_screening, write_extraction_progress

    arguments = _parser().parse_args()
    skipped_chunk_ids = set(arguments.skipped_chunk_id)
    screening_records = read_chunk_screening(arguments.screening_path)
    records = tuple(
        ExtractionProgressRecord(chunk_id=record.chunk_id)
        for record in screening_records
        if record.has_penalty_content and record.chunk_id not in skipped_chunk_ids
    )

    write_extraction_progress(arguments.progress_output_path, records)
    logger.info(
        "Backfilled {} processed chunks ({} left for retry), wrote {}",
        len(records),
        len(skipped_chunk_ids),
        arguments.progress_output_path,
    )


if __name__ == "__main__":
    main()
