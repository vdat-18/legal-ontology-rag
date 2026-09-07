"""Command-line interface for exporting candidates as evaluation ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the ground-truth export command-line interface."""
    parser = argparse.ArgumentParser(description="Export synthesized candidates for Legal RAG evaluation.")
    parser.add_argument("candidates_path", type=Path, help="Synthesized candidates JSONL path")
    parser.add_argument("output_path", type=Path, help="Destination evaluation ground-truth JSONL path")
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Treat candidates input as multi-turn conversations instead of single-turn samples",
    )
    return parser


def main() -> None:
    """Export synthesized candidates using the schema consumed by legal-evaluate."""
    from legal_rag.benchmark_synthesis.groundtruth_export import (
        export_ground_truth,
        export_multi_turn_ground_truth,
    )
    from legal_rag.benchmark_synthesis.writer import read_candidates, read_multi_turn_candidates
    from legal_rag.io_utils import write_jsonl_atomic

    arguments = _parser().parse_args()
    if arguments.multi_turn:
        conversations = read_multi_turn_candidates(arguments.candidates_path)
        records = export_multi_turn_ground_truth(conversations)
    else:
        candidates = read_candidates(arguments.candidates_path)
        records = export_ground_truth(candidates)
    write_jsonl_atomic(arguments.output_path, records)
    logger.info("Exported {} evaluation samples to {}", len(records), arguments.output_path)


if __name__ == "__main__":
    main()
