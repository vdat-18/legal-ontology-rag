"""Command-line entry point for the BAML + QUEST evaluation harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from loguru import logger

from evaluations.core.dataset import load_samples

_DEFAULT_BATCH_SIZE = 5
_DEFAULT_CONCURRENCY = 8


def _positive_int(raw: str) -> int:
    """Parse *raw* as an integer of at least 1, for the concurrency option."""
    value = int(raw)
    if value < 1:
        msg = f"expected a positive integer, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _parser() -> argparse.ArgumentParser:
    """Build the evaluation command-line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate Legal RAG predictions with a BAML LLM judge (QUEST rubric + grounding metrics)."
    )
    parser.add_argument(
        "predictions",
        type=Path,
        help="JSONL predictions file with question, response, retrieved_contexts, and optional reference.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON path for the evaluation report.")
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=_DEFAULT_BATCH_SIZE,
        help="Number of samples judged per LLM call (positive integer).",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=_DEFAULT_CONCURRENCY,
        help="Maximum number of concurrent batch calls (positive integer).",
    )
    return parser


def main() -> None:
    """Load predictions, judge them concurrently through BAML, and write the aggregate report."""
    from dotenv import find_dotenv, load_dotenv

    from evaluations.baml.judge import BamlJudgeClient
    from evaluations.baml.runner import evaluate_samples
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    samples = load_samples(arguments.predictions)
    if not samples:
        msg = f"No evaluation samples in {arguments.predictions}"
        raise SystemExit(msg)

    judge = BamlJudgeClient(Settings.from_environment())
    report = asyncio.run(
        evaluate_samples(samples, judge, batch_size=arguments.batch_size, concurrency=arguments.concurrency)
    )

    logger.info(
        "Evaluated {} sample(s), batch_size={}, concurrency={}",
        len(samples),
        arguments.batch_size,
        arguments.concurrency,
    )
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)


if __name__ == "__main__":
    main()
