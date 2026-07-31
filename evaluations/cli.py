"""Command-line entry point for the Ragas + QUEST evaluation harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from loguru import logger

from evaluations.core.dataset import load_samples


def _parser() -> argparse.ArgumentParser:
    """Build the evaluation command-line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate Legal RAG predictions with Ragas metrics and the QUEST rubric."
    )
    parser.add_argument(
        "predictions",
        type=Path,
        help="JSONL predictions file with question, response, retrieved_contexts, and optional reference.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON path for the evaluation report.")
    parser.add_argument(
        "--embedding-profile",
        type=str,
        default=None,
        help="Embedding profile id for the judge embeddings (defaults to EMBEDDING_PROFILE).",
    )
    return parser


def main() -> None:
    """Load predictions, run the metric suite, and write the aggregate report."""
    from dotenv import find_dotenv, load_dotenv

    from evaluations.ragas.evaluator import build_evaluator
    from evaluations.ragas.metrics import build_metrics
    from evaluations.ragas.runner import run_evaluation
    from legal_rag.embedding.profiles import load_configured_embedding_profile
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    samples = load_samples(arguments.predictions)
    if not samples:
        msg = f"No evaluation samples in {arguments.predictions}"
        raise SystemExit(msg)

    settings = Settings.from_environment()
    profile_id = arguments.embedding_profile or os.environ["EMBEDDING_PROFILE"]
    embedding_model = load_configured_embedding_profile(profile_id).model_id
    evaluator = build_evaluator(settings, embedding_model)

    with_reference = all(sample.reference for sample in samples)
    metrics = build_metrics(evaluator.llm, with_reference=with_reference)
    report = run_evaluation(samples, metrics, evaluator_llm=evaluator.llm, evaluator_embeddings=evaluator.embeddings)

    logger.info("Evaluated {} sample(s); reference-based metrics {}", len(samples), "on" if with_reference else "off")
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)


if __name__ == "__main__":
    main()
