"""Command-line interface for Legal RAG evaluation metrics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from legal_rag.evaluation.metrics import DEFAULT_K_VALUES, EvaluationSample, evaluate_samples, load_jsonl_samples


def main() -> None:
    """Run the evaluation command-line interface."""
    args = _parser().parse_args()
    if args.command == "score":
        _handle_score(args)
    elif args.command == "run":
        _handle_run(args)
    elif args.command == "retrieve":
        _handle_retrieve(args)


def _parser() -> argparse.ArgumentParser:
    """Build the evaluation parser."""
    parser = argparse.ArgumentParser(description="Evaluate Legal RAG predictions and retrieval outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Score an existing JSONL file with predictions.")
    score_parser.add_argument(
        "input_path",
        type=Path,
        help="JSONL file containing expected labels plus actual answer/retrieval fields.",
    )
    score_parser.add_argument(
        "--k-values",
        type=_parse_k_values,
        default=DEFAULT_K_VALUES,
        help="Comma-separated k values for @k retrieval metrics, for example: 1,3,5,10.",
    )
    score_parser.add_argument("--output", type=Path, help="Optional JSON path for aggregate metrics.")

    run_parser = subparsers.add_parser("run", help="Run CRAG on a ground_truth.jsonl file and score predictions.")
    run_parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/evaluation/ground_truth.jsonl"),
        help="Reviewed evaluation JSONL file.",
    )
    run_parser.add_argument("--limit", type=int, default=5, help="Maximum chunks to retrieve per question.")
    run_parser.add_argument(
        "--k-values",
        type=_parse_k_values,
        default=DEFAULT_K_VALUES,
        help="Comma-separated k values for @k retrieval metrics, for example: 1,3,5,10.",
    )
    run_parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/evaluation/predictions.jsonl"),
        help="Path where per-sample predictions are written.",
    )
    run_parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/evaluation/metrics.json"),
        help="Path where aggregate metrics are written.",
    )

    retrieve_parser = subparsers.add_parser("retrieve", help="Run retrieval-only evaluation without LLM calls.")
    retrieve_parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/evaluation/ground_truth.jsonl"),
        help="Reviewed evaluation JSONL file.",
    )
    retrieve_parser.add_argument("--limit", type=int, default=10, help="Maximum chunks to retrieve per question.")
    retrieve_parser.add_argument(
        "--k-values",
        type=_parse_k_values,
        default=DEFAULT_K_VALUES,
        help="Comma-separated k values for @k retrieval metrics, for example: 1,3,5,10.",
    )
    retrieve_parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/evaluation/retrieval_predictions.jsonl"),
        help="Path where per-sample retrieval predictions are written.",
    )
    retrieve_parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/evaluation/retrieval_metrics.json"),
        help="Path where aggregate retrieval metrics are written.",
    )
    return parser


def _handle_score(args: argparse.Namespace) -> None:
    """Score an existing prediction JSONL file."""
    if not args.input_path.exists():
        msg = f"Evaluation file not found: {args.input_path}"
        raise SystemExit(msg)

    metrics = evaluate_samples(load_jsonl_samples(args.input_path), k_values=args.k_values)
    if args.output:
        _write_json(args.output, metrics.to_dict())
    _write_stdout(metrics.to_dict())


def _handle_run(args: argparse.Namespace) -> None:
    """Run the CRAG pipeline for each ground-truth sample and score it."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    ground_truth_path = args.ground_truth
    if not ground_truth_path.exists():
        msg = f"Evaluation file not found: {ground_truth_path}"
        raise SystemExit(msg)

    from legal_rag.corrective_rag.generate.config import RAGConfig
    from legal_rag.corrective_rag.generate.pipeline import CorrectiveRAGPipeline

    samples = load_jsonl_samples(ground_truth_path)
    if not samples:
        msg = f"Evaluation file has no samples: {ground_truth_path}"
        raise SystemExit(msg)

    config = RAGConfig.from_env()
    pipeline = CorrectiveRAGPipeline(config)
    rows: list[dict[str, Any]] = []
    try:
        for index, sample in enumerate(samples, 1):
            logger.info("Evaluating sample {}/{}", index, len(samples))
            result = pipeline.run(sample.question, limit=args.limit)
            rows.append(_prediction_row(sample, result))
    finally:
        pipeline.close()

    _write_jsonl(args.predictions_output, rows)
    metrics = evaluate_samples([EvaluationSample.from_mapping(row) for row in rows], k_values=args.k_values)
    _write_json(args.metrics_output, metrics.to_dict())
    _write_stdout(metrics.to_dict())


def _handle_retrieve(args: argparse.Namespace) -> None:
    """Run vector retrieval for each ground-truth sample and score it."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    ground_truth_path = args.ground_truth
    if not ground_truth_path.exists():
        msg = f"Evaluation file not found: {ground_truth_path}"
        raise SystemExit(msg)

    from legal_rag.corrective_rag.generate.config import RAGConfig
    from legal_rag.corrective_rag.retrieval.database import LegalVectorDB
    from legal_rag.corrective_rag.retrieval.embeddings import EmbeddingModel

    samples = load_jsonl_samples(ground_truth_path)
    if not samples:
        msg = f"Evaluation file has no samples: {ground_truth_path}"
        raise SystemExit(msg)

    config = RAGConfig.from_env()
    embedder = EmbeddingModel(config.embedding_profile)
    db = LegalVectorDB(config, embedder)
    rows: list[dict[str, Any]] = []
    try:
        for index, sample in enumerate(samples, 1):
            logger.info("Retrieving sample {}/{}", index, len(samples))
            retrieved_chunks = db.search(sample.question, limit=args.limit)
            rows.append(_retrieval_prediction_row(sample, retrieved_chunks))
    finally:
        db.close()
        embedder.close()

    _write_jsonl(args.predictions_output, rows)
    metrics = evaluate_samples([EvaluationSample.from_mapping(row) for row in rows], k_values=args.k_values)
    _write_json(args.metrics_output, metrics.to_dict())
    _write_stdout(metrics.to_dict())


def _prediction_row(sample: EvaluationSample, result: dict[str, Any]) -> dict[str, Any]:
    """Merge one ground-truth sample with one CRAG result."""
    row = asdict(sample)
    graded_chunks = result.get("graded_chunks", [])
    row.update(
        {
            "actual_answer": result.get("answer"),
            "retrieved_chunk_ids": _extract_result_field(graded_chunks, "chunk_id"),
            "retrieved_source_names": _extract_result_field(graded_chunks, "source_name"),
            "crag_action": result.get("crag_action"),
            "status": result.get("status"),
            "reformulated_query": result.get("reformulated_query"),
            "graded_chunks": graded_chunks,
        }
    )
    return row


def _retrieval_prediction_row(sample: EvaluationSample, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge one ground-truth sample with vector retrieval output."""
    row = asdict(sample)
    row.update(
        {
            "retrieved_chunk_ids": _extract_result_field(retrieved_chunks, "chunk_id"),
            "retrieved_source_names": _extract_result_field(retrieved_chunks, "source_name"),
            "retrieved_chunks": retrieved_chunks,
        }
    )
    return row


def _extract_result_field(items: Any, key: str) -> list[str]:
    """Extract a string field from a list of result dictionaries."""
    if not isinstance(items, list):
        return []
    return [str(item[key]) for item in items if isinstance(item, dict) and item.get(key)]


def _parse_k_values(value: str) -> tuple[int, ...]:
    """Parse comma-separated positive k values."""
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        msg = f"Invalid --k-values: {value}"
        raise argparse.ArgumentTypeError(msg) from error

    if not parsed or any(item <= 0 for item in parsed):
        msg = "--k-values must contain positive integers"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def _write_stdout(data: dict[str, Any]) -> None:
    """Write JSON output to stdout."""
    sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
