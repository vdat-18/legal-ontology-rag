"""Command-line interface for Legal RAG evaluation metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from legal_rag.evaluation.metrics import DEFAULT_K_VALUES, EvaluationSample, evaluate_samples, load_jsonl_samples
from legal_rag.evaluation.runner import (
    CorrectiveEvaluationBackend,
    EvaluationBackend,
    GraphitiEvaluationBackend,
    NaiveEvaluationBackend,
    load_evaluation_records,
    run_evaluation,
)


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

    run_parser = subparsers.add_parser("run", help="Run one RAG pipeline and score it with the shared metrics.")
    run_parser.add_argument(
        "--pipeline",
        choices=("naive", "corrective", "graphiti"),
        default="corrective",
        help="RAG implementation to evaluate. All choices emit the same prediction schema.",
    )
    run_parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/benchmark_synthesis/groundtruth.jsonl"),
        help="Reviewed ground-truth JSONL file.",
    )
    run_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=5,
        help="Maximum ranked retrieval units supplied to each pipeline per question.",
    )
    run_parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=1,
        help="Maximum questions evaluated concurrently; each carries one LLM answer-generation call.",
    )
    run_parser.add_argument(
        "--k-values",
        type=_parse_k_values,
        default=DEFAULT_K_VALUES,
        help="Comma-separated k values for @k retrieval metrics, for example: 1,3,5,10.",
    )
    run_parser.add_argument(
        "--predictions-output",
        type=Path,
        default=None,
        help="Per-sample JSONL path; defaults to data/evaluation/<pipeline>_predictions.jsonl.",
    )
    run_parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Aggregate JSON path; defaults to data/evaluation/<pipeline>_metrics.json.",
    )
    run_parser.add_argument(
        "--generation",
        action="store_true",
        help="Also score generated answers with the BAML LLM judge and merge the aggregate into the metrics.",
    )
    run_parser.add_argument(
        "--judge-batch-size",
        type=_positive_int,
        default=5,
        help="Number of samples judged per LLM call when --generation is set.",
    )
    run_parser.add_argument(
        "--judge-concurrency",
        type=_positive_int,
        default=8,
        help="Maximum concurrent judge batch calls when --generation is set.",
    )
    run_parser.add_argument(
        "--generation-output",
        type=Path,
        default=None,
        help="Full per-sample judge report path; defaults to data/evaluation/<pipeline>_generation.json.",
    )
    run_parser.add_argument(
        "--naive-search-mode",
        choices=("keyword", "semantic", "hybrid"),
        default=None,
        help="Optional search-mode override for the naive pipeline.",
    )
    run_parser.add_argument(
        "--group-id",
        default="traffic_penalty_groundtruth_v1",
        help="Graphiti group to evaluate when --pipeline graphiti.",
    )
    run_parser.add_argument(
        "--corpus-config",
        type=Path,
        default=None,
        help="Optional Graphiti corpus config used to detect citations outside the indexed corpus.",
    )
    run_parser.add_argument(
        "--no-graph-expansion",
        action="store_true",
        help="Disable Graphiti graph-proximity expansion.",
    )
    run_parser.add_argument(
        "--cross-encoder",
        action="store_true",
        help="Use Graphiti cross-encoder ranking instead of rank fusion.",
    )

    retrieve_parser = subparsers.add_parser("retrieve", help="Run retrieval-only evaluation without LLM calls.")
    retrieve_parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/benchmark_synthesis/groundtruth.jsonl"),
        help="Reviewed evaluation JSONL file.",
    )
    retrieve_parser.add_argument(
        "--limit", type=_positive_int, default=10, help="Maximum chunks to retrieve per question."
    )
    retrieve_parser.add_argument(
        "--retriever",
        choices=("vector", "graph"),
        default="vector",
        help="Which retriever to score: the Milvus vector baseline or the Graphiti knowledge graph.",
    )
    retrieve_parser.add_argument(
        "--group-id",
        default="traffic_penalty_groundtruth_v1",
        help="Graphiti group to search, for --retriever graph.",
    )
    retrieve_parser.add_argument(
        "--no-graph-expansion",
        action="store_true",
        help="Disable graph-proximity expansion, to measure what the graph structure contributes.",
    )
    retrieve_parser.add_argument(
        "--cross-encoder",
        action="store_true",
        help="Rank with Graphiti's cross-encoder recipe instead of rank fusion.",
    )
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
    """Run one selected RAG pipeline and score it with retrieval and optional generation metrics."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    ground_truth_path = args.ground_truth
    if not ground_truth_path.exists():
        msg = f"Evaluation file not found: {ground_truth_path}"
        raise SystemExit(msg)

    records = load_evaluation_records(ground_truth_path)
    if not records:
        msg = f"Evaluation file has no samples: {ground_truth_path}"
        raise SystemExit(msg)

    backend = _build_evaluation_backend(args)
    predictions_output = args.predictions_output or Path(f"data/evaluation/{args.pipeline}_predictions.jsonl")
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    with predictions_output.open("w", encoding="utf-8", newline="\n") as predictions_file:

        def _append_prediction(row: dict[str, Any]) -> None:
            """Persist one finished prediction immediately so progress survives interruption."""
            predictions_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            predictions_file.flush()

        rows = asyncio.run(
            run_evaluation(
                records,
                backend,
                limit=args.limit,
                concurrency=args.concurrency,
                on_row=_append_prediction,
            )
        )
    metrics_output = args.metrics_output or Path(f"data/evaluation/{args.pipeline}_metrics.json")
    metrics = evaluate_samples([EvaluationSample.from_mapping(row) for row in rows], k_values=args.k_values)
    metrics_payload = {
        "pipeline": args.pipeline,
        "retrieval_limit": args.limit,
        **metrics.to_dict(),
    }
    if args.generation:
        report = _score_generation(rows, args)
        generation_output = args.generation_output or Path(f"data/evaluation/{args.pipeline}_generation.json")
        _write_json(generation_output, report.to_dict())
        metrics_payload["generation"] = report.aggregate
    _write_json(metrics_output, metrics_payload)
    _write_stdout(metrics_payload)


def _score_generation(rows: list[dict[str, Any]], args: argparse.Namespace) -> Any:
    """Judge the generated answers of *rows* with the BAML LLM judge and return the report."""
    from legal_rag.evaluation.generation.core.dataset import EvalSample
    from legal_rag.evaluation.generation.judge import BamlJudgeClient
    from legal_rag.evaluation.generation.runner import evaluate_samples as evaluate_generation_samples
    from legal_rag.settings import Settings

    samples = [EvalSample.from_mapping(row) for row in rows]
    judge = BamlJudgeClient(Settings.from_environment())
    return asyncio.run(
        evaluate_generation_samples(
            samples, judge, batch_size=args.judge_batch_size, concurrency=args.judge_concurrency
        )
    )


def _build_evaluation_backend(args: argparse.Namespace) -> EvaluationBackend:
    """Build the selected pipeline behind the shared evaluation boundary."""
    if args.pipeline == "naive":
        from legal_rag.naive_rag.pipeline import NaiveRAGPipeline
        from legal_rag.settings import MilvusSettings, NaiveRagSettings, Settings

        pipeline = NaiveRAGPipeline(
            Settings.from_environment(),
            MilvusSettings.from_environment(),
            NaiveRagSettings.from_environment(),
        )
        return NaiveEvaluationBackend(pipeline, search_mode=args.naive_search_mode)

    if args.pipeline == "corrective":
        from legal_rag.corrective_rag.generate.config import RAGConfig
        from legal_rag.corrective_rag.generate.pipeline import CorrectiveRAGPipeline

        return CorrectiveEvaluationBackend(CorrectiveRAGPipeline(RAGConfig.from_env()))

    from legal_rag.graphiti_rag.indexer import load_corpus_metadata
    from legal_rag.graphiti_rag.pipeline import GraphitiRAGPipeline
    from legal_rag.settings import GraphitiSettings, Neo4jSettings, Settings

    graphiti_settings = GraphitiSettings.from_environment()
    graphiti_settings = replace(
        graphiti_settings,
        expand_around_top_fact=not args.no_graph_expansion,
        rerank_with_cross_encoder=args.cross_encoder,
    )
    known_document_numbers: frozenset[str] | None = None
    if args.corpus_config is not None:
        corpus_metadata = load_corpus_metadata(args.corpus_config)
        known_document_numbers = frozenset(document.document_number for document in corpus_metadata.documents.values())
    pipeline = GraphitiRAGPipeline(
        Settings.from_environment(),
        Neo4jSettings.from_environment(),
        graphiti_settings,
        args.group_id,
        known_document_numbers=known_document_numbers,
    )
    return GraphitiEvaluationBackend(pipeline)


def _handle_retrieve(args: argparse.Namespace) -> None:
    """Run vector retrieval for each ground-truth sample and score it."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    ground_truth_path = args.ground_truth
    if not ground_truth_path.exists():
        msg = f"Evaluation file not found: {ground_truth_path}"
        raise SystemExit(msg)

    samples = load_jsonl_samples(ground_truth_path)
    if not samples:
        msg = f"Evaluation file has no samples: {ground_truth_path}"
        raise SystemExit(msg)

    retrieve = _retrieve_with_graph if args.retriever == "graph" else _retrieve_with_vector
    rows = retrieve(samples, args)

    _write_jsonl(args.predictions_output, rows)
    metrics = evaluate_samples([EvaluationSample.from_mapping(row) for row in rows], k_values=args.k_values)
    _write_json(args.metrics_output, metrics.to_dict())
    _write_stdout(metrics.to_dict())


def _retrieve_with_vector(samples: list[EvaluationSample], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Score the Milvus vector baseline, retrieving on each question as written."""
    from legal_rag.corrective_rag.generate.config import RAGConfig
    from legal_rag.corrective_rag.retrieval.database import LegalVectorDB
    from legal_rag.corrective_rag.retrieval.embeddings import EmbeddingModel

    config = RAGConfig.from_env()
    embedder = EmbeddingModel(config.embedding_profile)
    db = LegalVectorDB(config, embedder)
    rows: list[dict[str, Any]] = []
    try:
        for index, sample in enumerate(samples, 1):
            logger.info("Retrieving sample {}/{}", index, len(samples))
            rows.append(_retrieval_prediction_row(sample, db.search(sample.question, limit=args.limit)))
    finally:
        db.close()
        embedder.close()
    return rows


def _retrieve_with_graph(samples: list[EvaluationSample], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Score the Graphiti knowledge graph on the same questions and the same chunk ids.

    The graph is searched on each question as written, matching the vector baseline: query
    rewriting is a separate pipeline stage that costs an LLM call, and this command is defined
    as retrieval-only. A graph fact carries the provenance of every episode supporting it, and
    an episode is named by its chunk id, so one fact can contribute several retrieved chunks.
    """
    import asyncio

    from legal_rag.embedding.profiles import load_configured_embedding_profile
    from legal_rag.graphiti_rag.client import build_graphiti
    from legal_rag.graphiti_rag.retriever import search_facts
    from legal_rag.settings import GraphitiSettings, Neo4jSettings, Settings

    settings = Settings.from_environment()
    graphiti_settings = GraphitiSettings.from_environment()
    profile = load_configured_embedding_profile(graphiti_settings.embedding_profile)
    graphiti_client = build_graphiti(settings, Neo4jSettings.from_environment(), graphiti_settings, profile)

    async def retrieve_all() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            for index, sample in enumerate(samples, 1):
                logger.info("Retrieving sample {}/{}", index, len(samples))
                facts = await search_facts(
                    graphiti_client,
                    sample.question,
                    args.group_id,
                    args.limit,
                    expand_around_top_fact=not args.no_graph_expansion,
                    rerank_with_cross_encoder=args.cross_encoder,
                )
                retrieved: list[dict[str, Any]] = []
                seen: set[str] = set()
                for fact in facts:
                    for source in fact.sources:
                        if source.chunk_id not in seen:
                            seen.add(source.chunk_id)
                            retrieved.append(
                                {"chunk_id": source.chunk_id, "source_name": source.source_name, "text": fact.fact}
                            )
                rows.append(_retrieval_prediction_row(sample, retrieved[: args.limit]))
        finally:
            await graphiti_client.close()
        return rows

    return asyncio.run(retrieve_all())


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


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    parsed = int(value)
    if parsed < 1:
        msg = f"Expected a positive integer, got {parsed}"
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
