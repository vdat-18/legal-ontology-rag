"""Command-line interface for Legal RAG evaluation metrics."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from legal_rag.evaluation.metrics import (
    DEFAULT_K_VALUES,
    EvaluationSample,
    evaluate_samples,
    evaluate_slices,
    load_jsonl_samples,
)
from legal_rag.evaluation.runner import (
    CorrectiveEvaluationBackend,
    EvaluationBackend,
    GraphitiEvaluationBackend,
    NaiveEvaluationBackend,
    OntologyEvaluationBackend,
    load_evaluation_records,
    run_evaluation,
)
from legal_rag.io_utils import write_json_atomic, write_jsonl_atomic


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
        choices=("naive", "corrective", "graphiti", "ontology"),
        default="corrective",
        help="RAG implementation to evaluate. All choices emit the same prediction schema.",
    )
    run_parser.add_argument(
        "--ontology-paths",
        type=Path,
        nargs="+",
        default=None,
        help="Ontology TTL files for --pipeline ontology; defaults to the ONTOLOGY_PATHS env (comma-separated).",
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
        help="Per-sample JSONL path; defaults to <run-dir>/predictions.jsonl.",
    )
    run_parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Aggregate JSON path; defaults to <run-dir>/metrics.json.",
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
        help="Full per-sample judge report path; defaults to <run-dir>/generation.json.",
    )
    run_parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Experiment directory; defaults to data/evaluation/runs/<deterministic-run-id>.",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible interrupted run from its progress JSONL file.",
    )
    run_parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional model override for generation judging; recorded separately from the answer model.",
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
    run_parser.add_argument(
        "--rewrite-query",
        action="store_true",
        help="Rewrite queries before retrieval using LLM-based terminology normalization.",
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


def _ontology_paths_from_environment() -> list[Path]:
    """Read comma-separated ontology TTL paths from the ONTOLOGY_PATHS environment variable."""
    raw = os.environ.get("ONTOLOGY_PATHS", "")
    return [Path(part.strip()) for part in raw.split(",") if part.strip()]


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

    identity = _run_identity(args, ground_truth_path)
    run_id = _identity_hash(identity)
    run_dir = args.run_dir or Path("data/evaluation/runs") / run_id
    manifest_path = run_dir / "manifest.json"
    progress_path = run_dir / "predictions.progress.jsonl"
    predictions_output = args.predictions_output or run_dir / "predictions.jsonl"
    metrics_output = args.metrics_output or run_dir / "metrics.json"
    generation_output = args.generation_output or run_dir / "generation.json"
    already_complete = _prepare_run(manifest_path, progress_path, identity, run_id, resume=args.resume)
    if already_complete:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_metrics = manifest.get("outputs", {}).get("metrics")
        completed_metrics = Path(stored_metrics) if isinstance(stored_metrics, str) else metrics_output
        if not completed_metrics.exists():
            msg = f"Completed run is missing its metrics output: {completed_metrics}"
            raise SystemExit(msg)
        _write_stdout(json.loads(completed_metrics.read_text(encoding="utf-8")))
        return
    try:
        completed = _read_progress(progress_path) if args.resume else {}
        remaining_records = [record for record in records if _record_id(record.raw) not in completed]
        new_rows: list[dict[str, Any]] = []
        if remaining_records:
            backend = _build_evaluation_backend(args)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8", newline="\n") as progress_file:

                def _append_prediction(row: dict[str, Any]) -> None:
                    """Persist each completed prediction immediately for safe resume."""
                    row["evaluation_record_id"] = _record_id(row)
                    progress_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    progress_file.flush()

                new_rows = asyncio.run(
                    run_evaluation(
                        remaining_records,
                        backend,
                        limit=args.limit,
                        concurrency=args.concurrency,
                        on_row=_append_prediction,
                    )
                )

        rows_by_id = {**completed, **{_record_id(row): row for row in new_rows}}
        rows = [rows_by_id[_record_id(record.raw)] for record in records]
        _write_jsonl(predictions_output, rows)
        metrics = evaluate_samples([EvaluationSample.from_mapping(row) for row in rows], k_values=args.k_values)
        metrics_payload = {
            "schema_version": "2.0.0",
            "run_id": run_id,
            "pipeline": args.pipeline,
            "retrieval_limit": args.limit,
            **metrics.to_dict(),
            "slices": evaluate_slices(rows, k_values=args.k_values),
        }
        if args.generation:
            report = _score_generation(rows, args)
            _write_json(generation_output, report.to_dict())
            metrics_payload["generation"] = report.aggregate
        _write_json(metrics_output, metrics_payload)
        _update_manifest(
            manifest_path,
            status="complete",
            outputs={
                "predictions": predictions_output.as_posix(),
                "metrics": metrics_output.as_posix(),
                "generation": generation_output.as_posix() if args.generation else None,
            },
        )
    except BaseException as error:
        _update_manifest(manifest_path, status="failed", error_type=type(error).__name__)
        raise
    _write_stdout(metrics_payload)


def _score_generation(rows: list[dict[str, Any]], args: argparse.Namespace) -> Any:
    """Judge the generated answers of *rows* with the BAML LLM judge and return the report."""
    from legal_rag.evaluation.generation.core.dataset import EvalSample
    from legal_rag.evaluation.generation.judge import BamlJudgeClient
    from legal_rag.evaluation.generation.runner import evaluate_samples as evaluate_generation_samples
    from legal_rag.settings import Settings

    samples = [EvalSample.from_mapping(row) for row in rows]
    settings = Settings.from_environment()
    if args.judge_model:
        settings = replace(settings, llm_model=args.judge_model)
    judge = BamlJudgeClient(settings)
    return asyncio.run(
        evaluate_generation_samples(
            samples, judge, batch_size=args.judge_batch_size, concurrency=args.judge_concurrency
        )
    )


_RUN_ENVIRONMENT_FIELDS = (
    "LLM_MODEL",
    "EMBEDDING_PROFILE",
    "MILVUS_DATABASE",
    "NAIVE_RAG_MILVUS_COLLECTION_NAME",
    "NAIVE_RAG_SEARCH_MODE",
    "NAIVE_RAG_HYBRID_RRF_K",
    "NAIVE_RAG_HYBRID_POOL_MULTIPLIER",
    "CRAG_MILVUS_COLLECTION_NAME",
    "CRAG_WEB_SEARCH_ENABLED",
    "GRAPHITI_SEARCH_NUM_RESULTS",
    "GRAPHITI_MAX_CONTEXT_FACTS",
    "GRAPHITI_RERANKER_MODEL_ID",
    "GRAPHITI_RERANKER_MODEL_REVISION",
)


def _run_identity(args: argparse.Namespace, ground_truth_path: Path) -> dict[str, Any]:
    """Build the secret-free identity that distinguishes one experiment configuration."""
    ontology_paths = list(args.ontology_paths or _ontology_paths_from_environment())
    return {
        "pipeline": args.pipeline,
        "ground_truth_path": ground_truth_path.resolve().as_posix(),
        "ground_truth_sha256": _sha256_file(ground_truth_path),
        "retrieval_limit": args.limit,
        "k_values": list(args.k_values),
        "naive_search_mode": args.naive_search_mode,
        "group_id": args.group_id,
        "graph_expansion": not args.no_graph_expansion,
        "cross_encoder": args.cross_encoder,
        "rewrite_query": args.rewrite_query,
        "generation": args.generation,
        "judge_model": args.judge_model or os.environ.get("LLM_MODEL"),
        "ontology_files": {
            path.resolve().as_posix(): _sha256_file(path) if path.exists() and path.is_file() else None
            for path in ontology_paths
        },
        "environment": {name: os.environ[name] for name in _RUN_ENVIRONMENT_FIELDS if name in os.environ},
    }


def _identity_hash(identity: Mapping[str, Any]) -> str:
    """Return a compact deterministic run identifier."""
    serialized = json.dumps(identity, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()[:20]


def _prepare_run(
    manifest_path: Path,
    progress_path: Path,
    identity: Mapping[str, Any],
    run_id: str,
    *,
    resume: bool,
) -> bool:
    """Create or validate a run manifest before any expensive pipeline calls."""
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            msg = f"Run directory contains a different experiment: {manifest_path.parent}"
            raise SystemExit(msg)
        if not resume:
            msg = f"Run already exists; pass --resume or choose --run-dir: {manifest_path.parent}"
            raise SystemExit(msg)
        if existing.get("status") == "complete":
            return True
        existing["status"] = "running"
        existing["resumed_at"] = datetime.now(UTC).isoformat()
        existing.pop("error_type", None)
        _write_json(manifest_path, existing)
        return False
    if resume and progress_path.exists():
        msg = f"Cannot resume progress without a manifest: {progress_path}"
        raise SystemExit(msg)
    if progress_path.exists():
        msg = f"Progress file already exists; pass --resume or choose --run-dir: {progress_path}"
        raise SystemExit(msg)
    _write_json(
        manifest_path,
        {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "status": "running",
            "identity": dict(identity),
            "started_at": datetime.now(UTC).isoformat(),
        },
    )
    return False


def _update_manifest(
    manifest_path: Path,
    *,
    status: str,
    outputs: Mapping[str, Any] | None = None,
    error_type: str | None = None,
) -> None:
    """Atomically record a terminal run state while retaining its original identity."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = status
    timestamp_key = "completed_at" if status == "complete" else f"{status}_at"
    manifest[timestamp_key] = datetime.now(UTC).isoformat()
    if outputs is not None:
        manifest["outputs"] = dict(outputs)
    if error_type is not None:
        manifest["error_type"] = error_type
    _write_json(manifest_path, manifest)


def _read_progress(path: Path) -> dict[str, dict[str, Any]]:
    """Load completed prediction rows keyed by stable evaluation record identity."""
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                msg = f"Expected JSON object at {path}:{line_number}"
                raise TypeError(msg)
            record_id = _record_id(row)
            if record_id in rows:
                msg = f"Duplicate evaluation record in progress file: {record_id}"
                raise ValueError(msg)
            rows[record_id] = row
    return rows


def _record_id(row: Mapping[str, Any]) -> str:
    """Return a stable record ID for checkpointing single- or multi-turn evaluation rows."""
    value = row.get("evaluation_record_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    conversation_id = row.get("conversation_id")
    turn_number = row.get("turn_number")
    if isinstance(conversation_id, str) and conversation_id.strip() and turn_number is not None:
        return f"{conversation_id.strip()}:turn:{turn_number}"
    for field in ("question_id",):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    identity = {
        "conversation_id": row.get("conversation_id"),
        "turn_number": row.get("turn_number"),
        "question": row.get("question") or row.get("query"),
    }
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash one experiment input without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        return NaiveEvaluationBackend(
            pipeline,
            search_mode=args.naive_search_mode,
            rewrite_query=getattr(args, "rewrite_query", False),
        )

    if args.pipeline == "ontology":
        from legal_rag.ontology_rag.pipeline import OntologyRAGPipeline
        from legal_rag.settings import MilvusSettings, NaiveRagSettings, Settings

        ontology_paths = list(args.ontology_paths or _ontology_paths_from_environment())
        if not ontology_paths:
            msg = "--ontology-paths or the ONTOLOGY_PATHS environment variable is required for --pipeline ontology"
            raise SystemExit(msg)
        pipeline = OntologyRAGPipeline(
            Settings.from_environment(),
            MilvusSettings.from_environment(),
            NaiveRagSettings.from_environment(),
            ontology_paths,
        )
        return OntologyEvaluationBackend(
            pipeline,
            rewrite_query=getattr(args, "rewrite_query", False),
        )

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
    """Atomically replace a JSONL result file."""
    write_jsonl_atomic(path, rows)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace a JSON document."""
    write_json_atomic(path, data)


def _write_stdout(data: dict[str, Any]) -> None:
    """Write JSON output to stdout."""
    sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
