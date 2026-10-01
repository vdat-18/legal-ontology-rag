"""Command-line interface for Legal RAG evaluation metrics."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
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
    AdaptiveEvaluationBackend,
    CorrectiveEvaluationBackend,
    EvaluationBackend,
    EvaluationRecord,
    GraphitiEvaluationBackend,
    NaiveEvaluationBackend,
    OntologyEvaluationBackend,
    load_evaluation_records,
    run_evaluation,
)
from legal_rag.io_utils import write_json_atomic, write_jsonl_atomic

_SHARED_RUNTIME_SOURCE_MODULES = (
    "legal_rag.baml_transport",
    "legal_rag.benchmark_synthesis.vietnamese_lexical",
    "legal_rag.datastores",
    "legal_rag.embedding",
    "legal_rag.evaluation.cli",
    "legal_rag.evaluation.metrics",
    "legal_rag.evaluation.runner",
    "legal_rag.query_rewriting",
    "legal_rag.retrieval",
    "legal_rag.settings",
    "legal_rag.temporal",
)
_PREDICTION_SCHEMA_VERSION = "2.0.0"


@dataclass(frozen=True, slots=True)
class _RetrievalExecution:
    """Retrieval-only rows plus the verified active datastore identity."""

    rows: list[dict[str, Any]]
    active_backend_fingerprint: Mapping[str, Any]


def main() -> None:
    """Run the evaluation command-line interface."""
    args = _parser().parse_args()
    if args.command == "score":
        _handle_score(args)
    elif args.command == "run":
        _handle_run(args)
    elif args.command == "retrieve":
        _handle_retrieve(args)
    elif args.command == "compare":
        _handle_compare(args)


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
        choices=("naive", "adaptive", "corrective", "graphiti", "ontology"),
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
        default=10,
        help="Maximum ranked retrieval units supplied to each pipeline per question.",
    )
    run_parser.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        help="Optional number of ground-truth samples to evaluate; --limit remains the per-question retrieval budget.",
    )
    run_parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=1,
        help="Maximum samples evaluated concurrently; provider call count depends on the selected pipeline.",
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
        choices=("keyword", "semantic", "hybrid", "hybrid_rerank", "hierarchical"),
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
        help="Required for Graphiti evaluation; validates the exact local corpus and indexed group.",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resolve history-dependent queries for Naive, Adaptive, and Ontology evaluation; "
            "Graphiti and Corrective use their intrinsic history handling."
        ),
    )
    run_parser.add_argument(
        "--reference-date",
        type=_iso_date,
        default=datetime.now(UTC).date(),
        help=(
            "Frozen YYYY-MM-DD fallback for current-law questions without per-sample temporal targets; "
            "pass it explicitly when resuming later."
        ),
    )

    retrieve_parser = subparsers.add_parser("retrieve", help="Run retrieval-only evaluation without LLM calls.")
    retrieve_parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/benchmark_synthesis/groundtruth.jsonl"),
        help="Reviewed evaluation JSONL file.",
    )
    retrieve_parser.add_argument(
        "--limit", type=_positive_int, default=10, help="Maximum pipeline-native retrieval units per question."
    )
    retrieve_parser.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        help="Optional maximum number of ground-truth samples to evaluate, useful for smoke tests.",
    )
    retrieve_parser.add_argument(
        "--retriever",
        choices=("vector", "naive", "graph"),
        default="vector",
        help="Which retriever to score: the CRAG vector baseline, Naive RAG, or Graphiti.",
    )
    retrieve_parser.add_argument(
        "--naive-search-mode",
        choices=("keyword", "semantic", "hybrid", "hybrid_rerank", "hierarchical"),
        default="semantic",
        help="Naive RAG search strategy, for --retriever naive.",
    )
    retrieve_parser.add_argument(
        "--group-id",
        default="traffic_penalty_groundtruth_v1",
        help="Graphiti group to search, for --retriever graph.",
    )
    retrieve_parser.add_argument(
        "--corpus-config",
        type=Path,
        default=None,
        help="Required Graphiti corpus config used to fingerprint and validate the indexed group.",
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
        "--reference-date",
        type=_iso_date,
        default=datetime.now(UTC).date(),
        help=(
            "Frozen YYYY-MM-DD fallback for current-law Graphiti samples without query_dates, query_date, "
            "or explicit dates; unresolved temporal comparisons remain unfiltered."
        ),
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
        default=None,
        help="Path where per-sample retrieval predictions are written; defaults to a retriever-specific path.",
    )
    retrieve_parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Path where aggregate retrieval metrics are written; defaults to a retriever-specific path.",
    )
    compare_parser = subparsers.add_parser(
        "compare",
        help="Validate and compare two or more shared evaluation metric artifacts.",
    )
    compare_parser.add_argument("metrics", type=Path, nargs="+", help="Metrics JSON files to compare.")
    compare_parser.add_argument("--output", type=Path, help="Optional JSON path for the comparison report.")
    compare_parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="Write incompatibility diagnostics instead of failing when runs cannot be compared fairly.",
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
    if args.output is not None:
        _validate_input_output_artifact_paths(
            {"input predictions": args.input_path},
            {"metrics output": args.output},
        )

    metrics = evaluate_samples(load_jsonl_samples(args.input_path), k_values=args.k_values)
    if args.output:
        _write_json(args.output, metrics.to_dict())
    _write_stdout(metrics.to_dict())


def _handle_compare(args: argparse.Namespace) -> None:
    """Compare compatible evaluation artifacts without inventing a composite score."""
    from legal_rag.evaluation.comparison import compare_evaluation_runs

    if args.output is not None:
        _validate_input_output_artifact_paths(
            _comparison_input_paths(args.metrics),
            {"comparison output": args.output},
        )
    report = compare_evaluation_runs(args.metrics, allow_incompatible=args.allow_incompatible)
    if args.output:
        _write_json(args.output, report)
    _write_stdout(report)


def _load_run_records(ground_truth_path: Path, max_samples: int | None) -> list[EvaluationRecord]:
    """Load a non-empty evaluation dataset and apply the optional sample bound."""
    if not ground_truth_path.exists():
        msg = f"Evaluation file not found: {ground_truth_path}"
        raise SystemExit(msg)
    records = load_evaluation_records(ground_truth_path)
    if not records:
        msg = f"Evaluation file has no samples: {ground_truth_path}"
        raise SystemExit(msg)
    selected = records if max_samples is None else records[:max_samples]
    positions_by_id: dict[str, int] = {}
    for position, record in enumerate(selected, 1):
        record_id = _record_id(record.raw)
        previous = positions_by_id.get(record_id)
        if previous is not None:
            msg = (
                f"Duplicate evaluation record id {record_id!r} in {ground_truth_path} "
                f"at selected rows {previous} and {position}"
            )
            raise SystemExit(msg)
        positions_by_id[record_id] = position
    return selected


def _validate_ontology_paths(paths: Sequence[Path]) -> None:
    """Fail before evaluation when the requested ontology release is absent or not Turtle."""
    if not paths:
        msg = "--ontology-paths or the ONTOLOGY_PATHS environment variable is required for --pipeline ontology"
        raise SystemExit(msg)
    invalid = [path for path in paths if path.suffix.lower() != ".ttl" or not path.is_file()]
    if invalid:
        rendered = ", ".join(path.resolve().as_posix() for path in invalid)
        msg = f"Ontology inputs must be existing .ttl files: {rendered}"
        raise SystemExit(msg)


def _ontology_run_paths(pipeline: str, explicit_paths: Sequence[Path] | None) -> list[Path]:
    """Resolve and validate ontology inputs only for ontology evaluation runs."""
    paths = list(explicit_paths or _ontology_paths_from_environment())
    if pipeline == "ontology":
        _validate_ontology_paths(paths)
    return paths


def _ontology_evaluation_coverage(
    records: Sequence[EvaluationRecord],
    ontology_paths: Sequence[Path],
) -> dict[str, Any]:
    """Measure how much selected ground-truth evidence exists in ontology provenance."""
    from legal_rag.ontology_rag.index.chunk_index import ChunkIndex
    from legal_rag.ontology_rag.index.ontology_index import OntologyIndex

    ontology_index = OntologyIndex(list(ontology_paths))
    chunk_ids = ChunkIndex(ontology_index.graph).all_chunk_ids()
    coverage = _ontology_coverage_from_chunk_ids(records, chunk_ids)
    coverage["ontology_provenance_chunk_count"] = len(chunk_ids)
    return coverage


def _selected_ontology_coverage(
    pipeline: str,
    records: Sequence[EvaluationRecord],
    ontology_paths: Sequence[Path],
) -> dict[str, Any] | None:
    """Compute and log ontology coverage for the selected benchmark rows."""
    if pipeline != "ontology":
        return None
    coverage = _ontology_evaluation_coverage(records, ontology_paths)
    if coverage["sample_complete_coverage"] != 1.0:
        logger.warning(
            "Ontology provenance fully covers {} of {} evaluation samples with gold chunks ({:.1%})",
            coverage["samples_with_complete_ontology_gold"],
            coverage["samples_with_gold"],
            coverage["sample_complete_coverage"] or 0.0,
        )
    return coverage


def _ontology_coverage_metrics(coverage: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the optional metrics fragment for an ontology run."""
    return {"ontology_coverage": dict(coverage)} if coverage is not None else {}


def _validate_graph_corpus_config(args: argparse.Namespace) -> None:
    """Require a readable Graphiti corpus fingerprint whose group matches the run."""
    selected = getattr(args, "pipeline", None) == "graphiti" or getattr(args, "retriever", None) == "graph"
    if not selected:
        return
    corpus_config = getattr(args, "corpus_config", None)
    if corpus_config is None or not corpus_config.is_file():
        msg = "--corpus-config must name an existing JSON file for Graphiti evaluation"
        raise SystemExit(msg)

    from legal_rag.graphiti_rag.indexer import load_corpus_metadata

    corpus_metadata = load_corpus_metadata(corpus_config)
    if corpus_metadata.group_id != args.group_id:
        msg = (
            f"Graphiti corpus config group_id {corpus_metadata.group_id!r} does not match --group-id {args.group_id!r}"
        )
        raise SystemExit(msg)


def _ontology_coverage_from_chunk_ids(
    records: Sequence[EvaluationRecord],
    ontology_chunk_ids: frozenset[str],
) -> dict[str, Any]:
    """Summarize evidence coverage globally and by benchmark bundle kind."""

    def summarize(selected: Sequence[EvaluationRecord]) -> dict[str, Any]:
        gold_sets = [_evaluation_gold_chunk_ids(record) for record in selected]
        with_gold = [chunk_ids for chunk_ids in gold_sets if chunk_ids]
        reference_count = sum(len(chunk_ids) for chunk_ids in with_gold)
        covered_reference_count = sum(len(chunk_ids & ontology_chunk_ids) for chunk_ids in with_gold)
        unique_gold = frozenset().union(*with_gold) if with_gold else frozenset()
        covered_unique_gold = unique_gold & ontology_chunk_ids
        any_count = sum(bool(chunk_ids & ontology_chunk_ids) for chunk_ids in with_gold)
        complete_count = sum(chunk_ids <= ontology_chunk_ids for chunk_ids in with_gold)
        sample_denominator = len(with_gold)
        return {
            "sample_count": len(selected),
            "samples_with_gold": sample_denominator,
            "gold_chunk_reference_count": reference_count,
            "covered_gold_chunk_reference_count": covered_reference_count,
            "gold_chunk_reference_coverage": (covered_reference_count / reference_count if reference_count else None),
            "unique_gold_chunk_count": len(unique_gold),
            "covered_unique_gold_chunk_count": len(covered_unique_gold),
            "unique_gold_chunk_coverage": len(covered_unique_gold) / len(unique_gold) if unique_gold else None,
            "samples_with_any_ontology_gold": any_count,
            "sample_any_coverage": any_count / sample_denominator if sample_denominator else None,
            "samples_with_complete_ontology_gold": complete_count,
            "sample_complete_coverage": complete_count / sample_denominator if sample_denominator else None,
        }

    report = summarize(records)
    bundle_kinds = sorted({record.sample.bundle_kind or "unspecified" for record in records})
    report["by_bundle_kind"] = {
        bundle_kind: summarize(
            [record for record in records if (record.sample.bundle_kind or "unspecified") == bundle_kind]
        )
        for bundle_kind in bundle_kinds
    }
    return report


def _evaluation_gold_chunk_ids(record: EvaluationRecord) -> frozenset[str]:
    """Return every chunk referenced directly or through required claims."""
    return frozenset(
        (
            *record.sample.expected_chunk_ids,
            *(chunk_id for claim in record.sample.expected_claims for chunk_id in claim.chunk_ids),
        )
    )


def _handle_run(args: argparse.Namespace) -> None:
    """Run one selected RAG pipeline and score it with retrieval and optional generation metrics."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    _validate_retrieval_metric_budget(args)

    ontology_paths = _ontology_run_paths(args.pipeline, args.ontology_paths)
    input_paths = _evaluation_input_paths(
        args,
        ontology_paths=ontology_paths,
        include_corpus_chunks=False,
    )
    _validate_explicit_artifact_paths(
        args,
        include_generation=args.generation,
        input_paths=input_paths,
    )
    input_paths = _evaluation_input_paths(args, ontology_paths=ontology_paths)
    _validate_explicit_artifact_paths(
        args,
        include_generation=args.generation,
        input_paths=input_paths,
    )
    ground_truth_path = args.ground_truth
    records = _load_run_records(ground_truth_path, args.max_samples)
    _validate_graph_corpus_config(args)

    identity = _run_identity(args, ground_truth_path)
    completed_metrics = _completed_explicit_run(args, identity)
    if completed_metrics is not None:
        _write_stdout(completed_metrics)
        return

    backend, active_backend_fingerprint = _build_preflight_backend(args)
    identity["active_backend_fingerprint"] = _json_compatible(active_backend_fingerprint)
    identity["active_backend_validated"] = True
    run_id = _identity_hash(identity)
    run_dir = args.run_dir or Path("data/evaluation/runs") / run_id
    manifest_path = run_dir / "manifest.json"
    progress_path = run_dir / "predictions.progress.jsonl"
    result_paths = _run_result_paths(args, run_dir)
    predictions_output = result_paths["predictions"]
    metrics_output = result_paths["metrics"]
    generation_output = result_paths.get("generation")
    try:
        artifact_paths = {
            "manifest": manifest_path,
            "progress": progress_path,
            **result_paths,
            "metrics manifest": metrics_output.with_suffix(".manifest.json"),
        }
        _validate_distinct_artifact_paths(artifact_paths)
        _validate_input_output_artifact_paths(input_paths, artifact_paths)
    except BaseException:
        asyncio.run(backend.close())
        raise
    if _prepare_or_replay_active_run(
        manifest_path,
        progress_path,
        result_paths,
        identity,
        run_id,
        backend,
        resume=args.resume,
    ):
        return
    backend_closed = False
    try:
        ontology_coverage = _selected_ontology_coverage(args.pipeline, records, ontology_paths)
        expected_workloads = {
            _record_id(record.raw): _evaluation_workload_sha256(record.raw, args.pipeline) for record in records
        }
        completed = (
            _read_progress(
                progress_path,
                expected_workloads=expected_workloads,
                pipeline=args.pipeline,
            )
            if args.resume
            else {}
        )
        remaining_records = [record for record in records if _record_id(record.raw) not in completed]
        new_rows: list[dict[str, Any]] = []
        if remaining_records:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8", newline="\n") as progress_file:

                def _append_prediction(row: dict[str, Any]) -> None:
                    """Persist each completed prediction immediately for safe resume."""
                    record_id = _record_id(row)
                    row["evaluation_record_id"] = record_id
                    row["evaluation_workload_sha256"] = expected_workloads[record_id]
                    progress_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    progress_file.flush()

                try:
                    new_rows = asyncio.run(
                        run_evaluation(
                            remaining_records,
                            backend,
                            limit=args.limit,
                            concurrency=args.concurrency,
                            on_row=_append_prediction,
                        )
                    )
                finally:
                    backend_closed = True
        else:
            asyncio.run(backend.close())
            backend_closed = True

        rows_by_id = {**completed, **{_record_id(row): row for row in new_rows}}
        rows = [rows_by_id[_record_id(record.raw)] for record in records]
        _write_jsonl(predictions_output, rows)
        metrics = evaluate_samples([EvaluationSample.from_mapping(row) for row in rows], k_values=args.k_values)
        metrics_payload = {
            "schema_version": _PREDICTION_SCHEMA_VERSION,
            "run_id": run_id,
            "pipeline": args.pipeline,
            "active_backend_validated": True,
            "active_backend_fingerprint": identity["active_backend_fingerprint"],
            "answer_model": identity["answer_model"],
            "llm_base_url_sha256": identity["llm_base_url_sha256"],
            "embedding_profile": identity["embedding_profile"],
            "embedding_profile_sha256": identity["embedding_profile_sha256"],
            "llm_extra_request_body_sha256": identity["llm_extra_request_body_sha256"],
            "external_evidence_policy": identity["external_evidence_policy"],
            "corpus_chunks_sha256": identity["corpus_chunks_sha256"],
            "ground_truth_sha256": _sha256_file(ground_truth_path),
            "max_samples": args.max_samples,
            "retrieval_limit": args.limit,
            "concurrency": args.concurrency,
            "reference_date": args.reference_date.isoformat(),
            "history_resolution_enabled": _history_resolution_enabled(args),
            "query_rewrite_policy": _query_rewrite_policy(args),
            "judge_model": identity["judge_model"] if args.generation else None,
            "judge_batch_size": identity["judge_batch_size"],
            "judge_concurrency": identity["judge_concurrency"],
            "judge_prompt_sha256": (
                identity["prompt_sha256"].get("answer_evaluation.baml") if args.generation else None
            ),
            **metrics.to_dict(),
            "slices": evaluate_slices(rows, k_values=args.k_values),
            **_ontology_coverage_metrics(ontology_coverage),
        }
        if args.generation:
            report = _score_generation(rows, args)
            _write_generation_report(generation_output, report)
            metrics_payload["generation"] = report.aggregate
            metrics_payload["generation_metric_coverage"] = report.metric_coverage
        _write_json(metrics_output, metrics_payload)
        _update_manifest(
            manifest_path,
            status="complete",
            outputs={
                "predictions": predictions_output.resolve().as_posix(),
                "metrics": metrics_output.resolve().as_posix(),
                "generation": generation_output.resolve().as_posix() if generation_output is not None else None,
            },
            output_sha256={
                "predictions": _sha256_file(predictions_output),
                "metrics": _sha256_file(metrics_output),
                **({"generation": _sha256_file(generation_output)} if generation_output is not None else {}),
            },
        )
        _write_json(metrics_output.with_suffix(".manifest.json"), _read_run_manifest(manifest_path))
    except BaseException as error:
        _update_manifest(manifest_path, status="failed", error_type=type(error).__name__)
        raise
    finally:
        if not backend_closed:
            asyncio.run(backend.close())
    _write_stdout(metrics_payload)


def _completed_explicit_run(args: argparse.Namespace, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return one validated completed explicit run without opening live infrastructure."""
    if args.run_dir is None:
        return None
    manifest_path = args.run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None

    manifest = _read_run_manifest(manifest_path)
    _validate_persisted_run_identity(manifest, identity, manifest_path)
    if not args.resume:
        msg = f"Run already exists; pass --resume or choose --run-dir: {args.run_dir}"
        raise SystemExit(msg)
    if manifest.get("status") != "complete":
        return None
    result_paths = _run_result_paths(args, args.run_dir)
    _validate_manifest_output_paths(manifest, manifest_path, result_paths)
    return _completed_metrics(manifest, manifest_path, result_paths["metrics"])


def _read_run_manifest(path: Path) -> dict[str, Any]:
    """Read one run manifest as a JSON object."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        msg = f"Run manifest must be a JSON object: {path}"
        raise TypeError(msg)
    return manifest


def _validate_persisted_run_identity(
    manifest: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    """Validate static identity plus the persisted active-backend attestation."""
    manifest_identity = manifest.get("identity")
    if not isinstance(manifest_identity, Mapping):
        msg = f"Run manifest has no experiment identity: {manifest_path}"
        raise SystemExit(msg)
    persisted_identity = dict(manifest_identity)
    active_fingerprint = persisted_identity.pop("active_backend_fingerprint", None)
    active_validated = persisted_identity.pop("active_backend_validated", None)
    if persisted_identity != expected_identity:
        msg = f"Run directory contains a different experiment: {manifest_path.parent}"
        raise SystemExit(msg)
    if active_validated is not True or not isinstance(active_fingerprint, Mapping) or not active_fingerprint:
        msg = f"Completed run has no validated active-backend fingerprint: {manifest_path}"
        raise SystemExit(msg)
    if manifest.get("run_id") != _identity_hash(manifest_identity):
        msg = f"Completed run ID does not match its experiment identity: {manifest_path}"
        raise SystemExit(msg)


def _completed_metrics(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    default_metrics_path: Path,
) -> dict[str, Any]:
    """Load the metrics referenced by one completed run manifest."""
    _verify_manifest_output_digests(manifest, manifest_path)
    outputs = manifest.get("outputs")
    stored_metrics = outputs.get("metrics") if isinstance(outputs, Mapping) else None
    metrics_path = (
        _resolve_manifest_output_path(manifest_path, stored_metrics)
        if isinstance(stored_metrics, str)
        else default_metrics_path
    )
    if not metrics_path.is_file():
        msg = f"Completed run is missing its metrics output: {metrics_path}"
        raise SystemExit(msg)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        msg = f"Completed run metrics must be a JSON object: {metrics_path}"
        raise TypeError(msg)
    manifest_identity = manifest.get("identity")
    expected_active_validated = (
        manifest_identity.get("active_backend_validated") if isinstance(manifest_identity, Mapping) else None
    )
    expected_active_fingerprint = (
        manifest_identity.get("active_backend_fingerprint") if isinstance(manifest_identity, Mapping) else None
    )
    if metrics.get("run_id") != manifest.get("run_id"):
        msg = f"Completed run metrics ID does not match its manifest: {metrics_path}"
        raise SystemExit(msg)
    if metrics.get("active_backend_validated") != expected_active_validated:
        msg = f"Completed run metrics backend validation does not match its manifest: {metrics_path}"
        raise SystemExit(msg)
    if metrics.get("active_backend_fingerprint") != expected_active_fingerprint:
        msg = f"Completed run metrics backend fingerprint does not match its manifest: {metrics_path}"
        raise SystemExit(msg)
    return metrics


def _prepare_or_replay_active_run(
    manifest_path: Path,
    progress_path: Path,
    result_paths: Mapping[str, Path],
    identity: Mapping[str, Any],
    run_id: str,
    backend: EvaluationBackend,
    *,
    resume: bool,
) -> bool:
    """Prepare an active run or replay a completed default run after live preflight."""
    try:
        already_complete = _prepare_run(manifest_path, progress_path, identity, run_id, resume=resume)
    except BaseException:
        asyncio.run(backend.close())
        raise
    if not already_complete:
        return False
    try:
        manifest = _read_run_manifest(manifest_path)
        _validate_manifest_output_paths(manifest, manifest_path, result_paths)
        _write_stdout(_completed_metrics(manifest, manifest_path, result_paths["metrics"]))
    finally:
        asyncio.run(backend.close())
    return True


def _build_preflight_backend(args: argparse.Namespace) -> tuple[EvaluationBackend, Mapping[str, Any]]:
    """Build one backend, validate its active corpus, and clean up failed startup."""
    backend = _build_evaluation_backend(args)
    try:
        fingerprint = asyncio.run(backend.preflight_corpus())
    except BaseException:
        asyncio.run(backend.close())
        raise
    return backend, fingerprint


def _score_generation(rows: list[dict[str, Any]], args: argparse.Namespace) -> Any:
    """Judge the generated answers of *rows* with the BAML LLM judge and return the report."""
    from legal_rag.evaluation.generation.core.dataset import EvalSample
    from legal_rag.evaluation.generation.judge import BamlJudgeClient
    from legal_rag.evaluation.generation.runner import evaluate_samples as evaluate_generation_samples
    from legal_rag.settings import Settings

    samples = [replace(EvalSample.from_mapping(row), reference_date=args.reference_date) for row in rows]
    settings = Settings.from_environment()
    if args.judge_model:
        settings = replace(settings, llm_model=args.judge_model)
    judge = BamlJudgeClient(settings)
    return asyncio.run(
        evaluate_generation_samples(
            samples, judge, batch_size=args.judge_batch_size, concurrency=args.judge_concurrency
        )
    )


def _write_generation_report(output_path: Path | None, report: Any) -> None:
    """Write a generation report after enforcing the enabled-run path invariant."""
    if output_path is None:
        msg = "Generation output path is required when generation scoring is enabled"
        raise RuntimeError(msg)
    _write_json(output_path, report.to_dict())


def _history_resolution_enabled(args: argparse.Namespace) -> bool:
    """Return whether the selected backend resolves context-dependent follow-up questions."""
    if args.pipeline in {"corrective", "graphiti"}:
        return True
    return bool(args.rewrite_query)


def _query_rewrite_policy(args: argparse.Namespace) -> str:
    """Name the query-history policy persisted in the experiment identity."""
    if args.pipeline == "corrective":
        return "intrinsic_when_history_present"
    if args.pipeline == "graphiti":
        return "intrinsic"
    if args.pipeline == "ontology" and args.rewrite_query:
        return "ontology_planner_always"
    return "enabled_when_history_present" if args.rewrite_query else "disabled"


def _retrieval_run_identity(args: argparse.Namespace, ground_truth_path: Path) -> dict[str, Any]:
    """Build a reproducible identity for retrieval-only artifacts."""
    pipeline = {"vector": "corrective_vector", "naive": "naive", "graph": "graphiti"}[args.retriever]
    effective_config = _effective_retrieval_config(args)
    corpus_config = None
    if args.retriever == "graph" and args.corpus_config is not None:
        corpus_config = {
            "sha256": _sha256_file(args.corpus_config),
        }
    source_modules = {
        "vector": ("legal_rag.corrective_rag.retrieval",),
        "naive": ("legal_rag.naive_rag.retriever",),
        "graph": ("legal_rag.graphiti_rag.retriever",),
    }[args.retriever]
    return {
        "command": "retrieve",
        "pipeline": pipeline,
        "retriever": args.retriever,
        "ground_truth_sha256": _sha256_file(ground_truth_path),
        "max_samples": args.max_samples,
        "retrieval_limit": args.limit,
        "k_values": list(args.k_values),
        "concurrency": 1,
        "reference_date": args.reference_date.isoformat(),
        "history_resolution_enabled": False,
        "query_rewrite_policy": "disabled",
        "embedding_profile": effective_config["embedding_profile"],
        "external_evidence_policy": "closed_corpus",
        "naive_search_mode": args.naive_search_mode if args.retriever == "naive" else None,
        "group_id": args.group_id if args.retriever == "graph" else None,
        "graph_expansion": not args.no_graph_expansion if args.retriever == "graph" else None,
        "cross_encoder": args.cross_encoder if args.retriever == "graph" else None,
        "corpus_config": corpus_config,
        "embedding_profile_sha256": effective_config["embedding_profile_sha256"],
        "corpus_chunks_sha256": effective_config["chunks_sha256"],
        "dependency_lock_sha256": _dependency_lock_sha256(),
        "effective_config": effective_config,
        "source_sha256": {
            module_name: _python_source_sha256(module_name)
            for module_name in (
                *source_modules,
                *_SHARED_RUNTIME_SOURCE_MODULES,
            )
        },
    }


def _effective_retrieval_config(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve secret-free settings and local corpus fingerprints for retrieval-only runs."""
    chunks_dir = Path(os.environ.get("NAIVE_RAG_CHUNKS_DIR", "data/processed/groundtruth")).resolve()
    selected_sources = _graphiti_corpus_sources(args.corpus_config) if args.retriever == "graph" else None
    config: dict[str, Any] = {
        "embedding_profile": os.environ.get("EMBEDDING_PROFILE"),
        "embedding_profile_sha256": _embedding_profile_sha256(os.environ.get("EMBEDDING_PROFILE")),
        "llm_base_url_sha256": _optional_text_sha256(os.environ.get("LLM_BASE_URL")),
        "milvus_database": os.environ.get("MILVUS_DATABASE"),
        "milvus_uri_sha256": _optional_text_sha256(os.environ.get("MILVUS_URI")),
        "chunks_sha256": _chunk_corpus_sha256(chunks_dir, source_names=selected_sources),
    }
    if args.retriever == "naive" and os.environ.get("EMBEDDING_PROFILE"):
        from legal_rag.settings import NaiveRagSettings

        settings = replace(NaiveRagSettings.from_environment(), search_mode=args.naive_search_mode)
        naive_config = _json_compatible(asdict(settings))
        naive_config.pop("chunks_dir", None)
        config["naive_rag"] = naive_config
    elif args.retriever == "graph" and os.environ.get("EMBEDDING_PROFILE"):
        from legal_rag.settings import GraphitiSettings

        settings = replace(
            GraphitiSettings.from_environment(),
            expand_around_top_fact=not args.no_graph_expansion,
            rerank_with_cross_encoder=args.cross_encoder,
        )
        config["graphiti"] = _json_compatible(asdict(settings))
        config["neo4j_uri_sha256"] = _optional_text_sha256(os.environ.get("NEO4J_URI"))
    elif args.retriever == "vector":
        from legal_rag.corrective_rag.generate.config import CorrectiveRetrievalConfig

        retrieval_config = asdict(CorrectiveRetrievalConfig.from_env())
        retrieval_config.pop("milvus_token", None)
        milvus_uri = str(retrieval_config.pop("milvus_uri"))
        embedding_profile = str(retrieval_config["embedding_profile"])
        config.update(
            {
                "embedding_profile": embedding_profile,
                "embedding_profile_sha256": _embedding_profile_sha256(embedding_profile),
                "milvus_database": retrieval_config["milvus_database"],
                "milvus_uri_sha256": _optional_text_sha256(milvus_uri),
                "corrective_vector": _json_compatible(retrieval_config),
            }
        )
    return config


def _run_identity(args: argparse.Namespace, ground_truth_path: Path) -> dict[str, Any]:
    """Build the secret-free identity that distinguishes one experiment configuration."""
    effective_config = _effective_run_config(args)
    ontology_paths = list(args.ontology_paths or _ontology_paths_from_environment())
    adaptive_config = None
    if args.pipeline == "adaptive":
        from legal_rag.settings import AdaptiveRagSettings

        adaptive_config = asdict(AdaptiveRagSettings.from_environment())
    ontology_runtime = None
    if args.pipeline == "ontology":
        ontology_runtime = {
            "reference_date": args.reference_date.isoformat(),
            "query_planner_sha256": _baml_runtime_source_sha256("ontology_query_planning.baml"),
            "generation_prompt_sha256": _baml_runtime_source_sha256("naive_rag_generation.baml"),
        }
    chunks_dir = Path(os.environ.get("NAIVE_RAG_CHUNKS_DIR", "data/processed/groundtruth")).resolve()
    graphiti_sources: frozenset[str] | None = None
    graphiti_corpus = None
    if args.pipeline == "graphiti" and args.corpus_config is not None:
        graphiti_sources = _graphiti_corpus_sources(args.corpus_config)
        graphiti_corpus = {
            "sha256": _sha256_file(args.corpus_config),
            "chunks_sha256": _chunk_corpus_sha256(chunks_dir, source_names=graphiti_sources),
        }
    return {
        "pipeline": args.pipeline,
        "answer_model": effective_config.get("llm_model"),
        "llm_base_url_sha256": effective_config.get("llm_base_url_sha256"),
        "embedding_profile": effective_config.get("embedding_profile"),
        "embedding_profile_sha256": effective_config.get("embedding_profile_sha256"),
        "llm_extra_request_body_sha256": effective_config.get("llm_extra_request_body_sha256"),
        "external_evidence_policy": _external_evidence_policy(args.pipeline, effective_config),
        "corpus_chunks_sha256": _chunk_corpus_sha256(chunks_dir, source_names=graphiti_sources),
        "ground_truth_sha256": _sha256_file(ground_truth_path),
        "reference_date": args.reference_date.isoformat(),
        "max_samples": args.max_samples,
        "retrieval_limit": args.limit,
        "concurrency": args.concurrency,
        "k_values": list(args.k_values),
        "naive_search_mode": args.naive_search_mode if args.pipeline == "naive" else None,
        "group_id": args.group_id if args.pipeline == "graphiti" else None,
        "graph_expansion": not args.no_graph_expansion if args.pipeline == "graphiti" else None,
        "cross_encoder": args.cross_encoder if args.pipeline == "graphiti" else None,
        "rewrite_query": args.rewrite_query if args.pipeline in {"naive", "adaptive", "ontology"} else None,
        "history_resolution_enabled": _history_resolution_enabled(args),
        "query_rewrite_policy": _query_rewrite_policy(args),
        "adaptive_config": adaptive_config,
        "generation": args.generation,
        "judge_model": args.judge_model or os.environ.get("LLM_MODEL"),
        "judge_batch_size": args.judge_batch_size if args.generation else None,
        "judge_concurrency": args.judge_concurrency if args.generation else None,
        "ontology_sha256": (
            sorted(_sha256_file(path) for path in ontology_paths if path.exists() and path.is_file())
            if args.pipeline == "ontology"
            else None
        ),
        "ontology_runtime": ontology_runtime,
        "graphiti_corpus": graphiti_corpus,
        "dependency_lock_sha256": _dependency_lock_sha256(),
        "effective_config": effective_config,
        "prompt_sha256": _runtime_prompt_hashes(args),
        "source_sha256": _runtime_source_hashes(args.pipeline, include_generation_judge=args.generation),
    }


def _effective_run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve secret-free defaults that materially change an evaluation run."""
    config: dict[str, Any] = {
        "llm_model": os.environ.get("LLM_MODEL"),
        "llm_base_url_sha256": _optional_text_sha256(os.environ.get("LLM_BASE_URL")),
        "llm_timeout_seconds": _optional_environment_float("LLM_TIMEOUT_SECONDS"),
        "embedding_profile": os.environ.get("EMBEDDING_PROFILE"),
        "embedding_profile_sha256": _embedding_profile_sha256(os.environ.get("EMBEDDING_PROFILE")),
        "milvus_database": os.environ.get("MILVUS_DATABASE"),
        "llm_extra_request_body_sha256": _json_environment_sha256("LLM_EXTRA_REQUEST_BODY"),
    }
    if args.pipeline in {"naive", "adaptive", "ontology"} and os.environ.get("EMBEDDING_PROFILE"):
        from legal_rag.settings import NaiveRagSettings

        naive_settings = NaiveRagSettings.from_environment()
        if args.pipeline == "naive" and args.naive_search_mode:
            naive_settings = replace(naive_settings, search_mode=args.naive_search_mode)
        naive_config = _json_compatible(asdict(naive_settings))
        chunks_dir = naive_settings.chunks_dir.resolve()
        naive_config.pop("chunks_dir", None)
        naive_config["chunks_sha256"] = _directory_sha256(chunks_dir, "*.chunks.jsonl")
        config["naive_rag"] = naive_config
    if args.pipeline == "adaptive":
        from legal_rag.settings import AdaptiveRagSettings

        config["adaptive_rag"] = _json_compatible(asdict(AdaptiveRagSettings.from_environment()))
    if args.pipeline == "graphiti" and os.environ.get("EMBEDDING_PROFILE"):
        from legal_rag.settings import GraphitiSettings

        graphiti_settings = replace(
            GraphitiSettings.from_environment(),
            expand_around_top_fact=not args.no_graph_expansion,
            rerank_with_cross_encoder=args.cross_encoder,
        )
        config["graphiti"] = _json_compatible(asdict(graphiti_settings))
        config["neo4j_uri_sha256"] = _optional_text_sha256(os.environ.get("NEO4J_URI"))
    if args.pipeline == "corrective":
        corrective_config = _corrective_config_identity()
        config["corrective_rag"] = corrective_config
        config.update(
            {
                "llm_model": corrective_config["llm_model"],
                "llm_base_url_sha256": corrective_config["llm_base_url"],
                "llm_timeout_seconds": corrective_config["llm_timeout_seconds"],
                "embedding_profile": corrective_config["embedding_profile"],
                "embedding_profile_sha256": _embedding_profile_sha256(corrective_config["embedding_profile"]),
                "milvus_database": corrective_config["milvus_database"],
                "llm_extra_request_body_sha256": corrective_config["llm_extra_request_body_sha256"],
            }
        )
    return config


def _corrective_config_identity() -> dict[str, Any]:
    """Return full Corrective RAG settings with secrets and provider contents fingerprinted."""
    from legal_rag.corrective_rag.generate.config import RAGConfig

    corrective_config = asdict(RAGConfig.from_env())
    corrective_config.pop("llm_api_key", None)
    corrective_config.pop("milvus_token", None)
    corrective_config["llm_base_url"] = _optional_text_sha256(str(corrective_config["llm_base_url"]))
    corrective_config["milvus_uri"] = _optional_text_sha256(str(corrective_config["milvus_uri"]))
    extra_request_body = corrective_config.pop("llm_extra_request_body", {})
    corrective_config["llm_extra_request_body_sha256"] = _canonical_sha256(extra_request_body)
    return _json_compatible(corrective_config)


def _runtime_prompt_hashes(args: argparse.Namespace) -> dict[str, str]:
    """Fingerprint every BAML prompt used by the selected end-to-end workflow."""
    prompt_names = set() if args.pipeline == "corrective" else {"naive_rag_generation.baml"}
    if args.pipeline == "adaptive":
        prompt_names.add("adaptive_query_planning.baml")
    elif args.pipeline == "corrective":
        prompt_names.add("corrective_rag_grading.baml")
    elif args.pipeline == "ontology":
        prompt_names.add("ontology_query_planning.baml")
    if args.pipeline in {"corrective", "graphiti"} or args.rewrite_query:
        prompt_names.add("query_rewrite.baml")
    if args.generation:
        prompt_names.add("answer_evaluation.baml")
    return {name: _baml_runtime_source_sha256(name) for name in sorted(prompt_names)}


def _runtime_source_hashes(pipeline: str, *, include_generation_judge: bool) -> dict[str, str]:
    """Fingerprint pipeline and scoring Python sources even in an installed wheel."""
    packages = {
        "naive": ("legal_rag.naive_rag",),
        "adaptive": ("legal_rag.adaptive_rag", "legal_rag.naive_rag"),
        "corrective": ("legal_rag.corrective_rag",),
        "graphiti": ("legal_rag.graphiti_rag", "legal_rag.naive_rag.generator"),
        "ontology": ("legal_rag.ontology_rag", "legal_rag.naive_rag"),
    }[pipeline]
    shared = list(_SHARED_RUNTIME_SOURCE_MODULES)
    if include_generation_judge:
        shared.append("legal_rag.evaluation.generation")
    return {name: _python_source_sha256(name) for name in (*packages, *shared)}


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
    output_sha256: Mapping[str, str] | None = None,
    error_type: str | None = None,
) -> None:
    """Atomically record a terminal run state while retaining its original identity."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = status
    timestamp_key = "completed_at" if status == "complete" else f"{status}_at"
    manifest[timestamp_key] = datetime.now(UTC).isoformat()
    if outputs is not None:
        manifest["outputs"] = dict(outputs)
    if output_sha256 is not None:
        manifest["output_sha256"] = dict(output_sha256)
    if error_type is not None:
        manifest["error_type"] = error_type
    _write_json(manifest_path, manifest)


def _read_progress(
    path: Path,
    *,
    expected_workloads: Mapping[str, str],
    pipeline: str,
) -> dict[str, dict[str, Any]]:
    """Load checkpoint rows only when their input and workload bindings still match."""
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
            expected_workload = expected_workloads.get(record_id)
            if expected_workload is None:
                msg = f"Unexpected evaluation record in progress file: {record_id}"
                raise ValueError(msg)
            if record_id in rows:
                msg = f"Duplicate evaluation record in progress file: {record_id}"
                raise ValueError(msg)
            if row.get("prediction_schema_version") != _PREDICTION_SCHEMA_VERSION:
                msg = f"Progress row has an incompatible prediction schema: {record_id}"
                raise ValueError(msg)
            if row.get("pipeline") != pipeline:
                msg = f"Progress row belongs to a different pipeline: {record_id}"
                raise ValueError(msg)
            if row.get("evaluation_workload_sha256") != expected_workload:
                msg = f"Progress row does not match the current evaluation input: {record_id}"
                raise ValueError(msg)
            rows[record_id] = row
    return rows


def _evaluation_workload_sha256(row: Mapping[str, Any], pipeline: str) -> str:
    """Bind one checkpoint row to its canonical input, pipeline, and prediction schema."""
    return _canonical_sha256(
        {
            "input": row,
            "pipeline": pipeline,
            "prediction_schema_version": _PREDICTION_SCHEMA_VERSION,
        }
    )


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


def _dependency_lock_sha256() -> str | None:
    """Hash the project lock when it is available beside source or in the active checkout."""
    source_root = Path(__file__).resolve().parents[3]
    candidates = (source_root / "uv.lock", Path.cwd() / "uv.lock")
    for candidate in candidates:
        if candidate.is_file():
            return _sha256_file(candidate)
    return None


def _resolve_manifest_output_path(manifest_path: Path, value: str) -> Path:
    """Resolve one manifest output relative to the manifest that declares it."""
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _verify_manifest_output_digests(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    """Reject replay when a completed run output is missing or changed after completion."""
    outputs = manifest.get("outputs")
    output_sha256 = manifest.get("output_sha256")
    if not isinstance(outputs, Mapping) or not isinstance(output_sha256, Mapping):
        msg = f"Completed run manifest is missing output paths or SHA-256 digests: {manifest_path}"
        raise SystemExit(msg)
    expected_names = {
        name for name, value in outputs.items() if isinstance(name, str) and isinstance(value, str) and value.strip()
    }
    if not {"predictions", "metrics"} <= expected_names:
        msg = f"Completed run manifest must declare predictions and metrics outputs: {manifest_path}"
        raise SystemExit(msg)
    if set(output_sha256) != expected_names:
        msg = f"Completed run manifest output SHA-256 keys do not match its outputs: {manifest_path}"
        raise SystemExit(msg)
    for name in sorted(expected_names):
        raw_path = outputs[name]
        expected_digest = output_sha256[name]
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            msg = f"Completed run manifest has an invalid SHA-256 for {name}: {manifest_path}"
            raise SystemExit(msg)
        output_path = _resolve_manifest_output_path(manifest_path, raw_path)
        if not output_path.is_file():
            msg = f"Completed run is missing its {name} output: {output_path}"
            raise SystemExit(msg)
        if _sha256_file(output_path) != expected_digest:
            msg = f"Completed run {name} output does not match its manifest SHA-256: {output_path}"
            raise SystemExit(msg)


def _baml_runtime_source_sha256(file_name: str) -> str:
    """Hash the generated BAML source that the installed runtime executes."""
    from legal_rag.baml_client.inlinedbaml import get_baml_files

    source = get_baml_files().get(file_name)
    if not isinstance(source, str):
        msg = f"Generated BAML runtime is missing {file_name}"
        raise TypeError(msg)
    return hashlib.sha256(source.encode()).hexdigest()


def _python_source_sha256(module_name: str) -> str:
    """Hash one Python module or package without depending on the current working directory."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        msg = f"Could not locate Python source for {module_name}"
        raise ModuleNotFoundError(msg)
    if spec.submodule_search_locations:
        roots = [Path(path) for path in spec.submodule_search_locations]
        files = sorted((file.relative_to(root).as_posix(), file) for root in roots for file in root.rglob("*.py"))
    elif spec.origin:
        source_path = Path(spec.origin)
        files = [(source_path.name, source_path)]
    else:
        msg = f"Python source for {module_name} has no filesystem location"
        raise ModuleNotFoundError(msg)
    digest = hashlib.sha256()
    for relative_path, source_path in files:
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _directory_sha256(directory: Path, pattern: str) -> str | None:
    """Hash matching corpus files by relative path and content, or return None when absent."""
    if not directory.is_dir():
        return None
    files = sorted(directory.glob(pattern))
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _chunk_corpus_sha256(
    directory: Path,
    *,
    source_names: frozenset[str] | None = None,
) -> str | None:
    """Hash the exact logical chunk rows selected for one evaluation corpus.

    Hashing canonical rows rather than file boundaries gives every backend the same digest when
    it evaluates the same chunks. Graphiti can additionally restrict the digest to the documents
    selected by its corpus config, so a subset-backed graph cannot compare as a full-corpus run.
    """
    if not directory.is_dir():
        return None
    serialized_rows: list[str] = []
    for path in sorted(directory.glob("*.chunks.jsonl")):
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    msg = f"Expected a chunk JSON object at {path}:{line_number}"
                    raise TypeError(msg)
                source_name = row.get("source_name")
                if source_names is not None and source_name not in source_names:
                    continue
                serialized_rows.append(
                    json.dumps(_json_compatible(row), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                )
    if not serialized_rows:
        return None
    digest = hashlib.sha256()
    for row in sorted(serialized_rows):
        digest.update(row.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _graphiti_corpus_sources(corpus_config: Path | None) -> frozenset[str]:
    """Return the exact source-document selection declared for one Graphiti experiment."""
    if corpus_config is None:
        msg = "--corpus-config is required for Graphiti evaluation"
        raise SystemExit(msg)
    from legal_rag.graphiti_rag.indexer import load_corpus_metadata

    return frozenset(load_corpus_metadata(corpus_config).documents)


def _json_compatible(value: Any) -> Any:
    """Convert typed settings into deterministic JSON-compatible values."""
    if isinstance(value, Path):
        return value.resolve().as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, tuple | list):
        return [_json_compatible(item) for item in value]
    return value


def _canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value without persisting provider-specific contents."""
    encoded = json.dumps(_json_compatible(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_environment_sha256(name: str) -> str | None:
    """Hash an optional JSON environment value while keeping its contents out of manifests."""
    raw_value = os.environ.get(name)
    if not raw_value:
        return None
    return _canonical_sha256(json.loads(raw_value))


def _optional_environment_float(name: str) -> float | None:
    """Read an optional numeric setting for experiment identity."""
    value = os.environ.get(name)
    return float(value) if value is not None else None


def _optional_text_sha256(value: str | None) -> str | None:
    """Fingerprint an optional connection identifier without exposing it."""
    return hashlib.sha256(value.encode()).hexdigest() if value else None


def _embedding_profile_sha256(profile_id: str | None) -> str | None:
    """Fingerprint the resolved embedding model, tokenizer, and query instruction."""
    if not profile_id:
        return None
    from legal_rag.embedding.profiles import load_configured_embedding_profile

    return _canonical_sha256(asdict(load_configured_embedding_profile(profile_id)))


def _external_evidence_policy(pipeline: str, effective_config: Mapping[str, Any]) -> str:
    """Describe whether a run can retrieve evidence outside the versioned local corpus."""
    if pipeline != "corrective":
        return "closed_corpus"
    corrective = effective_config.get("corrective_rag")
    if isinstance(corrective, Mapping) and corrective.get("web_search_enabled") is True:
        return "live_web"
    return "closed_corpus"


def _has_environment(*names: str) -> bool:
    """Return whether every required runtime setting is available."""
    return all(os.environ.get(name) for name in names)


def _load_graphiti_evaluation_corpus(corpus_config: Path | None) -> tuple[Any, tuple[Any, ...], dict[str, date]]:
    """Load and validate the exact local Graphiti corpus shared by both evaluation commands."""
    from legal_rag.graphiti_rag.indexer import load_corpus_metadata
    from legal_rag.naive_rag.indexer import load_chunks

    if corpus_config is None:
        msg = "--corpus-config is required for Graphiti evaluation"
        raise SystemExit(msg)
    corpus_metadata = load_corpus_metadata(corpus_config)
    chunks_dir = Path(os.environ.get("NAIVE_RAG_CHUNKS_DIR", "data/processed/groundtruth"))
    selected_sources = frozenset(corpus_metadata.documents)
    expected_chunks = tuple(chunk for chunk in load_chunks(chunks_dir) if chunk.source_name in selected_sources)
    present_sources = frozenset(chunk.source_name for chunk in expected_chunks)
    missing_sources = sorted(selected_sources - present_sources)
    if missing_sources:
        msg = f"Graphiti corpus config references documents without local chunks: {', '.join(missing_sources)}"
        raise SystemExit(msg)
    expected_source_dates = {
        source_name: document.effective_date
        for source_name, document in corpus_metadata.documents.items()
        if getattr(document, "effective_date", None) is not None
    }
    return corpus_metadata, expected_chunks, expected_source_dates


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

    if args.pipeline == "adaptive":
        from legal_rag.adaptive_rag.pipeline import AdaptiveRAGPipeline
        from legal_rag.settings import AdaptiveRagSettings, MilvusSettings, NaiveRagSettings, Settings

        pipeline = AdaptiveRAGPipeline(
            Settings.from_environment(),
            MilvusSettings.from_environment(),
            NaiveRagSettings.from_environment(),
            AdaptiveRagSettings.from_environment(),
        )
        return AdaptiveEvaluationBackend(
            pipeline,
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
            reference_date=args.reference_date,
        )

    if args.pipeline == "corrective":
        from legal_rag.corrective_rag.generate.config import RAGConfig
        from legal_rag.corrective_rag.generate.pipeline import CorrectiveRAGPipeline

        chunks_dir = Path(os.environ.get("NAIVE_RAG_CHUNKS_DIR", "data/processed/groundtruth")).resolve()
        return CorrectiveEvaluationBackend(CorrectiveRAGPipeline(RAGConfig.from_env()), chunks_dir=chunks_dir)

    from legal_rag.embedding.profiles import load_configured_embedding_profile
    from legal_rag.graphiti_rag.pipeline import GraphitiRAGPipeline
    from legal_rag.settings import GraphitiSettings, Neo4jSettings, Settings

    graphiti_settings = GraphitiSettings.from_environment()
    graphiti_settings = replace(
        graphiti_settings,
        expand_around_top_fact=not args.no_graph_expansion,
        rerank_with_cross_encoder=args.cross_encoder,
    )
    corpus_metadata, expected_chunks, expected_source_dates = _load_graphiti_evaluation_corpus(args.corpus_config)
    known_document_numbers = frozenset(document.document_number for document in corpus_metadata.documents.values())
    neo4j_settings = Neo4jSettings.from_environment()
    pipeline = GraphitiRAGPipeline(
        Settings.from_environment(),
        neo4j_settings,
        graphiti_settings,
        args.group_id,
        known_document_numbers=known_document_numbers,
    )
    return GraphitiEvaluationBackend(
        pipeline,
        neo4j_settings=neo4j_settings,
        group_id=args.group_id,
        expected_chunks=expected_chunks,
        expected_source_dates=expected_source_dates,
        embedding_profile=load_configured_embedding_profile(graphiti_settings.embedding_profile),
        reference_date=args.reference_date,
    )


def _handle_retrieve(args: argparse.Namespace) -> None:
    """Run the selected retriever for each ground-truth sample and score it."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    _validate_retrieval_metric_budget(args)

    input_paths = _evaluation_input_paths(args, include_corpus_chunks=False)
    _validate_explicit_artifact_paths(
        args,
        include_generation=False,
        input_paths=input_paths,
    )
    input_paths = _evaluation_input_paths(args)
    _validate_explicit_artifact_paths(
        args,
        include_generation=False,
        input_paths=input_paths,
    )
    ground_truth_path = args.ground_truth
    records = _load_run_records(ground_truth_path, args.max_samples)
    _validate_graph_corpus_config(args)

    run_started_at = datetime.now(UTC).isoformat()

    retrieve = {
        "vector": _retrieve_with_vector,
        "naive": _retrieve_with_naive,
        "graph": _retrieve_with_graph,
    }[args.retriever]
    identity = _retrieval_run_identity(args, ground_truth_path)
    preflight = retrieve((), args)
    if preflight.rows:
        msg = "Retrieval preflight unexpectedly executed workload queries"
        raise RuntimeError(msg)
    identity["active_backend_fingerprint"] = _json_compatible(preflight.active_backend_fingerprint)
    identity["active_backend_validated"] = True
    run_id = _identity_hash(identity)

    predictions_output, metrics_output = _retrieval_output_paths(args, run_id)
    manifest_path = _retrieval_manifest_path(metrics_output)
    artifact_paths = {
        "predictions": predictions_output,
        "metrics": metrics_output,
        "metrics manifest": manifest_path,
    }
    _validate_distinct_artifact_paths(artifact_paths)
    _validate_input_output_artifact_paths(input_paths, artifact_paths)
    if _replay_completed_retrieval(
        manifest_path,
        predictions_output,
        metrics_output,
        identity,
        run_id,
    ):
        return
    execution = retrieve(records, args)
    active_backend_fingerprint = _json_compatible(execution.active_backend_fingerprint)
    if active_backend_fingerprint != identity["active_backend_fingerprint"]:
        msg = "Active retrieval backend changed between preflight and workload execution"
        raise SystemExit(msg)
    rows = execution.rows
    _write_jsonl(predictions_output, rows)
    metrics = evaluate_samples([EvaluationSample.from_mapping(row) for row in rows], k_values=args.k_values)
    metrics_payload = {
        "schema_version": _PREDICTION_SCHEMA_VERSION,
        "run_id": run_id,
        "pipeline": {"vector": "corrective_vector", "naive": "naive", "graph": "graphiti"}[args.retriever],
        "retriever": args.retriever,
        "search_mode": args.naive_search_mode if args.retriever == "naive" else None,
        "ground_truth_sha256": _sha256_file(ground_truth_path),
        "max_samples": args.max_samples,
        "retrieval_limit": args.limit,
        "concurrency": 1,
        "reference_date": args.reference_date.isoformat(),
        "history_resolution_enabled": False,
        "query_rewrite_policy": "disabled",
        "active_backend_validated": True,
        "active_backend_fingerprint": identity["active_backend_fingerprint"],
        "embedding_profile": identity["effective_config"]["embedding_profile"],
        "embedding_profile_sha256": identity["embedding_profile_sha256"],
        "external_evidence_policy": identity["external_evidence_policy"],
        "corpus_chunks_sha256": identity["corpus_chunks_sha256"],
        "effective_config": identity["effective_config"],
        **metrics.to_dict(),
        "slices": evaluate_slices(rows, k_values=args.k_values),
    }
    _write_json(metrics_output, metrics_payload)
    _write_json(
        manifest_path,
        {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "status": "complete",
            "identity": identity,
            "started_at": run_started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "outputs": {
                "predictions": predictions_output.resolve().as_posix(),
                "metrics": metrics_output.resolve().as_posix(),
            },
            "output_sha256": {
                "predictions": _sha256_file(predictions_output),
                "metrics": _sha256_file(metrics_output),
            },
        },
    )
    _write_stdout(metrics_payload)


def _validate_retrieval_metric_budget(args: argparse.Namespace) -> None:
    """Reject metric cutoffs that the configured hard retrieval cap can never reach."""
    largest_k = max(args.k_values)
    if largest_k > args.limit:
        msg = f"--limit ({args.limit}) must be at least the largest --k-values cutoff ({largest_k})"
        raise SystemExit(msg)


def _retrieval_output_paths(args: argparse.Namespace, run_id: str) -> tuple[Path, Path]:
    """Resolve identity-addressed defaults so distinct experiments cannot overwrite each other."""
    run_dir = Path("data/evaluation/runs") / run_id
    predictions_default = run_dir / "predictions.jsonl"
    metrics_default = run_dir / "metrics.json"
    return args.predictions_output or predictions_default, args.metrics_output or metrics_default


def _run_result_paths(args: argparse.Namespace, run_dir: Path) -> dict[str, Path]:
    """Resolve the exact result paths requested for one end-to-end run."""
    paths = {
        "predictions": args.predictions_output or run_dir / "predictions.jsonl",
        "metrics": args.metrics_output or run_dir / "metrics.json",
    }
    if args.generation:
        paths["generation"] = args.generation_output or run_dir / "generation.json"
    return paths


def _evaluation_input_paths(
    args: argparse.Namespace,
    *,
    ontology_paths: Sequence[Path] = (),
    include_corpus_chunks: bool = True,
) -> dict[str, Path]:
    """Return versioned files that evaluation output must never overwrite."""
    paths = {"ground truth": args.ground_truth}
    corpus_config = getattr(args, "corpus_config", None)
    if corpus_config is not None:
        paths["corpus config"] = corpus_config
    if include_corpus_chunks:
        paths.update(
            {f"corpus chunks {index}": path for index, path in enumerate(_evaluation_chunk_input_paths(args), 1)}
        )
    paths.update({f"ontology input {index}": path for index, path in enumerate(ontology_paths, 1)})
    return paths


def _evaluation_chunk_input_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    """Return chunk files whose rows contribute to the selected evaluation corpus."""
    chunks_dir = Path(os.environ.get("NAIVE_RAG_CHUNKS_DIR", "data/processed/groundtruth")).resolve()
    chunk_paths = tuple(sorted(chunks_dir.glob("*.chunks.jsonl")))
    is_graphiti = getattr(args, "pipeline", None) == "graphiti" or getattr(args, "retriever", None) == "graph"
    if not is_graphiti:
        return chunk_paths

    selected_sources = _graphiti_corpus_sources(getattr(args, "corpus_config", None))
    selected_paths: list[Path] = []
    for path in chunk_paths:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    msg = f"Expected a chunk JSON object at {path}:{line_number}"
                    raise TypeError(msg)
                if row.get("source_name") in selected_sources:
                    selected_paths.append(path)
                    break
    return tuple(selected_paths)


def _comparison_input_paths(metrics_paths: Sequence[Path]) -> dict[str, Path]:
    """Return metrics, adjacent manifests, and every output declared by those manifests."""
    paths: dict[str, Path] = {}
    for run_index, metrics_path in enumerate(metrics_paths, 1):
        paths[f"metrics input {run_index}"] = metrics_path
        manifest_candidates = (
            metrics_path.with_suffix(".manifest.json"),
            metrics_path.parent / "manifest.json",
        )
        for manifest_index, manifest_path in enumerate(manifest_candidates, 1):
            prefix = f"run {run_index} manifest {manifest_index}"
            paths[prefix] = manifest_path
            if not manifest_path.is_file():
                continue
            manifest = _read_run_manifest(manifest_path)
            outputs = manifest.get("outputs")
            if not isinstance(outputs, Mapping):
                continue
            for name, value in outputs.items():
                if isinstance(name, str) and isinstance(value, str) and value.strip():
                    paths[f"{prefix} output {name}"] = _resolve_manifest_output_path(manifest_path, value)
    return paths


def _validate_explicit_artifact_paths(
    args: argparse.Namespace,
    *,
    include_generation: bool,
    input_paths: Mapping[str, Path],
) -> None:
    """Reject user-selected output collisions before opening any evaluation backend."""
    paths: dict[str, Path] = {}
    if args.predictions_output is not None:
        paths["predictions"] = args.predictions_output
    if args.metrics_output is not None:
        paths["metrics"] = args.metrics_output
        paths["metrics manifest"] = args.metrics_output.with_suffix(".manifest.json")
    if include_generation and args.generation_output is not None:
        paths["generation"] = args.generation_output
    run_dir = getattr(args, "run_dir", None)
    if run_dir is not None:
        paths.update(_run_result_paths(args, run_dir))
        paths["metrics manifest"] = paths["metrics"].with_suffix(".manifest.json")
        paths["manifest"] = run_dir / "manifest.json"
        paths["progress"] = run_dir / "predictions.progress.jsonl"
    _validate_distinct_artifact_paths(paths)
    _validate_input_output_artifact_paths(input_paths, paths)


def _validate_distinct_artifact_paths(paths: Mapping[str, Path]) -> None:
    """Require each logical artifact to resolve to a different filesystem path."""
    names_by_path: dict[Path, list[str]] = {}
    for name, path in paths.items():
        names_by_path.setdefault(path.resolve(), []).append(name)
    collisions = [names for names in names_by_path.values() if len(names) > 1]
    if collisions:
        rendered = "; ".join(" = ".join(names) for names in collisions)
        msg = f"Evaluation artifact paths must be distinct: {rendered}"
        raise SystemExit(msg)


def _validate_input_output_artifact_paths(
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
) -> None:
    """Reject an output that resolves to a file consumed by the same command."""
    input_names_by_path: dict[Path, list[str]] = {}
    for name, path in inputs.items():
        input_names_by_path.setdefault(path.resolve(), []).append(name)
    collisions = [
        (output_name, input_names_by_path[output_path.resolve()])
        for output_name, output_path in outputs.items()
        if output_path.resolve() in input_names_by_path
    ]
    if collisions:
        rendered = "; ".join(f"{output_name} = {' = '.join(input_names)}" for output_name, input_names in collisions)
        msg = f"Evaluation outputs must not overwrite inputs: {rendered}"
        raise SystemExit(msg)


def _validate_manifest_output_paths(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    expected_outputs: Mapping[str, Path],
) -> None:
    """Require replay arguments to resolve to the manifest's exact declared outputs."""
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        msg = f"Completed run manifest has no output paths: {manifest_path}"
        raise SystemExit(msg)
    declared_names = {
        name for name, value in outputs.items() if isinstance(name, str) and isinstance(value, str) and value.strip()
    }
    if declared_names != set(expected_outputs):
        msg = f"Completed run manifest outputs do not match the requested outputs: {manifest_path}"
        raise SystemExit(msg)
    for name, expected_path in expected_outputs.items():
        stored_path = outputs.get(name)
        if (
            not isinstance(stored_path, str)
            or _resolve_manifest_output_path(manifest_path, stored_path).resolve() != expected_path.resolve()
        ):
            msg = f"Completed run manifest declares a different {name} output: {manifest_path}"
            raise SystemExit(msg)


def _retrieval_manifest_path(metrics_output: Path) -> Path:
    """Keep each retrieval manifest beside its metrics without colliding with other modes."""
    return metrics_output.with_suffix(".manifest.json")


def _replay_completed_retrieval(
    manifest_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    identity: Mapping[str, Any],
    run_id: str,
) -> bool:
    """Replay an identical complete retrieval artifact and refuse every unsafe overwrite."""
    existing_paths = [path for path in (manifest_path, predictions_path, metrics_path) if path.exists()]
    if not existing_paths:
        return False
    if not manifest_path.is_file():
        paths = ", ".join(str(path) for path in existing_paths)
        msg = f"Refusing to overwrite retrieval outputs without a complete manifest: {paths}"
        raise SystemExit(msg)

    manifest = _read_run_manifest(manifest_path)
    if manifest.get("identity") != identity or manifest.get("run_id") != run_id:
        msg = f"Retrieval outputs belong to a different experiment: {manifest_path}"
        raise SystemExit(msg)
    if manifest.get("status") != "complete":
        msg = f"Refusing to overwrite incomplete retrieval run: {manifest_path}"
        raise SystemExit(msg)
    _validate_manifest_output_paths(
        manifest,
        manifest_path,
        {"predictions": predictions_path, "metrics": metrics_path},
    )

    _write_stdout(_completed_metrics(manifest, manifest_path, metrics_path))
    return True


def _retrieve_with_vector(
    records: Sequence[EvaluationRecord | EvaluationSample], args: argparse.Namespace
) -> _RetrievalExecution:
    """Score the Milvus vector baseline, retrieving on each question as written."""
    from legal_rag.corrective_rag.generate.config import CorrectiveRetrievalConfig
    from legal_rag.corrective_rag.retrieval.database import LegalVectorDB
    from legal_rag.corrective_rag.retrieval.embeddings import EmbeddingModel

    config = CorrectiveRetrievalConfig.from_env()
    embedder = EmbeddingModel(config.embedding_profile)
    db: LegalVectorDB | None = None
    rows: list[dict[str, Any]] = []
    active_error: BaseException | None = None
    try:
        db = LegalVectorDB(config, embedder)
        chunks_dir = Path(os.environ.get("NAIVE_RAG_CHUNKS_DIR", "data/processed/groundtruth"))
        fingerprint = db.preflight_corpus(chunks_dir).to_dict()
        for index, item in enumerate(records, 1):
            record = _as_retrieval_record(item)
            logger.info("Retrieving sample {}/{}", index, len(records))
            started_at = perf_counter()
            hits = db.search(record.sample.question, limit=args.limit)
            rows.append(
                _retrieval_prediction_row(
                    record.sample,
                    hits,
                    raw=record.raw,
                    latency_ms=(perf_counter() - started_at) * 1000,
                )
            )
    except BaseException as error:
        active_error = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        close_callbacks = ([db.close] if db is not None else []) + [embedder.close]
        for close in close_callbacks:
            try:
                close()
            except BaseException as error:
                if active_error is not None:
                    active_error.add_note(f"Retrieval cleanup failed: {error!r}")
                elif cleanup_error is None:
                    cleanup_error = error
                else:
                    cleanup_error.add_note(f"Additional retrieval cleanup failure: {error!r}")
        if active_error is None and cleanup_error is not None:
            raise cleanup_error
    return _RetrievalExecution(rows=rows, active_backend_fingerprint=fingerprint)


def _retrieve_with_naive(
    records: Sequence[EvaluationRecord | EvaluationSample], args: argparse.Namespace
) -> _RetrievalExecution:
    """Score one Naive RAG search mode without query rewriting or answer generation."""
    from legal_rag.datastores.milvus import build_milvus_client
    from legal_rag.embedding.encoder import EmbeddingEncoder
    from legal_rag.embedding.profiles import load_configured_embedding_profile
    from legal_rag.naive_rag.retriever import Retriever, SearchMode
    from legal_rag.settings import MilvusSettings, NaiveRagSettings

    settings = NaiveRagSettings.from_environment()
    profile = load_configured_embedding_profile(settings.embedding_profile)
    encoder = EmbeddingEncoder.from_profile(profile)
    retriever = Retriever(encoder, build_milvus_client(MilvusSettings.from_environment()), settings)
    mode = SearchMode(args.naive_search_mode)
    rows: list[dict[str, Any]] = []
    try:
        fingerprint = retriever.preflight_corpus().to_dict()
        for index, item in enumerate(records, 1):
            record = _as_retrieval_record(item)
            logger.info("Retrieving sample {}/{} with Naive RAG {} search", index, len(records), mode.value)
            started_at = perf_counter()
            hits = retriever.retrieve(record.sample.question, top_k=args.limit, mode=mode)
            retrieved_chunks = [_naive_hit_mapping(hit) for hit in hits]
            rows.append(
                _retrieval_prediction_row(
                    record.sample,
                    retrieved_chunks,
                    raw=record.raw,
                    latency_ms=(perf_counter() - started_at) * 1000,
                )
            )
    finally:
        retriever.close()
    return _RetrievalExecution(rows=rows, active_backend_fingerprint=fingerprint)


def _retrieve_with_graph(
    records: Sequence[EvaluationRecord | EvaluationSample], args: argparse.Namespace
) -> _RetrievalExecution:
    """Score the Graphiti knowledge graph on the same questions and the same chunk ids.

    The graph is searched on each question as written, matching the vector baseline: query
    rewriting is a separate pipeline stage that costs an LLM call, and this command is defined
    as retrieval-only. A graph fact carries the provenance of every episode supporting it, and
    an episode is named by its chunk id, so one fact can contribute several retrieved chunks.
    """
    import asyncio

    from legal_rag.datastores.neo4j import build_neo4j_driver
    from legal_rag.embedding.profiles import load_configured_embedding_profile
    from legal_rag.graphiti_rag.client import build_graphiti
    from legal_rag.graphiti_rag.index_metadata import validate_graphiti_index_metadata
    from legal_rag.graphiti_rag.retriever import search_facts_at_dates
    from legal_rag.graphiti_rag.snapshot import snapshot_graphiti_group
    from legal_rag.settings import GraphitiSettings, Neo4jSettings, Settings
    from legal_rag.temporal import resolve_temporal_targets

    settings = Settings.from_environment()
    graphiti_settings = GraphitiSettings.from_environment()
    profile = load_configured_embedding_profile(graphiti_settings.embedding_profile)
    neo4j_settings = Neo4jSettings.from_environment()
    _, expected_chunks, expected_source_dates = _load_graphiti_evaluation_corpus(args.corpus_config)
    graphiti_client = build_graphiti(settings, neo4j_settings, graphiti_settings, profile)

    async def retrieve_all() -> _RetrievalExecution:
        rows: list[dict[str, Any]] = []
        neo4j_driver = None
        active_error: BaseException | None = None
        try:
            neo4j_driver = build_neo4j_driver(neo4j_settings)
            embedding_metadata = await validate_graphiti_index_metadata(
                neo4j_driver,
                args.group_id,
                profile,
            )
            snapshot = await snapshot_graphiti_group(
                neo4j_driver,
                args.group_id,
                expected_chunks=expected_chunks,
                expected_source_dates=expected_source_dates,
            )
            snapshot.validate_for_evaluation()
            for index, item in enumerate(records, 1):
                record = _as_retrieval_record(item)
                logger.info("Retrieving sample {}/{}", index, len(records))
                started_at = perf_counter()
                as_of_dates = resolve_temporal_targets(
                    record.sample.question,
                    query_dates=record.query_dates,
                    query_date=record.query_date,
                    reference_date=args.reference_date,
                    temporal_comparison=record.temporal_comparison,
                )
                facts = await search_facts_at_dates(
                    graphiti_client,
                    record.sample.question,
                    args.group_id,
                    graphiti_settings.search_num_results,
                    expand_around_top_fact=not args.no_graph_expansion,
                    rerank_with_cross_encoder=args.cross_encoder,
                    as_of_dates=as_of_dates,
                )
                rows.append(
                    _retrieval_prediction_row(
                        record.sample,
                        [_graph_fact_mapping(fact) for fact in facts[: args.limit]],
                        raw=record.raw,
                        latency_ms=(perf_counter() - started_at) * 1000,
                    )
                )
        except BaseException as error:
            active_error = error
            raise
        finally:
            cleanup_error: BaseException | None = None
            close_callbacks = ([neo4j_driver.close] if neo4j_driver is not None else []) + [graphiti_client.close]
            for close in close_callbacks:
                try:
                    await close()
                except BaseException as error:
                    if active_error is not None:
                        active_error.add_note(f"Graphiti retrieval cleanup failed: {error!r}")
                    elif cleanup_error is None:
                        cleanup_error = error
                    else:
                        cleanup_error.add_note(f"Additional Graphiti retrieval cleanup failure: {error!r}")
            if active_error is None and cleanup_error is not None:
                raise cleanup_error
        return _RetrievalExecution(
            rows=rows,
            active_backend_fingerprint={
                **snapshot.to_dict(),
                "embedding_metadata": asdict(embedding_metadata),
            },
        )

    return asyncio.run(retrieve_all())


def _retrieval_prediction_row(
    sample: EvaluationRecord | EvaluationSample,
    retrieved_chunks: list[dict[str, Any]],
    *,
    raw: Mapping[str, Any] | None = None,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    """Merge one ground-truth sample with vector retrieval output."""
    record = _as_retrieval_record(sample)
    row = dict(record.raw if raw is None else raw)
    row.update(asdict(record.sample))
    row["required_claims"] = [
        {
            "claim": expected_claim["claim"],
            "evidence": [{"chunk_id": chunk_id} for chunk_id in expected_claim["chunk_ids"]],
        }
        for expected_claim in row.pop("expected_claims")
    ]
    retrieved_units = [_retrieved_unit_mapping(item) for item in retrieved_chunks]
    evidence = [item for unit in retrieved_units for item in unit["evidence"]]
    row.update(
        {
            "retrieved_chunk_ids": _unique_mapping_field(evidence, "chunk_id"),
            "retrieved_source_names": _unique_mapping_field(evidence, "source_name"),
            "retrieved_chunks": retrieved_chunks,
            "retrieved_units": retrieved_units,
            "evaluation_metadata": {
                "latency_ms": latency_ms,
                "retrieved_unit_count": len(retrieved_units),
                "unique_evidence_chunk_count": len(_unique_mapping_field(evidence, "chunk_id")),
            },
        }
    )
    return row


def _as_retrieval_record(item: EvaluationRecord | EvaluationSample) -> EvaluationRecord:
    """Normalize direct adapter tests while production callers preserve raw dataset metadata."""
    if isinstance(item, EvaluationRecord):
        return item
    return EvaluationRecord(raw=asdict(item), sample=item)


def _naive_hit_mapping(hit: Any) -> dict[str, Any]:
    """Serialize one Naive RAG result without assuming a leaf-only unit."""
    evidence_chunk_ids = getattr(hit, "effective_evidence_chunk_ids", (hit.chunk_id,))
    return {
        "chunk_id": hit.chunk_id,
        "source_name": hit.source_name,
        "text": hit.text,
        "kind": hit.kind,
        "score": hit.score,
        "unit_id": getattr(hit, "effective_unit_id", hit.chunk_id),
        "unit_kind": getattr(hit, "unit_kind", "chunk"),
        "evidence": [{"chunk_id": chunk_id, "source_name": hit.source_name} for chunk_id in evidence_chunk_ids],
    }


def _graph_fact_mapping(fact: Any) -> dict[str, Any]:
    """Serialize one ranked Graphiti fact as one native unit with all chunk provenance."""
    if not fact.sources:
        msg = "Graph evaluation requires chunk provenance on every retrieved fact"
        raise ValueError(msg)
    evidence = [{"chunk_id": source.chunk_id, "source_name": source.source_name} for source in fact.sources]
    evidence_ids = "\n".join(sorted({item["chunk_id"] for item in evidence}))
    unit_id = hashlib.sha256(f"{fact.fact}\n{evidence_ids}".encode()).hexdigest()
    first_source = fact.sources[0]
    return {
        "chunk_id": first_source.chunk_id,
        "source_name": first_source.source_name,
        "text": fact.fact,
        "unit_id": unit_id,
        "unit_kind": "graph_fact",
        "evidence": evidence,
        "as_of_dates": [item.isoformat() for item in getattr(fact, "as_of_dates", ())],
    }


def _retrieved_unit_mapping(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a flat chunk or native unit into the persisted v2 unit shape."""
    chunk_id = str(item["chunk_id"])
    source_name = item.get("source_name")
    raw_evidence = item.get("evidence")
    evidence = raw_evidence if isinstance(raw_evidence, list) else []
    if not evidence:
        evidence = [{"chunk_id": chunk_id, "source_name": source_name}]
    return {
        "unit_id": str(item.get("unit_id") or chunk_id),
        "unit_kind": str(item.get("unit_kind") or "chunk"),
        "text": str(item.get("text") or ""),
        "evidence": evidence,
        "score": item.get("score"),
    }


def _unique_mapping_field(items: list[dict[str, Any]], key: str) -> list[str]:
    """Extract unique non-empty string fields while preserving ranked order."""
    return list(dict.fromkeys(str(item[key]) for item in items if item.get(key)))


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
    return tuple(sorted(set(parsed)))


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    parsed = int(value)
    if parsed < 1:
        msg = f"Expected a positive integer, got {parsed}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _iso_date(value: str) -> date:
    """Parse one canonical calendar date for reproducible temporal evaluation."""
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        msg = f"Expected YYYY-MM-DD, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from error
    if parsed.isoformat() != value:
        msg = f"Expected YYYY-MM-DD, got {value!r}"
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
