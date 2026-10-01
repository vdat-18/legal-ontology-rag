"""Defensible comparison of shared Legal RAG evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from legal_rag.evaluation.generation.core.report import GENERATION_METRIC_NAMES, EvaluationReport
from legal_rag.evaluation.metrics import EvaluationSample

_METRIC_MAP_FIELDS = (
    "hit_rate_at_k",
    "precision_at_k",
    "recall_at_k",
    "claim_evidence_any_at_k",
    "claim_evidence_complete_at_k",
)
_REPORTED_OPERATIONAL_FIELDS = (
    "latency",
    "latency_ms",
    "latency_seconds",
    "usage",
    "token_usage",
    "cost",
    "cost_usd",
    "operational",
)
_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)
_CORE_METRIC_FIELDS = (
    "claim_sample_count",
    "hit_rate",
    "hit_rate_at_k",
    "precision_at_k",
    "recall_at_k",
    "context_precision",
    "context_recall",
    "empty_retrieval_rate",
    "claim_evidence_any_at_k",
    "claim_evidence_complete_at_k",
)
_MATERIAL_IDENTITY_FIELDS = (
    "pipeline",
    "retriever",
    "ground_truth_sha256",
    "max_samples",
    "retrieval_limit",
    "k_values",
    "concurrency",
    "reference_date",
    "history_resolution_enabled",
    "query_rewrite_policy",
    "answer_model",
    "llm_base_url_sha256",
    "embedding_profile",
    "embedding_profile_sha256",
    "llm_extra_request_body_sha256",
    "external_evidence_policy",
    "corpus_chunks_sha256",
    "active_backend_validated",
    "active_backend_fingerprint",
    "effective_config",
)
_GENERATION_IDENTITY_FIELDS = ("judge_model", "judge_batch_size", "judge_concurrency")
_SHARED_SOURCE_CONTROLS = {
    "baml_transport_sha256": "legal_rag.baml_transport",
    "vietnamese_lexical_source_sha256": "legal_rag.benchmark_synthesis.vietnamese_lexical",
    "datastores_source_sha256": "legal_rag.datastores",
    "embedding_source_sha256": "legal_rag.embedding",
    "evaluation_cli_sha256": "legal_rag.evaluation.cli",
    "evaluation_metrics_sha256": "legal_rag.evaluation.metrics",
    "evaluation_runner_sha256": "legal_rag.evaluation.runner",
    "query_rewriting_source_sha256": "legal_rag.query_rewriting",
    "retrieval_source_sha256": "legal_rag.retrieval",
    "settings_sha256": "legal_rag.settings",
    "temporal_source_sha256": "legal_rag.temporal",
}
_GENERATION_SOURCE_CONTROL = "legal_rag.evaluation.generation"
_GENERATION_METRIC_BOUNDS = {
    "faithfulness": (0.0, 1.0),
    "correctness": (1.0, 5.0),
    "answer_relevancy": (1.0, 5.0),
    "temporal_correctness": (0.0, 1.0),
    "answerability_correctness": (0.0, 1.0),
    "required_claim_coverage": (0.0, 1.0),
}


class IncompatibleEvaluationRunsError(ValueError):
    """Raised when evaluation artifacts cannot be compared fairly."""

    def __init__(self, issues: Sequence[str]) -> None:
        """Store every incompatibility so callers can fix them in one pass."""
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"Evaluation artifacts are not comparable:\n{details}")


@dataclass(frozen=True, slots=True)
class _EvaluationArtifact:
    """Validated files and comparison metadata for one evaluation run."""

    metrics_path: Path
    metrics: Mapping[str, Any]
    manifest: Mapping[str, Any] | None
    predictions_path: Path | None
    predictions: tuple[Mapping[str, Any], ...]
    sample_set_sha256: str | None
    ground_truth_sha256: str | None
    warnings: tuple[str, ...]
    contract_issues: tuple[str, ...]

    @property
    def name(self) -> str:
        """Return the pipeline and variant name used in reports and errors."""
        identity = _mapping(self.manifest, "identity")
        pipeline = _first_string(self.metrics, ("pipeline", "retriever")) or _first_string(identity, ("pipeline",))
        mode = _first_string(self.metrics, ("search_mode",)) or _first_string(identity, ("naive_search_mode",))
        if pipeline == "adaptive":
            mode = mode or _first_string(_mapping(identity, "adaptive_config"), ("base_search_mode",))
        if pipeline == "corrective":
            corrective = _mapping(_mapping(identity, "effective_config"), "corrective_rag")
            enabled = corrective.get("web_search_enabled")
            if isinstance(enabled, bool):
                mode = "web_enabled" if enabled else "web_disabled"
        if pipeline == "graphiti":
            ranker = "cross_encoder" if identity.get("cross_encoder") is True else "rank_fusion"
            expansion = "expanded" if identity.get("graph_expansion") is True else "no_expansion"
            mode = f"{ranker}+{expansion}"
        if pipeline and mode:
            return f"{pipeline}:{mode}"
        return pipeline or self.metrics_path.stem

    @property
    def run_id(self) -> str | None:
        """Return the stable run ID when one was persisted."""
        return _first_string(self.metrics, ("run_id",)) or _first_string(self.manifest, ("run_id",))


@dataclass(frozen=True, slots=True)
class _UsageObservation:
    """Provider usage reported for one evaluated sample."""

    call_count: int
    available_call_count: int
    unavailable_call_count: int
    partial: bool
    models: Mapping[str, int]
    tokens: Mapping[str, list[float]]


def compare_evaluation_runs(
    metrics_paths: Sequence[Path],
    *,
    allow_incompatible: bool = False,
) -> dict[str, Any]:
    """Compare metrics files after validating their evaluation populations and contracts.

    Args:
        metrics_paths: Two or more shared ``metrics.json`` artifacts.
        allow_incompatible: Return a report with incompatibilities instead of raising.

    Returns:
        A deterministic JSON-compatible comparison without a synthetic composite score.

    Raises:
        IncompatibleEvaluationRunsError: If the runs differ on a required comparison dimension.
        ValueError: If an artifact is malformed or fewer than two distinct paths are supplied.
    """
    normalized_paths = tuple(Path(path).resolve() for path in metrics_paths)
    if len(normalized_paths) < 2:
        msg = "At least two metrics files are required for comparison"
        raise ValueError(msg)
    if len(set(normalized_paths)) != len(normalized_paths):
        msg = "Each metrics file may be supplied only once"
        raise ValueError(msg)

    artifacts = tuple(_load_artifact(path) for path in normalized_paths)
    issues, warnings, basis = _compatibility(artifacts)
    if issues and not allow_incompatible:
        raise IncompatibleEvaluationRunsError(issues)

    ordered = sorted(artifacts, key=lambda item: (item.name, item.run_id or "", item.metrics_path.as_posix()))
    generation_available = [isinstance(item.metrics.get("generation"), Mapping) for item in ordered]
    if any(generation_available) and not all(generation_available):
        warnings.append("Generation metrics are unavailable for some runs and cannot be compared across every run")

    return {
        "schema_version": "1.0.0",
        "compatibility": {
            "status": "incompatible" if issues else "compatible",
            "issues": sorted(set(issues)),
            "warnings": sorted(set(warnings)),
            "basis": basis,
            "generation_metrics_comparable": _generation_metrics_comparable(ordered, issues),
        },
        "runs": [_run_report(artifact) for artifact in ordered],
    }


def _load_artifact(metrics_path: Path) -> _EvaluationArtifact:
    """Load one metrics file plus its adjacent manifest and predictions when available."""
    metrics = _read_json_object(metrics_path, "metrics")
    _validate_metrics(metrics_path, metrics)

    manifest_path = _find_manifest_path(metrics_path)
    manifest = _read_json_object(manifest_path, "manifest") if manifest_path is not None else None
    contract_issues = _metrics_contract_issues(metrics_path, metrics)
    if manifest is None:
        contract_issues.append(f"{metrics_path}: a complete evaluation manifest is required for ranked comparison")
    else:
        _validate_manifest(metrics_path, metrics, manifest_path, manifest)
        contract_issues.extend(_manifest_contract_issues(metrics_path, metrics, manifest))
    predictions_path = _find_predictions_path(metrics_path, manifest_path, manifest)
    predictions = _read_jsonl_objects(predictions_path) if predictions_path is not None else ()
    warnings: list[str] = []
    sample_set_sha256 = _sample_set_sha256(predictions) if predictions else None
    if predictions_path is None:
        warnings.append(f"{metrics_path}: predictions file was not found")

    identity = _mapping(manifest, "identity")
    ground_truth_sha256 = _first_string(identity, ("ground_truth_sha256",)) or _first_string(
        metrics, ("ground_truth_sha256",)
    )
    if predictions and len(predictions) != metrics["sample_count"]:
        msg = (
            f"{metrics_path}: sample_count is {metrics['sample_count']} but "
            f"{predictions_path} contains {len(predictions)} rows"
        )
        raise ValueError(msg)
    _validate_generation_artifact(metrics_path, metrics, manifest_path, manifest, predictions)

    return _EvaluationArtifact(
        metrics_path=metrics_path,
        metrics=metrics,
        manifest=manifest,
        predictions_path=predictions_path,
        predictions=predictions,
        sample_set_sha256=sample_set_sha256,
        ground_truth_sha256=ground_truth_sha256,
        warnings=tuple(warnings),
        contract_issues=tuple(contract_issues),
    )


def _validate_metrics(path: Path, metrics: Mapping[str, Any]) -> None:
    """Validate comparison-critical fields at the artifact boundary."""
    if metrics.get("schema_version") == "2.0.0":
        missing_fields = [field for field in _CORE_METRIC_FIELDS if field not in metrics]
        if missing_fields:
            msg = f"{path}: schema 2.0.0 metrics are missing required fields: {', '.join(missing_fields)}"
            raise ValueError(msg)
    _validate_metric_counts(path, metrics)
    _validate_generation_metrics(path, metrics)
    k_values = _validate_k_values(path, metrics.get("k_values"))
    _validate_k_metric_maps(path, metrics, k_values)
    _validate_retrieval_limit(path, metrics.get("retrieval_limit"))
    concurrency = metrics.get("concurrency")
    if concurrency is not None and (
        not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0
    ):
        msg = f"{path}: concurrency must be a positive integer when present"
        raise ValueError(msg)


def _metrics_contract_issues(path: Path, metrics: Mapping[str, Any]) -> list[str]:
    """Return ranking-contract gaps that remain useful in a diagnostic report."""
    issues: list[str] = []
    if metrics.get("schema_version") != "2.0.0":
        issues.append(f"{path}: ranked comparison requires metrics schema_version 2.0.0")
    if metrics.get("active_backend_validated") is not True:
        issues.append(f"{path}: active_backend_validated must be true for ranked comparison")
    fingerprint = metrics.get("active_backend_fingerprint")
    if not isinstance(fingerprint, Mapping) or not fingerprint:
        issues.append(f"{path}: active_backend_fingerprint must be a non-empty object for ranked comparison")
    return issues


def _validate_generation_metrics(path: Path, metrics: Mapping[str, Any]) -> None:
    """Validate the fixed six-metric answer-quality contract and its denominators."""
    has_generation = "generation" in metrics
    has_coverage = "generation_metric_coverage" in metrics
    if not has_generation and not has_coverage:
        return
    if has_generation != has_coverage:
        msg = f"{path}: generation and generation_metric_coverage must be provided together"
        raise ValueError(msg)

    generation = metrics.get("generation")
    coverage = metrics.get("generation_metric_coverage")
    if not isinstance(generation, Mapping) or not isinstance(coverage, Mapping):
        msg = f"{path}: generation and generation_metric_coverage must be objects"
        raise TypeError(msg)
    _validate_generation_metric_names(path, "generation", generation)
    _validate_generation_metric_names(path, "generation_metric_coverage", coverage)

    sample_count = metrics["sample_count"]
    for name in GENERATION_METRIC_NAMES:
        _validate_generation_metric(path, name, generation[name], coverage[name], sample_count)


def _validate_generation_metric(
    path: Path,
    name: str,
    value: Any,
    metric_coverage: Any,
    sample_count: int,
) -> None:
    """Validate one aggregate score and the coverage from which it was calculated."""
    number = _number(value)
    lower, upper = _GENERATION_METRIC_BOUNDS[name]
    if value is not None and (number is None or not math.isfinite(number) or not lower <= number <= upper):
        msg = f"{path}: generation metric {name} must be null or between {lower:g} and {upper:g}"
        raise ValueError(msg)
    if not isinstance(metric_coverage, Mapping):
        msg = f"{path}: generation coverage for {name} must be an object"
        raise TypeError(msg)
    scored_count = _integer(metric_coverage.get("scored_count"))
    null_count = _integer(metric_coverage.get("null_count"))
    ratio = _number(metric_coverage.get("coverage"))
    if scored_count is None or null_count is None or ratio is None or not math.isfinite(ratio):
        msg = f"{path}: generation coverage for {name} has invalid counters"
        raise ValueError(msg)
    if scored_count + null_count != sample_count:
        msg = f"{path}: generation coverage counts for {name} must sum to sample_count"
        raise ValueError(msg)
    expected_ratio = scored_count / sample_count if sample_count else 0.0
    if not 0.0 <= ratio <= 1.0 or not math.isclose(ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-12):
        msg = f"{path}: generation coverage ratio for {name} does not match its counters"
        raise ValueError(msg)
    if (value is None) != (scored_count == 0):
        msg = f"{path}: generation aggregate for {name} does not match its scored coverage"
        raise ValueError(msg)


def _validate_generation_metric_names(path: Path, field: str, value: Mapping[str, Any]) -> None:
    """Require exactly the versioned generation metric names."""
    expected = set(GENERATION_METRIC_NAMES)
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(str(name) for name in actual - expected)
    details = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected {', '.join(unexpected)}")
    msg = f"{path}: {field} must contain exactly six generation metrics ({'; '.join(details)})"
    raise ValueError(msg)


def _validate_generation_artifact(
    metrics_path: Path,
    metrics: Mapping[str, Any],
    manifest_path: Path | None,
    manifest: Mapping[str, Any] | None,
    predictions: Sequence[Mapping[str, Any]],
) -> None:
    """Reconcile a declared generation report with metrics and prediction identities."""
    if not isinstance(metrics.get("generation"), Mapping) or manifest_path is None:
        return
    generation_output = _mapping(manifest, "outputs").get("generation")
    if not isinstance(generation_output, str) or not generation_output.strip():
        return

    generation_path = _resolve_output_path(manifest_path, generation_output)
    report = _read_json_object(generation_path, "generation report")
    sample_count = _integer(report.get("sample_count"))
    if sample_count != metrics["sample_count"] or sample_count != len(predictions):
        msg = f"{generation_path}: sample_count must match metrics and predictions ({metrics['sample_count']} samples)"
        raise ValueError(msg)
    per_sample = _generation_per_sample(
        metrics_path,
        generation_path,
        report,
        predictions,
        sample_count=sample_count,
    )
    _validate_generation_per_sample(generation_path, per_sample)
    recomputed = EvaluationReport.from_per_sample(per_sample)
    _validate_generation_report_maps(generation_path, report, metrics, recomputed)


def _generation_per_sample(
    metrics_path: Path,
    generation_path: Path,
    report: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    sample_count: int,
) -> tuple[Mapping[str, Any], ...]:
    """Validate generation detail shape and bind each row to its prediction identity."""
    value = report.get("per_sample")
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        msg = f"{generation_path}: per_sample must be an array of score objects"
        raise TypeError(msg)
    per_sample = tuple(value)
    if len(per_sample) != sample_count:
        msg = f"{generation_path}: per_sample must contain exactly sample_count rows"
        raise ValueError(msg)

    expected_ids = tuple(_generation_sample_id(row, index) for index, row in enumerate(predictions))
    if len(set(expected_ids)) != len(expected_ids):
        msg = f"{metrics_path}: prediction sample identities must be unique for generation validation"
        raise ValueError(msg)
    actual_ids = tuple(_first_string(row, ("sample_id",)) for row in per_sample)
    if any(sample_id is None for sample_id in actual_ids):
        msg = f"{generation_path}: every per_sample row must contain a non-empty string sample_id"
        raise ValueError(msg)
    if len(set(actual_ids)) != len(actual_ids):
        msg = f"{generation_path}: per_sample sample_id values must be unique"
        raise ValueError(msg)
    if actual_ids != expected_ids:
        msg = f"{generation_path}: per_sample sample_id values do not match predictions in order"
        raise ValueError(msg)
    return per_sample


def _validate_generation_report_maps(
    generation_path: Path,
    report: Mapping[str, Any],
    metrics: Mapping[str, Any],
    recomputed: EvaluationReport,
) -> None:
    """Recompute generation aggregate and coverage, then bind them to metrics."""
    for report_field, expected in (
        ("aggregate", recomputed.aggregate),
        ("metric_coverage", recomputed.metric_coverage),
    ):
        value = report.get(report_field)
        if not isinstance(value, Mapping):
            msg = f"{generation_path}: {report_field} must be an object"
            raise TypeError(msg)
        if _canonical_json(value) != _canonical_json(expected):
            msg = f"{generation_path}: {report_field} does not match per_sample scores"
            raise ValueError(msg)
    if _canonical_json(report["aggregate"]) != _canonical_json(metrics["generation"]):
        msg = f"{generation_path}: aggregate does not match metrics generation"
        raise ValueError(msg)
    if _canonical_json(report["metric_coverage"]) != _canonical_json(metrics["generation_metric_coverage"]):
        msg = f"{generation_path}: metric_coverage does not match metrics generation_metric_coverage"
        raise ValueError(msg)


def _validate_generation_per_sample(path: Path, per_sample: Sequence[Mapping[str, Any]]) -> None:
    """Validate fixed metric names and score bounds in every generation detail row."""
    expected_fields = {"sample_id", *GENERATION_METRIC_NAMES}
    for index, row in enumerate(per_sample, 1):
        if set(row) != expected_fields:
            missing = sorted(expected_fields - set(row))
            unexpected = sorted(str(name) for name in set(row) - expected_fields)
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected {', '.join(unexpected)}")
            msg = f"{path}: per_sample row {index} has invalid fields ({'; '.join(details)})"
            raise ValueError(msg)
        for name in GENERATION_METRIC_NAMES:
            value = row[name]
            number = _number(value)
            lower, upper = _GENERATION_METRIC_BOUNDS[name]
            if value is not None and (number is None or not math.isfinite(number) or not lower <= number <= upper):
                msg = f"{path}: per_sample row {index} metric {name} must be null or between {lower:g} and {upper:g}"
                raise ValueError(msg)


def _generation_sample_id(row: Mapping[str, Any], index: int) -> str:
    """Mirror generation dataset identity precedence for one prediction row."""
    return _first_string(row, ("evaluation_record_id", "question_id", "sample_id")) or str(index)


def _validate_metric_counts(path: Path, metrics: Mapping[str, Any]) -> None:
    """Validate aggregate denominator fields."""
    for field in ("sample_count", "retrieval_sample_count", "claim_sample_count"):
        value = metrics.get(field)
        if field == "claim_sample_count" and value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            msg = f"{path}: {field} must be a non-negative integer"
            raise ValueError(msg)


def _validate_k_values(path: Path, value: Any) -> list[int]:
    """Validate and return the ordered retrieval cutoffs."""
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value
    ):
        msg = f"{path}: k_values must be a list of positive integers"
        raise ValueError(msg)
    if value != sorted(set(value)):
        msg = f"{path}: k_values must be unique and sorted"
        raise ValueError(msg)
    return value


def _validate_k_metric_maps(path: Path, metrics: Mapping[str, Any], k_values: Sequence[int]) -> None:
    """Ensure every available @k map uses the declared cutoffs."""
    expected_k_keys = {str(value) for value in k_values}
    for field in _METRIC_MAP_FIELDS:
        value = metrics.get(field)
        if value is None:
            continue
        if not isinstance(value, Mapping) or set(value) != expected_k_keys:
            msg = f"{path}: {field} keys must exactly match k_values"
            raise ValueError(msg)
        if any(item is not None and _number(item) is None for item in value.values()):
            msg = f"{path}: {field} values must be numeric or null"
            raise ValueError(msg)


def _validate_retrieval_limit(path: Path, value: Any) -> None:
    """Validate the native retrieval-unit budget when it is available."""
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        msg = f"{path}: retrieval_limit must be a positive integer when present"
        raise ValueError(msg)


def _validate_manifest(
    metrics_path: Path,
    metrics: Mapping[str, Any],
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Reject internally inconsistent run artifacts before cross-run comparison."""
    manifest_run_id = _first_string(manifest, ("run_id",))
    metrics_run_id = _first_string(metrics, ("run_id",))
    if manifest_run_id and metrics_run_id and manifest_run_id != metrics_run_id:
        msg = f"{metrics_path}: metrics run_id does not match manifest run_id"
        raise ValueError(msg)

    identity = _mapping(manifest, "identity")
    for field in _MATERIAL_IDENTITY_FIELDS:
        if field in identity and field in metrics and identity[field] != metrics[field]:
            msg = f"{metrics_path}: metrics {field} does not match manifest identity"
            raise ValueError(msg)
    if (
        "search_mode" in metrics
        and "naive_search_mode" in identity
        and metrics["search_mode"] != identity["naive_search_mode"]
    ):
        msg = f"{metrics_path}: metrics search_mode does not match manifest identity"
        raise ValueError(msg)
    if isinstance(metrics.get("generation"), Mapping):
        for field in _GENERATION_IDENTITY_FIELDS:
            if field in identity and field in metrics and identity[field] != metrics[field]:
                msg = f"{metrics_path}: metrics {field} does not match manifest identity"
                raise ValueError(msg)
        prompt_sha256 = _mapping(identity, "prompt_sha256").get("answer_evaluation.baml")
        if prompt_sha256 is not None and prompt_sha256 != metrics.get("judge_prompt_sha256"):
            msg = f"{metrics_path}: metrics judge_prompt_sha256 does not match manifest identity"
            raise ValueError(msg)
    _validate_manifest_output_digests(metrics_path, manifest_path, manifest)


def _manifest_contract_issues(
    metrics_path: Path,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[str]:
    """Return incomplete-manifest findings without blocking diagnostic comparison."""
    issues: list[str] = []
    if manifest.get("schema_version") != "1.0.0":
        issues.append(f"{metrics_path}: manifest schema_version must be 1.0.0 for ranked comparison")
    if manifest.get("status") != "complete":
        issues.append(f"{metrics_path}: manifest status must be 'complete' for ranked comparison")
    if _first_string(manifest, ("run_id",)) is None:
        issues.append(f"{metrics_path}: manifest run_id is required for ranked comparison")
    issues.extend(_manifest_identity_contract_issues(metrics_path, metrics, manifest))
    issues.extend(_manifest_output_contract_issues(metrics_path, metrics, manifest))
    return issues


def _manifest_identity_contract_issues(
    metrics_path: Path,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[str]:
    """Check that the manifest attests every material control reported by metrics."""
    issues: list[str] = []

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping) or not identity:
        issues.append(f"{metrics_path}: manifest identity must be a non-empty object for ranked comparison")
        identity = {}
    issues.extend(
        f"{metrics_path}: manifest identity is missing metrics control {field}"
        for field in _MATERIAL_IDENTITY_FIELDS
        if field in metrics and field not in identity
    )
    if "search_mode" in metrics and "naive_search_mode" not in identity:
        issues.append(f"{metrics_path}: manifest identity is missing metrics control search_mode")
    if "dependency_lock_sha256" not in identity:
        issues.append(f"{metrics_path}: manifest identity is missing dependency_lock_sha256")
    pipeline = _first_string(metrics, ("pipeline", "retriever")) or _first_string(identity, ("pipeline",))
    if pipeline == "ontology":
        ontology_digests = identity.get("ontology_sha256")
        if (
            not isinstance(ontology_digests, Sequence)
            or isinstance(ontology_digests, str)
            or not ontology_digests
            or any(not _valid_sha256(item) for item in ontology_digests)
        ):
            issues.append(f"{metrics_path}: ontology runs require ontology_sha256 in manifest identity")
    source_sha256 = _mapping(identity, "source_sha256")
    issues.extend(
        f"{metrics_path}: manifest identity is missing shared source digest {module_name}"
        for module_name in _SHARED_SOURCE_CONTROLS.values()
        if module_name not in source_sha256
    )
    has_generation = isinstance(metrics.get("generation"), Mapping)
    if "generation" in identity and identity["generation"] is not has_generation:
        issues.append(f"{metrics_path}: manifest generation flag does not match generation metrics")
    if has_generation:
        issues.extend(
            f"{metrics_path}: manifest identity is missing metrics control {field}"
            for field in _GENERATION_IDENTITY_FIELDS
            if field not in identity
        )
        prompt_sha256 = _mapping(identity, "prompt_sha256").get("answer_evaluation.baml")
        if prompt_sha256 is None:
            issues.append(f"{metrics_path}: manifest identity is missing metrics control judge_prompt_sha256")
        if _GENERATION_SOURCE_CONTROL not in source_sha256:
            issues.append(
                f"{metrics_path}: manifest identity is missing generation source digest {_GENERATION_SOURCE_CONTROL}"
            )
    issues.extend(
        _runtime_identity_contract_issues(
            metrics_path,
            identity,
            pipeline=pipeline,
            has_generation=has_generation,
        )
    )
    return issues


def _runtime_identity_contract_issues(
    metrics_path: Path,
    identity: Mapping[str, Any],
    *,
    pipeline: str | None,
    has_generation: bool,
) -> list[str]:
    """Require resolved configuration and the exact prompt inventory for ranked runs."""
    issues: list[str] = []
    effective_config = identity.get("effective_config")
    if not isinstance(effective_config, Mapping) or not effective_config:
        issues.append(f"{metrics_path}: manifest identity requires a non-empty effective_config")
    if identity.get("command") == "retrieve" or pipeline is None:
        return issues
    prompt_sha256 = identity.get("prompt_sha256")
    expected_prompts = _expected_prompt_names(pipeline, identity, has_generation=has_generation)
    if not isinstance(prompt_sha256, Mapping) or set(prompt_sha256) != expected_prompts:
        issues.append(
            f"{metrics_path}: manifest prompt_sha256 must contain exactly the prompts used by pipeline "
            f"{pipeline!r}: {', '.join(sorted(expected_prompts))}"
        )
    return issues


def _expected_prompt_names(
    pipeline: str,
    identity: Mapping[str, Any],
    *,
    has_generation: bool,
) -> set[str]:
    """Return the BAML prompt sources used by one persisted end-to-end identity."""
    prompts_by_pipeline = {
        "naive": {"naive_rag_generation.baml"},
        "adaptive": {"adaptive_query_planning.baml", "naive_rag_generation.baml"},
        "corrective": {"corrective_rag_grading.baml", "query_rewrite.baml"},
        "graphiti": {"naive_rag_generation.baml", "query_rewrite.baml"},
        "ontology": {"naive_rag_generation.baml", "ontology_query_planning.baml"},
    }
    prompts = set(prompts_by_pipeline.get(pipeline, ()))
    rewrite_policy = identity.get("query_rewrite_policy")
    if pipeline in {"naive", "adaptive", "ontology"} and rewrite_policy in {
        "enabled_when_history_present",
        "ontology_planner_always",
    }:
        prompts.add("query_rewrite.baml")
    if has_generation:
        prompts.add("answer_evaluation.baml")
    return prompts


def _manifest_output_contract_issues(
    metrics_path: Path,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[str]:
    """Check that every ranked output is declared and protected by a digest."""
    issues: list[str] = []
    has_generation = isinstance(metrics.get("generation"), Mapping)

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        issues.append(f"{metrics_path}: manifest outputs must be an object for ranked comparison")
        outputs = {}
    required_outputs = {"predictions", "metrics"}
    if has_generation:
        required_outputs.add("generation")
    for name in sorted(required_outputs):
        value = outputs.get(name)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"{metrics_path}: manifest output {name} is required for ranked comparison")

    output_sha256 = manifest.get("output_sha256")
    if not isinstance(output_sha256, Mapping):
        issues.append(f"{metrics_path}: manifest output_sha256 must be an object for ranked comparison")
    else:
        expected_digest_names = {
            name
            for name, value in outputs.items()
            if isinstance(name, str) and isinstance(value, str) and value.strip()
        }
        if set(output_sha256) != expected_digest_names:
            issues.append(f"{metrics_path}: manifest output_sha256 keys must exactly match declared outputs")
    return issues


def _validate_manifest_output_digests(
    metrics_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Verify every declared output digest and bind the manifest to the supplied metrics file."""
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        return
    declared_metrics = outputs.get("metrics")
    if isinstance(declared_metrics, str) and declared_metrics.strip():
        resolved_metrics = _resolve_output_path(manifest_path, declared_metrics)
        if resolved_metrics.resolve() != metrics_path.resolve():
            msg = f"{metrics_path}: manifest references a different metrics output: {resolved_metrics}"
            raise ValueError(msg)

    output_sha256 = manifest.get("output_sha256")
    if not isinstance(output_sha256, Mapping):
        return
    for name, expected_digest in output_sha256.items():
        output_value = outputs.get(name)
        if not isinstance(name, str) or not isinstance(output_value, str) or not output_value.strip():
            msg = f"{metrics_path}: manifest SHA-256 entry {name!r} has no declared output"
            raise ValueError(msg)
        if not _valid_sha256(expected_digest):
            msg = f"{metrics_path}: manifest output_sha256[{name!r}] is not a SHA-256 digest"
            raise ValueError(msg)
        output_path = _resolve_output_path(manifest_path, output_value)
        if not output_path.is_file():
            msg = f"{metrics_path}: manifest output {name!r} was not found: {output_path}"
            raise ValueError(msg)
        if _sha256_file(output_path) != expected_digest:
            msg = f"{metrics_path}: manifest output {name!r} does not match its SHA-256 digest"
            raise ValueError(msg)


def _compatibility(
    artifacts: Sequence[_EvaluationArtifact],
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Compare the dimensions that determine whether aggregate metrics share meaning."""
    issues = [issue for artifact in artifacts for issue in artifact.contract_issues]
    warnings = [warning for artifact in artifacts for warning in artifact.warnings]

    _compare_field(artifacts, "metrics schema_version", lambda item: item.metrics.get("schema_version"), issues)
    _compare_field(artifacts, "sample_count", lambda item: item.metrics["sample_count"], issues)
    _compare_field(artifacts, "retrieval_sample_count", lambda item: item.metrics["retrieval_sample_count"], issues)
    _compare_field(artifacts, "claim_sample_count", lambda item: item.metrics.get("claim_sample_count"), issues)
    _compare_field(artifacts, "k_values", lambda item: tuple(item.metrics["k_values"]), issues)
    _compare_field(artifacts, "retrieval_limit", lambda item: item.metrics.get("retrieval_limit"), issues)

    concurrency_values = [_artifact_concurrency(artifact) for artifact in artifacts]
    known_concurrency_values = [value for value in concurrency_values if value is not None]
    if len(known_concurrency_values) == len(artifacts):
        _compare_field(artifacts, "concurrency", _artifact_concurrency, issues)
    else:
        warnings.append("concurrency is unavailable for one or more runs; latency comparability cannot be verified")

    control_values = _validate_shared_controls(artifacts, issues, warnings)
    _validate_same_pipeline_fingerprints(artifacts, issues)

    schema_versions = [artifact.metrics.get("schema_version") for artifact in artifacts]
    if any(value is None for value in schema_versions):
        issues.append("metrics schema_version is missing from one or more runs")
    retrieval_limits = [artifact.metrics.get("retrieval_limit") for artifact in artifacts]
    if any(value is None for value in retrieval_limits):
        issues.append("retrieval_limit is missing from one or more runs")

    sample_digests = [artifact.sample_set_sha256 for artifact in artifacts]
    if any(value is None for value in sample_digests):
        issues.append(
            "the evaluated sample set cannot be verified because one or more prediction files are unavailable"
        )
    elif len(set(sample_digests)) != 1:
        issues.append("prediction files contain different evaluated sample sets")

    ground_truth_digests = [artifact.ground_truth_sha256 for artifact in artifacts]
    known_ground_truth_digests = {value for value in ground_truth_digests if value is not None}
    if len(known_ground_truth_digests) > 1:
        issues.append("manifests identify different ground-truth datasets")
    elif any(value is None for value in ground_truth_digests):
        warnings.append(
            "ground-truth digest is unavailable for one or more runs; sample content fingerprints were used"
        )

    basis = {
        "metrics_schema_version": _common_or_none(schema_versions),
        "ground_truth_sha256": _common_or_none(ground_truth_digests),
        "sample_set_sha256": _common_or_none(sample_digests),
        "sample_count": _common_or_none([artifact.metrics["sample_count"] for artifact in artifacts]),
        "retrieval_sample_count": _common_or_none(
            [artifact.metrics["retrieval_sample_count"] for artifact in artifacts]
        ),
        "claim_sample_count": _common_or_none([artifact.metrics.get("claim_sample_count") for artifact in artifacts]),
        "k_values": _common_or_none([list(artifact.metrics["k_values"]) for artifact in artifacts]),
        "retrieval_limit": _common_or_none(retrieval_limits),
        "concurrency": _common_or_none(concurrency_values),
        "controls": {field: _common_or_none(values) for field, values in control_values.items()},
    }
    return issues, warnings, basis


def _validate_shared_controls(
    artifacts: Sequence[_EvaluationArtifact],
    issues: list[str],
    warnings: list[str],
) -> dict[str, list[Any]]:
    """Compare shared model, corpus, history, temporal, and judge controls."""
    run_fields = (
        "active_backend_validated",
        "reference_date",
        "history_resolution_enabled",
        "answer_model",
        "llm_base_url_sha256",
        "embedding_profile",
        "embedding_profile_sha256",
        "llm_extra_request_body_sha256",
        "external_evidence_policy",
        "corpus_chunks_sha256",
        "dependency_lock_sha256",
        *_SHARED_SOURCE_CONTROLS,
    )
    judge_fields = (
        "judge_model",
        "judge_batch_size",
        "judge_concurrency",
        "judge_prompt_sha256",
        "generation_source_sha256",
    )
    recorded_fields = (*run_fields, "query_rewrite_policy", *judge_fields)
    values_by_field = {
        field: [_artifact_control(artifact, field) for artifact in artifacts] for field in recorded_fields
    }
    fairness_critical = {"history_resolution_enabled"}
    for field in run_fields:
        values = values_by_field[field]
        known_values = [value for value in values if value is not None]
        if len(known_values) == len(artifacts):
            _compare_field(artifacts, field, lambda item, name=field: _artifact_control(item, name), issues)
        elif known_values:
            message = f"{field} is unavailable for one or more runs"
            (issues if field in fairness_critical else warnings).append(message)
    if "live_web" in values_by_field["external_evidence_policy"]:
        issues.append("live_web evidence is not a frozen corpus snapshot and cannot be ranked as a compatible run")

    generation_available = [isinstance(artifact.metrics.get("generation"), Mapping) for artifact in artifacts]
    if all(generation_available):
        for field in judge_fields:
            values = values_by_field[field]
            if all(value is not None for value in values):
                _compare_field(artifacts, field, lambda item, name=field: _artifact_control(item, name), issues)
            else:
                warnings.append(f"{field} is unavailable for one or more generation runs")
    return values_by_field


def _validate_same_pipeline_fingerprints(
    artifacts: Sequence[_EvaluationArtifact],
    issues: list[str],
) -> None:
    """Require repeated variants of one backend to use the same active datastore state."""
    artifacts_by_pipeline: dict[str, list[_EvaluationArtifact]] = {}
    for artifact in artifacts:
        identity = _mapping(artifact.manifest, "identity")
        pipeline = _first_string(artifact.metrics, ("pipeline", "retriever")) or _first_string(identity, ("pipeline",))
        if pipeline is not None:
            artifacts_by_pipeline.setdefault(pipeline, []).append(artifact)

    for pipeline, pipeline_artifacts in artifacts_by_pipeline.items():
        if len(pipeline_artifacts) < 2:
            continue
        _compare_field(
            pipeline_artifacts,
            f"active_backend_fingerprint within pipeline {pipeline!r}",
            lambda item: _artifact_control(item, "active_backend_fingerprint"),
            issues,
        )
        _compare_field(
            pipeline_artifacts,
            f"query_rewrite_policy within pipeline {pipeline!r}",
            lambda item: _artifact_control(item, "query_rewrite_policy"),
            issues,
        )
        _compare_field(
            pipeline_artifacts,
            f"effective_config within pipeline {pipeline!r}",
            lambda item, name=pipeline: _same_pipeline_effective_config(item, name),
            issues,
        )
        _compare_field(
            pipeline_artifacts,
            f"prompt_sha256 within pipeline {pipeline!r}",
            _pipeline_prompt_hashes,
            issues,
        )
        if pipeline == "ontology":
            _compare_field(
                pipeline_artifacts,
                "ontology_sha256 within pipeline 'ontology'",
                lambda item: _artifact_control(item, "ontology_sha256"),
                issues,
            )


def _same_pipeline_effective_config(artifact: _EvaluationArtifact, pipeline: str) -> Mapping[str, Any]:
    """Remove only CLI-selected ablation knobs before comparing one pipeline's runtime config."""
    config = _canonical_value(_mapping(_mapping(artifact.manifest, "identity"), "effective_config"))
    if not isinstance(config, dict):
        return {}
    if pipeline == "naive":
        naive_config = config.get("naive_rag")
        if isinstance(naive_config, dict):
            naive_config.pop("search_mode", None)
    elif pipeline == "graphiti":
        graphiti_config = config.get("graphiti")
        if isinstance(graphiti_config, dict):
            graphiti_config.pop("expand_around_top_fact", None)
            graphiti_config.pop("rerank_with_cross_encoder", None)
    return config


def _pipeline_prompt_hashes(artifact: _EvaluationArtifact) -> Mapping[str, Any]:
    """Return generation/planning prompt hashes while leaving judge controls to their own contract."""
    prompts = dict(_mapping(_mapping(artifact.manifest, "identity"), "prompt_sha256"))
    prompts.pop("answer_evaluation.baml", None)
    return prompts


def _generation_metrics_comparable(
    artifacts: Sequence[_EvaluationArtifact],
    compatibility_issues: Sequence[str],
) -> bool:
    """Return whether answer scores share the full run and judge contract."""
    if compatibility_issues or not all(_has_complete_generation_contract(artifact.metrics) for artifact in artifacts):
        return False
    judge_fields = (
        "judge_model",
        "judge_batch_size",
        "judge_concurrency",
        "judge_prompt_sha256",
        "generation_source_sha256",
    )
    for field in judge_fields:
        values = [_artifact_control(artifact, field) for artifact in artifacts]
        if any(value is None for value in values) or len({_canonical_json(value) for value in values}) != 1:
            return False
    return True


def _has_complete_generation_contract(metrics: Mapping[str, Any]) -> bool:
    """Return whether both aggregate and coverage maps expose the six fixed metrics."""
    expected = set(GENERATION_METRIC_NAMES)
    generation = metrics.get("generation")
    coverage = metrics.get("generation_metric_coverage")
    return (
        isinstance(generation, Mapping)
        and set(generation) == expected
        and isinstance(coverage, Mapping)
        and set(coverage) == expected
    )


def _compare_field(
    artifacts: Sequence[_EvaluationArtifact],
    field: str,
    read: Any,
    issues: list[str],
) -> None:
    """Append a readable incompatibility when values differ between runs."""
    values = [(artifact.name, read(artifact)) for artifact in artifacts]
    canonical_values = {_canonical_json(value) for _, value in values}
    if len(canonical_values) > 1:
        rendered = ", ".join(f"{name}={value!r}" for name, value in values)
        issues.append(f"{field} differs: {rendered}")


def _artifact_concurrency(artifact: _EvaluationArtifact) -> int | None:
    """Read the measured pipeline concurrency from metrics or the run manifest."""
    value = artifact.metrics.get("concurrency")
    if value is None:
        value = _mapping(artifact.manifest, "identity").get("concurrency")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _artifact_control(artifact: _EvaluationArtifact, field: str) -> Any:
    """Read one shared experiment control from metrics or the persisted identity."""
    value = artifact.metrics.get(field)
    if value is not None:
        return value
    identity = _mapping(artifact.manifest, "identity")
    source_module = _SHARED_SOURCE_CONTROLS.get(field)
    if source_module is not None:
        return _mapping(identity, "source_sha256").get(source_module)
    if field == "generation_source_sha256":
        return _mapping(identity, "source_sha256").get(_GENERATION_SOURCE_CONTROL)
    value = identity.get(field)
    if value is not None:
        return value
    if field == "judge_prompt_sha256":
        return _mapping(identity, "prompt_sha256").get("answer_evaluation.baml")
    return None


def _run_report(artifact: _EvaluationArtifact) -> dict[str, Any]:
    """Build one comparison row with metrics grouped by their actual semantics."""
    metrics = artifact.metrics
    reported_operational = {
        field: _canonical_value(metrics[field]) for field in _REPORTED_OPERATIONAL_FIELDS if field in metrics
    }
    return {
        "name": artifact.name,
        "run_id": artifact.run_id,
        "identity": _canonical_value(_mapping(artifact.manifest, "identity")),
        "metrics_path": artifact.metrics_path.as_posix(),
        "predictions_path": artifact.predictions_path.as_posix() if artifact.predictions_path else None,
        "sample_count": metrics["sample_count"],
        "retrieval_sample_count": metrics["retrieval_sample_count"],
        "claim_sample_count": metrics.get("claim_sample_count"),
        "retrieval_limit": metrics.get("retrieval_limit"),
        "concurrency": _artifact_concurrency(artifact),
        "k_values": list(metrics["k_values"]),
        "native_unit_metrics": _selected_metrics(
            metrics,
            ("hit_rate", "hit_rate_at_k", "precision_at_k", "context_precision", "empty_retrieval_rate"),
        ),
        "evidence_coverage_metrics": _selected_metrics(
            metrics,
            ("recall_at_k", "context_recall", "claim_evidence_any_at_k", "claim_evidence_complete_at_k"),
        ),
        "generation_metrics": _canonical_value(metrics.get("generation")),
        "generation_metric_coverage": _canonical_value(metrics.get("generation_metric_coverage")),
        "ontology_coverage": _canonical_value(metrics.get("ontology_coverage")),
        "context_budget": _prediction_budget(artifact.predictions),
        "operational": {
            "reported": reported_operational,
            "derived_from_predictions": _prediction_operational(artifact.predictions),
        },
        "slices": _slice_report(metrics.get("slices")),
    }


def _selected_metrics(metrics: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    """Copy selected metrics without inventing a combined score."""
    return {field: _canonical_value(metrics[field]) for field in fields if field in metrics}


def _slice_report(value: Any) -> dict[str, Any]:
    """Group slice metrics with the same semantics as top-level metrics."""
    if not isinstance(value, Mapping):
        return {}
    report: dict[str, Any] = {}
    for field, groups in sorted(value.items(), key=lambda item: str(item[0])):
        if not isinstance(groups, Mapping):
            continue
        report[str(field)] = {
            str(group): {
                "sample_count": metrics.get("sample_count"),
                "retrieval_sample_count": metrics.get("retrieval_sample_count"),
                "claim_sample_count": metrics.get("claim_sample_count"),
                "native_unit_metrics": _selected_metrics(
                    metrics,
                    ("hit_rate", "hit_rate_at_k", "precision_at_k", "context_precision", "empty_retrieval_rate"),
                ),
                "evidence_coverage_metrics": _selected_metrics(
                    metrics,
                    (
                        "recall_at_k",
                        "context_recall",
                        "claim_evidence_any_at_k",
                        "claim_evidence_complete_at_k",
                    ),
                ),
            }
            for group, metrics in sorted(groups.items(), key=lambda item: str(item[0]))
            if isinstance(metrics, Mapping)
        }
    return report


def _prediction_budget(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Derive native-unit, evidence, and generation-context budgets from prediction rows."""
    if not rows:
        return None
    unit_counts: list[int] = []
    evidence_counts: list[int] = []
    context_counts: list[int] = []
    context_characters: list[int] = []
    unit_kinds: Counter[str] = Counter()
    for row in rows:
        units = _sequence(row.get("retrieved_units"))
        if units:
            unit_counts.append(len(units))
            evidence_ids = {
                str(evidence.get("chunk_id"))
                for unit in units
                if isinstance(unit, Mapping)
                for evidence in _sequence(unit.get("evidence"))
                if isinstance(evidence, Mapping) and evidence.get("chunk_id")
            }
            evidence_counts.append(len(evidence_ids))
            unit_kinds.update(str(unit.get("unit_kind") or "unknown") for unit in units if isinstance(unit, Mapping))
        else:
            chunk_ids = {str(value) for value in _sequence(row.get("retrieved_chunk_ids")) if value}
            unit_counts.append(len(chunk_ids))
            evidence_counts.append(len(chunk_ids))
            if chunk_ids:
                unit_kinds["legacy_chunk"] += len(chunk_ids)

        if "retrieved_contexts" in row or "generation_contexts" in row:
            contexts = _sequence(row.get("retrieved_contexts")) or _sequence(row.get("generation_contexts"))
            context_counts.append(len(contexts))
            context_characters.append(sum(len(_context_text(context)) for context in contexts))

    return {
        "mean_retrieved_units": fmean(unit_counts),
        "mean_unique_evidence_chunks": fmean(evidence_counts),
        "generation_context_sample_count": len(context_counts),
        "mean_generation_contexts": fmean(context_counts) if context_counts else None,
        "mean_generation_context_characters": fmean(context_characters) if context_characters else None,
        "unit_kind_counts": dict(sorted(unit_kinds.items())),
    }


def _prediction_operational(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Aggregate declared per-sample latency, token usage, and cost when present."""
    latency_values: list[float] = []
    cost_values: list[float] = []
    usage: dict[str, list[float]] = {field: [] for field in _TOKEN_FIELDS}
    usage_sample_count = 0
    usage_call_count = 0
    usage_available_call_count = 0
    usage_unavailable_call_count = 0
    partial_usage_sample_count = 0
    usage_models: Counter[str] = Counter()
    for row in rows:
        sources = _operational_sources(row)
        latency = _latency_ms(sources)
        if latency is not None:
            latency_values.append(latency)
        cost = _first_number(sources, ("cost_usd",))
        if cost is not None:
            cost_values.append(cost)
        observation = _usage_observation(sources)
        if observation is None:
            continue
        usage_sample_count += 1
        usage_call_count += observation.call_count
        usage_available_call_count += observation.available_call_count
        usage_unavailable_call_count += observation.unavailable_call_count
        if observation.partial:
            partial_usage_sample_count += 1
        usage_models.update(observation.models)
        for field, values in observation.tokens.items():
            usage[field].extend(values)

    return _operational_summary(
        latency_values=latency_values,
        cost_values=cost_values,
        usage=usage,
        usage_sample_count=usage_sample_count,
        usage_call_count=usage_call_count,
        usage_available_call_count=usage_available_call_count,
        usage_unavailable_call_count=usage_unavailable_call_count,
        partial_usage_sample_count=partial_usage_sample_count,
        usage_models=usage_models,
    )


def _operational_summary(
    *,
    latency_values: Sequence[float],
    cost_values: Sequence[float],
    usage: Mapping[str, Sequence[float]],
    usage_sample_count: int,
    usage_call_count: int,
    usage_available_call_count: int,
    usage_unavailable_call_count: int,
    partial_usage_sample_count: int,
    usage_models: Mapping[str, int],
) -> dict[str, Any] | None:
    """Render observed operational values without imputing unavailable measurements."""
    result: dict[str, Any] = {}
    if latency_values:
        result["latency_ms"] = _numeric_summary(latency_values)
    populated_usage = {field: _numeric_summary(values) for field, values in usage.items() if values}
    if populated_usage:
        result["token_usage"] = populated_usage
    if usage_sample_count:
        result["llm_usage"] = {
            "sample_count": usage_sample_count,
            "call_count": usage_call_count,
            "usage_available_call_count": usage_available_call_count,
            "usage_unavailable_call_count": usage_unavailable_call_count,
            "partial_sample_count": partial_usage_sample_count,
            "models": dict(sorted(usage_models.items())),
        }
    if cost_values:
        result["cost_usd"] = _numeric_summary(cost_values)
    return result or None


def _latency_ms(sources: Sequence[Mapping[str, Any]]) -> float | None:
    """Read a declared duration and normalize it to milliseconds."""
    latency = _first_number(sources, ("latency_ms", "duration_ms", "elapsed_ms"))
    if latency is not None:
        return latency
    seconds = _first_number(sources, ("latency_seconds", "duration_seconds", "elapsed_seconds"))
    return seconds * 1000 if seconds is not None else None


def _usage_mapping(sources: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the first structured provider-usage summary from ordered metadata sources."""
    for source in sources:
        for key in ("usage", "token_usage"):
            usage_value = source.get(key)
            if isinstance(usage_value, Mapping):
                return usage_value
    return None


def _usage_observation(sources: Sequence[Mapping[str, Any]]) -> _UsageObservation | None:
    """Normalize usage totals and coverage for one prediction row."""
    usage_value = _usage_mapping(sources)
    if usage_value is None:
        return None
    has_token_counts = any(_number(usage_value.get(field)) is not None for field in _TOKEN_FIELDS)
    available_calls = _integer(usage_value.get("usage_available_call_count"))
    if available_calls is None:
        available_calls = 1 if has_token_counts else 0
    unavailable_calls = _integer(usage_value.get("usage_unavailable_call_count"))
    call_count = _integer(usage_value.get("call_count"))
    if call_count is None:
        call_count = max(available_calls + (unavailable_calls or 0), 1)
    if unavailable_calls is None:
        unavailable_calls = max(call_count - available_calls, 0)
    raw_models = usage_value.get("models")
    models = (
        {str(model): count for model, raw_count in raw_models.items() if (count := _integer(raw_count)) is not None}
        if isinstance(raw_models, Mapping)
        else {}
    )
    return _UsageObservation(
        call_count=call_count,
        available_call_count=available_calls,
        unavailable_call_count=unavailable_calls,
        partial=usage_value.get("partial") is True or unavailable_calls > 0,
        models=models,
        tokens=_token_usage(usage_value, available_calls=available_calls),
    )


def _token_usage(
    usage_value: Mapping[str, Any],
    *,
    available_calls: int,
) -> dict[str, list[float]]:
    """Collect token counters only when at least one provider call reported usage."""
    usage: dict[str, list[float]] = {field: [] for field in _TOKEN_FIELDS}
    if available_calls <= 0:
        return usage
    for field in _TOKEN_FIELDS:
        number = _number(usage_value.get(field))
        if number is not None:
            usage[field].append(number)
    return usage


def _operational_sources(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return known containers that may hold per-sample operational measurements."""
    sources: list[Mapping[str, Any]] = [row]
    for key in ("evaluation_metadata", "operational", "pipeline_metadata"):
        value = row.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
            nested = value.get("operational")
            if isinstance(nested, Mapping):
                sources.append(nested)
    return tuple(sources)


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize observed values without imputing missing samples."""
    return {
        "sample_count": len(values),
        "total": sum(values),
        "mean": fmean(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _sample_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint the selected ground-truth sample multiset, independent of run output."""
    samples = sorted(_canonical_json(_ground_truth_payload(row)) for row in rows)
    return hashlib.sha256(_canonical_json(samples).encode()).hexdigest()


def _ground_truth_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize ground-truth fields shared by retrieval-only and end-to-end outputs."""
    sample = EvaluationSample.from_mapping(row)
    payload = asdict(sample)
    for field in (
        "actual_answer",
        "retrieved_chunk_ids",
        "retrieved_source_names",
        "retrieved_units",
    ):
        payload.pop(field, None)
    identity = {
        field: row[field]
        for field in (
            "question_id",
            "conversation_id",
            "turn_number",
            "scenario_id",
            "query_date",
            "query_dates",
            "reference_date",
        )
        if field in row
    }
    if identity:
        payload["evaluation_identity"] = identity
    return payload


def _find_predictions_path(
    metrics_path: Path,
    manifest_path: Path | None,
    manifest: Mapping[str, Any] | None,
) -> Path | None:
    """Resolve a predictions artifact from manifest output or current filename conventions."""
    candidates: list[Path] = []
    outputs = _mapping(manifest, "outputs")
    output_path = outputs.get("predictions")
    if manifest_path is not None and isinstance(output_path, str) and output_path.strip():
        declared_path = _resolve_output_path(manifest_path, output_path).resolve()
        return declared_path if declared_path.is_file() else None

    candidates.append(metrics_path.parent / "predictions.jsonl")
    stem = metrics_path.stem
    if stem.endswith("_retrieval_metrics"):
        prefix = stem[: -len("_retrieval_metrics")]
        candidates.extend(
            (
                metrics_path.with_name(f"{prefix}_retrieval_predictions.jsonl"),
                metrics_path.with_name(f"{prefix}.jsonl"),
            )
        )
    elif stem.endswith("_metrics"):
        prefix = stem[: -len("_metrics")]
        candidates.extend(
            (
                metrics_path.with_name(f"{prefix}.jsonl"),
                metrics_path.with_name(f"{prefix}_predictions.jsonl"),
            )
        )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _resolve_output_path(manifest_path: Path, value: str) -> Path:
    """Resolve one declared output relative to its manifest."""
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _valid_sha256(value: Any) -> bool:
    """Return whether *value* is a lowercase or uppercase SHA-256 hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _sha256_file(path: Path) -> str:
    """Hash one artifact without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_manifest_path(metrics_path: Path) -> Path | None:
    """Find an end-to-end or metrics-specific retrieval manifest beside one metrics file."""
    candidates = (
        metrics_path.with_suffix(".manifest.json"),
        metrics_path.parent / "manifest.json",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    """Read a JSON object and identify malformed artifacts by path."""
    if not path.is_file():
        msg = f"{label.capitalize()} file not found: {path}"
        raise ValueError(msg)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Could not read {label} file {path}: {error}"
        raise ValueError(msg) from error
    if not isinstance(value, Mapping):
        msg = f"{path}: expected a JSON object"
        raise TypeError(msg)
    return value


def _read_jsonl_objects(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Read prediction rows with precise line diagnostics."""
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    msg = f"{path}:{line_number}: expected a JSON object"
                    raise TypeError(msg)
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Could not read predictions file {path}: {error}"
        raise ValueError(msg) from error
    return tuple(rows)


def _mapping(value: Any, key: str) -> Mapping[str, Any]:
    """Read a nested mapping without accepting other JSON types."""
    if not isinstance(value, Mapping):
        return {}
    nested = value.get(key)
    return nested if isinstance(nested, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    """Return a JSON array while excluding strings and mappings."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _first_string(value: Any, keys: Sequence[str]) -> str | None:
    """Return the first non-empty string field from a mapping."""
    if not isinstance(value, Mapping):
        return None
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def _first_number(sources: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> float | None:
    """Return the first numeric operational value from ordered metadata sources."""
    for source in sources:
        for key in keys:
            value = _number(source.get(key))
            if value is not None:
                return value
    return None


def _number(value: Any) -> float | None:
    """Accept JSON numbers while excluding booleans."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _integer(value: Any) -> int | None:
    """Accept non-negative JSON integer counters while excluding booleans."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _context_text(value: Any) -> str:
    """Read context text from either the v2 object shape or a legacy string."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        text = value.get("text")
        return text if isinstance(text, str) else ""
    return ""


def _common_or_none(values: Sequence[Any]) -> Any:
    """Return a shared value only when every run reports the same non-null value."""
    if not values or any(value is None for value in values):
        return None
    canonical_values = {_canonical_json(value) for value in values}
    return _canonical_value(values[0]) if len(canonical_values) == 1 else None


def _canonical_value(value: Any) -> Any:
    """Return a recursively key-sorted JSON-compatible value."""
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    """Serialize a comparison value deterministically."""
    return json.dumps(_canonical_value(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
