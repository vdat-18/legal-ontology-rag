"""Tests for defensible comparison of RAG evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from legal_rag.evaluation.comparison import IncompatibleEvaluationRunsError, compare_evaluation_runs


def _generation_scores(**overrides: float | None) -> dict[str, float | None]:
    """Return one complete set of aggregate generation metrics."""
    scores: dict[str, float | None] = {
        "faithfulness": 1.0,
        "correctness": 5.0,
        "answer_relevancy": 5.0,
        "temporal_correctness": 1.0,
        "answerability_correctness": 1.0,
        "required_claim_coverage": 1.0,
    }
    scores.update(overrides)
    return scores


def _prompt_hashes(
    pipeline: str,
    *,
    generation: bool,
    query_rewrite_policy: str,
) -> dict[str, str]:
    """Return production-shaped prompt digests for one synthetic run."""
    names = {
        "naive": {"naive_rag_generation.baml"},
        "adaptive": {"adaptive_query_planning.baml", "naive_rag_generation.baml"},
        "corrective": {"corrective_rag_grading.baml", "query_rewrite.baml"},
        "graphiti": {"naive_rag_generation.baml", "query_rewrite.baml"},
        "ontology": {"naive_rag_generation.baml", "ontology_query_planning.baml"},
    }.get(pipeline, set())
    if pipeline in {"naive", "adaptive", "ontology"} and query_rewrite_policy in {
        "enabled_when_history_present",
        "ontology_planner_always",
    }:
        names.add("query_rewrite.baml")
    if generation:
        names.add("answer_evaluation.baml")
    return {
        name: "judge-prompt-digest" if name == "answer_evaluation.baml" else f"{name}-digest" for name in sorted(names)
    }


def test_compare_evaluation_runs_groups_metrics_and_context_budgets(tmp_path: Path) -> None:
    """A compatible report separates native units, evidence coverage, and measured budgets."""
    rerank = _write_run(
        tmp_path / "rerank",
        pipeline="naive",
        mode="hybrid_rerank",
        generation=_generation_scores(correctness=4.5, temporal_correctness=None),
        latency_ms=120.0,
    )
    hybrid = _write_run(
        tmp_path / "hybrid",
        pipeline="naive",
        mode="hybrid",
        generation=None,
        latency_ms=80.0,
    )

    report = compare_evaluation_runs([rerank, hybrid])

    assert report["compatibility"]["status"] == "compatible"
    assert report["compatibility"]["basis"]["ground_truth_sha256"] == "dataset-digest"
    assert report["compatibility"]["generation_metrics_comparable"] is False
    assert [run["name"] for run in report["runs"]] == ["naive:hybrid", "naive:hybrid_rerank"]
    first = report["runs"][0]
    assert first["native_unit_metrics"]["precision_at_k"] == {"1": 1.0, "3": 1 / 3}
    assert first["evidence_coverage_metrics"]["recall_at_k"] == {"1": 0.5, "3": 1.0}
    assert first["context_budget"] == {
        "mean_retrieved_units": 1.0,
        "mean_unique_evidence_chunks": 2.0,
        "generation_context_sample_count": 1,
        "mean_generation_contexts": 1.0,
        "mean_generation_context_characters": 12.0,
        "unit_kind_counts": {"article": 1},
    }
    assert first["operational"]["derived_from_predictions"]["latency_ms"]["mean"] == 80.0
    assert first["operational"]["derived_from_predictions"]["token_usage"]["total_tokens"]["total"] == 30.0
    assert first["slices"]["bundle_kind"]["multi_hop"]["sample_count"] == 1
    assert report["runs"][1]["generation_metrics"]["temporal_correctness"] is None
    assert "composite" not in json.dumps(report)


def test_compare_discovers_metrics_adjacent_manifest_for_custom_output(tmp_path: Path) -> None:
    """An end-to-end metrics artifact remains rankable when exported outside its run directory."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    original = _write_run(tmp_path / "second", pipeline="adaptive", mode=None)
    exported = tmp_path / "exports" / "adaptive.json"
    exported.parent.mkdir()
    original.replace(exported)
    manifest = json.loads((original.parent / "manifest.json").read_text(encoding="utf-8"))
    manifest["outputs"]["metrics"] = exported.resolve().as_posix()
    manifest["output_sha256"]["metrics"] = _file_sha256(exported)
    exported.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_evaluation_runs([first, exported])

    assert report["compatibility"]["status"] == "compatible"
    assert [run["name"] for run in report["runs"]] == ["adaptive", "naive:hybrid"]


def test_manifest_relative_predictions_never_resolve_from_process_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparison reads the exact manifest-relative file whose digest was verified."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(tmp_path / "second", pipeline="adaptive", mode=None)
    verified_directory = second.parent / "verified"
    verified_directory.mkdir()
    verified_predictions = verified_directory / "predictions.jsonl"
    (second.parent / "predictions.jsonl").replace(verified_predictions)
    manifest_path = second.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["predictions"] = "verified/predictions.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_output_digests(second.parent)

    cwd = tmp_path / "cwd"
    misleading_directory = cwd / "verified"
    misleading_directory.mkdir(parents=True)
    (misleading_directory / "predictions.jsonl").write_text(
        json.dumps({"question_id": "wrong", "question": "wrong population"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(cwd)

    report = compare_evaluation_runs([first, second])

    assert report["compatibility"]["status"] == "compatible"
    adaptive = next(run for run in report["runs"] if run["name"] == "adaptive")
    assert adaptive["predictions_path"] == verified_predictions.resolve().as_posix()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": "3.0.0"}, "schema_version"),
        ({"k_values": [1, 5], "metric_k": [1, 5]}, "k_values"),
        ({"retrieval_limit": 5}, "retrieval_limit"),
        ({"sample_count": 2, "retrieval_sample_count": 2}, "sample_count"),
        ({"concurrency": 2}, "concurrency"),
        ({"reference_date": "2026-10-02"}, "reference_date"),
        ({"history_resolution_enabled": False}, "history_resolution_enabled"),
        ({"answer_model": "different-model"}, "answer_model"),
        ({"llm_base_url_sha256": "different-endpoint"}, "llm_base_url_sha256"),
        ({"embedding_profile_sha256": "different-embedding"}, "embedding_profile_sha256"),
        ({"external_evidence_policy": "live_web"}, "external_evidence_policy"),
        ({"corpus_chunks_sha256": "different-corpus"}, "corpus_chunks_sha256"),
    ],
)
def test_compare_evaluation_runs_rejects_incompatible_metric_contracts(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    """Comparison fails clearly when aggregate metrics do not share a contract."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(tmp_path / "second", pipeline="adaptive", mode=None, metric_changes=change)

    with pytest.raises(IncompatibleEvaluationRunsError, match=message):
        compare_evaluation_runs([first, second])


def test_compare_evaluation_runs_rejects_different_sample_content(tmp_path: Path) -> None:
    """Equal sample counts and k-values cannot hide a different evaluated population."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(tmp_path / "second", pipeline="adaptive", mode=None, question="different question")

    with pytest.raises(IncompatibleEvaluationRunsError, match="different evaluated sample sets"):
        compare_evaluation_runs([first, second])


def test_compare_accepts_five_pipeline_history_policies_under_one_history_contract(tmp_path: Path) -> None:
    """Variant-native rewrite mechanisms remain comparable when all resolve benchmark history."""
    configurations = (
        ("naive", "enabled_when_history_present"),
        ("adaptive", "enabled_when_history_present"),
        ("corrective", "intrinsic_when_history_present"),
        ("graphiti", "intrinsic"),
        ("ontology", "ontology_planner_always"),
    )
    metrics_paths = [
        _write_run(
            tmp_path / pipeline,
            pipeline=pipeline,
            mode="hybrid" if pipeline == "naive" else None,
            metric_changes={"query_rewrite_policy": policy},
        )
        for pipeline, policy in configurations
    ]

    report = compare_evaluation_runs(metrics_paths)

    assert report["compatibility"]["status"] == "compatible"
    assert report["compatibility"]["basis"]["controls"]["history_resolution_enabled"] is True
    assert report["compatibility"]["basis"]["controls"]["query_rewrite_policy"] is None


def test_compare_rejects_different_rewrite_policy_within_one_pipeline(tmp_path: Path) -> None:
    """A pipeline ablation must retain a single query preprocessing contract."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(
        tmp_path / "second",
        pipeline="naive",
        mode="hybrid_rerank",
        metric_changes={"query_rewrite_policy": "disabled"},
    )

    with pytest.raises(IncompatibleEvaluationRunsError, match="query_rewrite_policy within pipeline"):
        compare_evaluation_runs([first, second])


def test_compare_rejects_different_material_config_within_one_pipeline(tmp_path: Path) -> None:
    """Search-mode ablations still require identical pools and context budgets."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(tmp_path / "second", pipeline="naive", mode="hybrid_rerank")
    manifest_path = second.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["effective_config"]["naive_rag"]["bm25_k"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IncompatibleEvaluationRunsError, match="effective_config within pipeline 'naive'"):
        compare_evaluation_runs([first, second])


def test_compare_rejects_different_pipeline_prompt_within_one_pipeline(tmp_path: Path) -> None:
    """A changed answer or planning prompt defines a different same-pipeline experiment contract."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(tmp_path / "second", pipeline="naive", mode="hybrid_rerank")
    manifest_path = second.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["prompt_sha256"]["naive_rag_generation.baml"] = "changed-prompt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IncompatibleEvaluationRunsError, match="prompt_sha256 within pipeline 'naive'"):
        compare_evaluation_runs([first, second])


def test_compare_evaluation_runs_fingerprints_plural_query_dates(tmp_path: Path) -> None:
    """Changing temporal targets changes the evaluated sample set even when text is unchanged."""
    first = _write_run(
        tmp_path / "first",
        pipeline="naive",
        mode="hybrid",
        query_dates=["2025-01-01", "2026-03-01"],
    )
    second = _write_run(
        tmp_path / "second",
        pipeline="adaptive",
        mode=None,
        query_dates=["2025-01-01", "2026-08-15"],
    )

    with pytest.raises(IncompatibleEvaluationRunsError, match="different evaluated sample sets"):
        compare_evaluation_runs([first, second])


def test_compare_evaluation_runs_rejects_missing_history_policy_in_one_run(tmp_path: Path) -> None:
    """A retrieval-only artifact cannot silently mix with a run whose history policy is known."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(tmp_path / "second", pipeline="adaptive", mode=None)
    metrics = json.loads(second.read_text(encoding="utf-8"))
    metrics.pop("history_resolution_enabled")
    second.write_text(json.dumps(metrics), encoding="utf-8")
    manifest_path = second.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"].pop("history_resolution_enabled")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_output_digests(second.parent)

    with pytest.raises(IncompatibleEvaluationRunsError, match="history_resolution_enabled is unavailable"):
        compare_evaluation_runs([first, second])


def test_compare_evaluation_runs_rejects_different_ontology_releases(tmp_path: Path) -> None:
    """Ontology variants are comparable only when their Turtle releases are identical."""
    first = _write_run(
        tmp_path / "first",
        pipeline="ontology",
        mode=None,
        ontology_sha256=["a" * 64],
    )
    second = _write_run(
        tmp_path / "second",
        pipeline="ontology",
        mode=None,
        ontology_sha256=["b" * 64],
    )

    with pytest.raises(IncompatibleEvaluationRunsError, match="ontology_sha256"):
        compare_evaluation_runs([first, second])


def test_compare_evaluation_runs_can_report_incompatibilities_without_raising(tmp_path: Path) -> None:
    """Callers can render diagnostics for incompatible runs without treating them as comparable."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(
        tmp_path / "second",
        pipeline="adaptive",
        mode=None,
        ground_truth_sha256="other-dataset",
    )

    report = compare_evaluation_runs([second, first], allow_incompatible=True)

    assert report["compatibility"]["status"] == "incompatible"
    assert any("different ground-truth" in issue for issue in report["compatibility"]["issues"])
    assert [run["name"] for run in report["runs"]] == ["adaptive", "naive:hybrid"]


def test_legacy_metrics_without_manifest_are_diagnostic_only(tmp_path: Path) -> None:
    """Legacy standalone metrics remain inspectable but never become ranked-compatible runs."""
    complete = _write_run(tmp_path / "complete", pipeline="naive", mode="hybrid")
    legacy = _write_run(tmp_path / "legacy", pipeline="adaptive", mode=None)
    payload = json.loads(legacy.read_text(encoding="utf-8"))
    payload["schema_version"] = "1.0.0"
    legacy.write_text(json.dumps(payload), encoding="utf-8")
    (legacy.parent / "manifest.json").unlink()

    report = compare_evaluation_runs([complete, legacy], allow_incompatible=True)

    assert report["compatibility"]["status"] == "incompatible"
    assert any("schema_version 2.0.0" in issue for issue in report["compatibility"]["issues"])
    assert any("complete evaluation manifest" in issue for issue in report["compatibility"]["issues"])
    with pytest.raises(IncompatibleEvaluationRunsError):
        compare_evaluation_runs([complete, legacy])


def test_incomplete_manifest_is_diagnostic_only_when_explicitly_allowed(tmp_path: Path) -> None:
    """A running manifest cannot enter rankings, while diagnostics can explain why."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(tmp_path / "second", pipeline="adaptive", mode=None)
    manifest_path = second.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_evaluation_runs([first, second], allow_incompatible=True)

    assert report["compatibility"]["status"] == "incompatible"
    assert any("status must be 'complete'" in issue for issue in report["compatibility"]["issues"])


def test_manifest_without_output_digests_is_diagnostic_only(tmp_path: Path) -> None:
    """A legacy manifest without integrity attestations cannot be ranked as complete."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(tmp_path / "second", pipeline="adaptive", mode=None)
    manifest_path = second.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("output_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_evaluation_runs([first, second], allow_incompatible=True)

    assert report["compatibility"]["status"] == "incompatible"
    assert any("output_sha256" in issue for issue in report["compatibility"]["issues"])


@pytest.mark.parametrize("output_name", ["predictions", "metrics", "generation"])
def test_comparison_rejects_tampered_manifest_outputs(tmp_path: Path, output_name: str) -> None:
    """Declared prediction, metric, and generation bytes are all integrity checked."""
    first = _write_run(
        tmp_path / "first",
        pipeline="naive",
        mode="hybrid",
        generation=_generation_scores(),
    )
    second = _write_run(
        tmp_path / "second",
        pipeline="adaptive",
        mode=None,
        generation=_generation_scores(),
    )
    manifest = json.loads((second.parent / "manifest.json").read_text(encoding="utf-8"))
    output_path = Path(manifest["outputs"][output_name])
    output_path.write_bytes(output_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match=f"output '{output_name}'.*SHA-256"):
        compare_evaluation_runs([first, second], allow_incompatible=True)


def test_manifest_identity_reconciles_material_model_control(tmp_path: Path) -> None:
    """A manifest cannot attest one model while its metrics report another."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(tmp_path / "second", pipeline="adaptive", mode=None)
    manifest_path = second.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["answer_model"] = "other-model"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="answer_model does not match manifest identity"):
        compare_evaluation_runs([first, second], allow_incompatible=True)


def test_comparison_rejects_different_shared_evaluator_source(tmp_path: Path) -> None:
    """Runs scored by different shared evaluator code cannot enter one ranking."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(tmp_path / "second", pipeline="adaptive", mode=None)
    manifest_path = second.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["source_sha256"]["legal_rag.evaluation.metrics"] = "changed-metrics-source"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IncompatibleEvaluationRunsError, match="evaluation_metrics_sha256"):
        compare_evaluation_runs([first, second])


@pytest.mark.parametrize(
    ("module_name", "control_name"),
    [
        ("legal_rag.embedding", "embedding_source_sha256"),
        ("legal_rag.query_rewriting", "query_rewriting_source_sha256"),
        ("legal_rag.retrieval", "retrieval_source_sha256"),
        ("legal_rag.temporal", "temporal_source_sha256"),
    ],
)
def test_comparison_rejects_different_shared_runtime_source(
    tmp_path: Path,
    module_name: str,
    control_name: str,
) -> None:
    """Query-time helpers are reproducibility controls across every RAG variant."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(tmp_path / "second", pipeline="adaptive", mode=None)
    manifest_path = second.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["source_sha256"][module_name] = f"changed-{module_name}"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IncompatibleEvaluationRunsError, match=control_name):
        compare_evaluation_runs([first, second])


def test_comparison_rejects_different_generation_evaluator_source(tmp_path: Path) -> None:
    """Six-metric rankings bind dataset mapping, judging, batching, and aggregation code."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid", generation=_generation_scores())
    second = _write_run(tmp_path / "second", pipeline="adaptive", mode=None, generation=_generation_scores())
    manifest_path = second.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["source_sha256"]["legal_rag.evaluation.generation"] = "changed-generation-source"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IncompatibleEvaluationRunsError, match="generation_source_sha256"):
        compare_evaluation_runs([first, second])


def test_compare_rejects_different_active_snapshots_within_one_pipeline(tmp_path: Path) -> None:
    """Graphiti ablations are rankable only when they queried the same graph snapshot."""
    first = _write_run(tmp_path / "first", pipeline="graphiti", mode=None)
    second = _write_run(
        tmp_path / "second",
        pipeline="graphiti",
        mode=None,
        metric_changes={"active_backend_fingerprint": {"kind": "graphiti", "sha256": "other-snapshot"}},
    )

    with pytest.raises(IncompatibleEvaluationRunsError, match="active_backend_fingerprint within pipeline 'graphiti'"):
        compare_evaluation_runs([first, second])


def test_compare_evaluation_runs_requires_predictions_to_verify_sample_set(tmp_path: Path) -> None:
    """Matching aggregates alone do not establish that two runs evaluated the same rows."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(tmp_path / "second", pipeline="adaptive", mode=None)
    (second.parent / "predictions.jsonl").unlink()

    with pytest.raises(ValueError, match="manifest output 'predictions' was not found"):
        compare_evaluation_runs([first, second])


def test_compare_rejects_metrics_without_verified_active_backend(tmp_path: Path) -> None:
    """A local corpus hash alone cannot make an unverified live index comparable."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid")
    second = _write_run(tmp_path / "second", pipeline="adaptive", mode=None)
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload.pop("active_backend_fingerprint")
    payload.pop("active_backend_validated")
    second.write_text(json.dumps(payload), encoding="utf-8")
    _refresh_output_digests(second.parent)

    with pytest.raises(ValueError, match="active_backend"):
        compare_evaluation_runs([first, second])


def test_compare_treats_live_web_runs_as_diagnostic_only(tmp_path: Path) -> None:
    """Two live-web runs still lack a frozen evidence universe and cannot be fairly ranked."""
    change = {"external_evidence_policy": "live_web"}
    first = _write_run(tmp_path / "first", pipeline="corrective", mode=None, metric_changes=change)
    second = _write_run(tmp_path / "second", pipeline="corrective", mode=None, metric_changes=change)

    with pytest.raises(IncompatibleEvaluationRunsError, match="not a frozen corpus snapshot"):
        compare_evaluation_runs([first, second])


def test_comparison_does_not_report_zero_tokens_when_provider_usage_is_unavailable(tmp_path: Path) -> None:
    """Missing provider counters remain missing while call coverage stays visible."""
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid", usage_available=False)
    second = _write_run(tmp_path / "second", pipeline="adaptive", mode=None, usage_available=False)

    report = compare_evaluation_runs([first, second])

    operational = report["runs"][0]["operational"]["derived_from_predictions"]
    assert "token_usage" not in operational
    assert operational["llm_usage"] == {
        "sample_count": 1,
        "call_count": 1,
        "usage_available_call_count": 0,
        "usage_unavailable_call_count": 1,
        "partial_sample_count": 1,
        "models": {"test-model": 1},
    }


def test_generation_comparison_rejects_different_judge_controls(tmp_path: Path) -> None:
    """Answer scores from different judge models cannot be treated as one comparable scale."""
    generation = _generation_scores(correctness=4.0)
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid", generation=generation)
    second = _write_run(
        tmp_path / "second",
        pipeline="adaptive",
        mode=None,
        generation=generation,
        metric_changes={"judge_model": "different-judge"},
    )

    with pytest.raises(IncompatibleEvaluationRunsError, match="judge_model"):
        compare_evaluation_runs([first, second])

    diagnostic = compare_evaluation_runs([first, second], allow_incompatible=True)
    assert diagnostic["compatibility"]["generation_metrics_comparable"] is False


def test_generation_comparison_accepts_and_exposes_the_complete_six_metric_contract(tmp_path: Path) -> None:
    """Complete generation aggregates and coverage remain visible and comparable."""
    generation = _generation_scores(temporal_correctness=None)
    first = _write_run(tmp_path / "first", pipeline="naive", mode="hybrid", generation=generation)
    second = _write_run(tmp_path / "second", pipeline="adaptive", mode=None, generation=generation)

    report = compare_evaluation_runs([first, second])

    assert report["compatibility"]["generation_metrics_comparable"] is True
    for run in report["runs"]:
        assert set(run["generation_metrics"]) == set(generation)
        assert set(run["generation_metric_coverage"]) == set(generation)


@pytest.mark.parametrize("generation", [{}, {"correctness": 4.0}])
def test_comparison_rejects_empty_or_incomplete_generation_metrics(
    tmp_path: Path, generation: dict[str, float | None]
) -> None:
    """A generation mapping cannot be ranked unless all six versioned metrics are present."""
    first = _write_run(
        tmp_path / "first",
        pipeline="naive",
        mode="hybrid",
        generation=_generation_scores(),
    )
    second = _write_run(
        tmp_path / "second",
        pipeline="adaptive",
        mode=None,
        generation=generation,
    )

    with pytest.raises(ValueError, match="must contain exactly six generation metrics"):
        compare_evaluation_runs([first, second])


def test_comparison_rejects_generation_coverage_that_does_not_match_sample_count(tmp_path: Path) -> None:
    """Coverage counters must account for every evaluated sample before scores are compared."""
    first = _write_run(
        tmp_path / "first",
        pipeline="naive",
        mode="hybrid",
        generation=_generation_scores(),
    )
    second = _write_run(
        tmp_path / "second",
        pipeline="adaptive",
        mode=None,
        generation=_generation_scores(),
    )
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["generation_metric_coverage"]["faithfulness"]["scored_count"] = 0
    second.write_text(json.dumps(payload), encoding="utf-8")
    _refresh_output_digests(second.parent)

    with pytest.raises(ValueError, match="coverage counts for faithfulness must sum to sample_count"):
        compare_evaluation_runs([first, second])


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("sample_count", "sample_count must match metrics and predictions"),
        ("sample_id", "sample_id values do not match predictions in order"),
        ("aggregate", "aggregate does not match per_sample scores"),
        ("coverage", "metric_coverage does not match per_sample scores"),
    ],
)
def test_comparison_rejects_semantically_inconsistent_generation_report(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    """A refreshed byte digest cannot hide generation report content that contradicts its rows."""
    first = _write_run(
        tmp_path / "first",
        pipeline="naive",
        mode="hybrid",
        generation=_generation_scores(),
    )
    second = _write_run(
        tmp_path / "second",
        pipeline="adaptive",
        mode=None,
        generation=_generation_scores(),
    )
    generation_path = second.parent / "generation.json"
    report = json.loads(generation_path.read_text(encoding="utf-8"))
    if tamper == "sample_count":
        report["sample_count"] = 2
    elif tamper == "sample_id":
        report["per_sample"][0]["sample_id"] = "wrong-sample"
    elif tamper == "aggregate":
        report["aggregate"]["faithfulness"] = 0.0
    else:
        report["metric_coverage"]["faithfulness"] = {
            "scored_count": 0,
            "null_count": 1,
            "coverage": 0.0,
        }
    generation_path.write_text(json.dumps(report), encoding="utf-8")
    _refresh_output_digests(second.parent)

    with pytest.raises(ValueError, match=message):
        compare_evaluation_runs([first, second])


def _write_run(
    directory: Path,
    *,
    pipeline: str,
    mode: str | None,
    generation: dict[str, float | None] | None = None,
    latency_ms: float = 50.0,
    ground_truth_sha256: str = "dataset-digest",
    question: str = "question",
    metric_changes: dict[str, Any] | None = None,
    usage_available: bool = True,
    query_dates: list[str] | None = None,
    ontology_sha256: list[str] | None = None,
) -> Path:
    """Write one minimal complete shared-evaluation artifact set."""
    directory.mkdir(parents=True)
    metrics_path = directory / "metrics.json"
    predictions_path = directory / "predictions.jsonl"
    metric_changes = metric_changes or {}
    k_values = metric_changes.get("metric_k", metric_changes.get("k_values", [1, 3]))
    metrics: dict[str, Any] = {
        "schema_version": "2.0.0",
        "run_id": directory.name,
        "pipeline": pipeline,
        "active_backend_validated": True,
        "active_backend_fingerprint": {"kind": pipeline, "sha256": f"{pipeline}-active"},
        "search_mode": mode,
        "retrieval_limit": 3,
        "concurrency": 1,
        "reference_date": "2026-10-01",
        "history_resolution_enabled": True,
        "query_rewrite_policy": "enabled_when_history_present",
        "answer_model": "test-model",
        "llm_base_url_sha256": "endpoint-digest",
        "embedding_profile": "test-profile",
        "embedding_profile_sha256": "embedding-digest",
        "llm_extra_request_body_sha256": "extra-body-digest",
        "external_evidence_policy": "closed_corpus",
        "corpus_chunks_sha256": "corpus-digest",
        "ground_truth_sha256": ground_truth_sha256,
        "judge_model": "judge-model",
        "judge_batch_size": 5,
        "judge_concurrency": 2,
        "judge_prompt_sha256": "judge-prompt-digest",
        "sample_count": 1,
        "retrieval_sample_count": 1,
        "claim_sample_count": 1,
        "k_values": k_values,
        "hit_rate": 1.0,
        "hit_rate_at_k": {str(k): 1.0 for k in k_values},
        "precision_at_k": {str(k): 1 / k for k in k_values},
        "recall_at_k": {str(k): 1.0 if k == max(k_values) else 0.5 for k in k_values},
        "context_precision": 1.0,
        "context_recall": 1.0,
        "empty_retrieval_rate": 0.0,
        "claim_evidence_any_at_k": {str(k): 1.0 for k in k_values},
        "claim_evidence_complete_at_k": {str(k): 1.0 for k in k_values},
        "slices": {
            "bundle_kind": {
                "multi_hop": {
                    "sample_count": 1,
                    "retrieval_sample_count": 1,
                    "claim_sample_count": 1,
                    "hit_rate": 1.0,
                    "hit_rate_at_k": {str(k): 1.0 for k in k_values},
                    "precision_at_k": {str(k): 1.0 for k in k_values},
                    "recall_at_k": {str(k): 1.0 for k in k_values},
                }
            }
        },
    }
    metrics.update({key: value for key, value in metric_changes.items() if key != "metric_k"})
    if generation is not None:
        metrics["generation"] = generation
        metrics["generation_metric_coverage"] = {
            name: {
                "scored_count": metrics["sample_count"] if value is not None else 0,
                "null_count": 0 if value is not None else metrics["sample_count"],
                "coverage": 1.0 if value is not None else 0.0,
            }
            for name, value in generation.items()
        }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    predictions = [
        {
            "question_id": f"question-{index}",
            "question": question if index == 0 else f"{question} {index}",
            "expected_answer": "answer",
            "expected_chunk_ids": ["chunk-1", "chunk-2"],
            "required_claims": [{"claim": "answer", "evidence": [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-2"}]}],
            "bundle_kind": "multi_hop",
            "answerability": "answerable",
            "reasoning_hops": 2,
            "query_style": "standard",
            **({"query_dates": query_dates} if query_dates is not None else {}),
            "actual_answer": "generated",
            "retrieved_chunk_ids": ["chunk-1", "chunk-2"],
            "retrieved_units": [
                {
                    "unit_id": "article-1",
                    "unit_kind": "article",
                    "text": "unit",
                    "evidence": [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-2"}],
                }
            ],
            "retrieved_contexts": [{"text": "context text"}],
            "evaluation_metadata": {
                "latency_ms": latency_ms,
                "usage": (
                    {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
                    if usage_available
                    else {
                        "call_count": 1,
                        "usage_available_call_count": 0,
                        "usage_unavailable_call_count": 1,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "models": {"test-model": 1},
                        "partial": True,
                    }
                ),
            },
        }
        for index in range(metrics["sample_count"])
    ]
    predictions_path.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    generation_path = directory / "generation.json"
    if generation is not None:
        generation_path.write_text(
            json.dumps(
                {
                    "sample_count": metrics["sample_count"],
                    "aggregate": generation,
                    "metric_coverage": metrics["generation_metric_coverage"],
                    "per_sample": [{"sample_id": row["question_id"], **generation} for row in predictions],
                }
            ),
            encoding="utf-8",
        )
    outputs = {"predictions": predictions_path.as_posix(), "metrics": metrics_path.as_posix()}
    if generation is not None:
        outputs["generation"] = generation_path.as_posix()
    manifest = {
        "schema_version": "1.0.0",
        "run_id": directory.name,
        "status": "complete",
        "identity": {
            "pipeline": pipeline,
            "active_backend_validated": metrics["active_backend_validated"],
            "active_backend_fingerprint": metrics["active_backend_fingerprint"],
            "naive_search_mode": mode,
            "ground_truth_sha256": ground_truth_sha256,
            "max_samples": metrics.get("max_samples"),
            "retrieval_limit": metrics["retrieval_limit"],
            "concurrency": metrics["concurrency"],
            "reference_date": metrics["reference_date"],
            "history_resolution_enabled": metrics["history_resolution_enabled"],
            "query_rewrite_policy": metrics["query_rewrite_policy"],
            "answer_model": metrics["answer_model"],
            "llm_base_url_sha256": metrics["llm_base_url_sha256"],
            "embedding_profile": metrics["embedding_profile"],
            "embedding_profile_sha256": metrics["embedding_profile_sha256"],
            "llm_extra_request_body_sha256": metrics["llm_extra_request_body_sha256"],
            "external_evidence_policy": metrics["external_evidence_policy"],
            "corpus_chunks_sha256": metrics["corpus_chunks_sha256"],
            "k_values": metrics["k_values"],
            "generation": generation is not None,
            "dependency_lock_sha256": "lock-digest",
            "effective_config": {
                "llm_model": "test-model",
                "llm_timeout_seconds": 30.0,
                **(
                    {
                        "naive_rag": {
                            "search_mode": mode or "hybrid",
                            "bm25_k": 20,
                            "dense_k": 20,
                            "context_max_characters": 12000,
                        }
                    }
                    if pipeline in {"naive", "adaptive", "ontology"}
                    else {}
                ),
                **({"adaptive_rag": {"classification_temperature": 0.0}} if pipeline == "adaptive" else {}),
                **(
                    {"corrective_rag": {"web_search_enabled": False, "retrieval_k": 20}}
                    if pipeline == "corrective"
                    else {}
                ),
                **(
                    {
                        "graphiti": {
                            "expand_around_top_fact": True,
                            "rerank_with_cross_encoder": False,
                            "search_limit": 20,
                        }
                    }
                    if pipeline == "graphiti"
                    else {}
                ),
            },
            **({"ontology_sha256": ontology_sha256 or ["a" * 64]} if pipeline == "ontology" else {}),
            "source_sha256": {
                "legal_rag.baml_transport": "baml-transport-digest",
                "legal_rag.benchmark_synthesis.vietnamese_lexical": "vietnamese-lexical-digest",
                "legal_rag.datastores": "datastores-digest",
                "legal_rag.embedding": "embedding-digest",
                "legal_rag.evaluation.cli": "evaluation-cli-digest",
                "legal_rag.evaluation.metrics": "evaluation-metrics-digest",
                "legal_rag.evaluation.runner": "evaluation-runner-digest",
                "legal_rag.query_rewriting": "query-rewriting-digest",
                "legal_rag.retrieval": "retrieval-digest",
                "legal_rag.settings": "settings-digest",
                "legal_rag.temporal": "temporal-digest",
                **({"legal_rag.evaluation.generation": "generation-source-digest"} if generation is not None else {}),
            },
            "judge_model": metrics["judge_model"] if generation is not None else None,
            "judge_batch_size": metrics["judge_batch_size"] if generation is not None else None,
            "judge_concurrency": metrics["judge_concurrency"] if generation is not None else None,
            "prompt_sha256": _prompt_hashes(
                pipeline,
                generation=generation is not None,
                query_rewrite_policy=metrics["query_rewrite_policy"],
            ),
        },
        "outputs": outputs,
        "output_sha256": {name: _file_sha256(Path(path)) for name, path in outputs.items()},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return metrics_path


def _file_sha256(path: Path) -> str:
    """Hash one synthetic artifact exactly as the production manifest does."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_output_digests(directory: Path) -> None:
    """Refresh fixture digests after a test intentionally changes a non-integrity field."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"] = {
        name: _file_sha256(_resolve_fixture_output(manifest_path, path))
        for name, path in manifest["outputs"].items()
        if isinstance(path, str) and path
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _resolve_fixture_output(manifest_path: Path, value: str) -> Path:
    """Resolve one synthetic manifest output with the production relative-path rule."""
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path
