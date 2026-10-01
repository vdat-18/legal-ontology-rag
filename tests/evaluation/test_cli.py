"""Tests for the shared Legal RAG evaluation CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import pytest

from legal_rag.evaluation import cli
from legal_rag.evaluation.metrics import EvaluationSample
from legal_rag.evaluation.runner import (
    EvaluationRecord,
    PipelinePrediction,
    PredictionContext,
    PredictionEvidence,
    RetrievedUnit,
)


@pytest.mark.parametrize("pipeline", ["naive", "adaptive", "corrective", "graphiti", "ontology"])
def test_run_accepts_every_comparable_pipeline(pipeline: str) -> None:
    """The end-to-end command exposes every comparable implementation behind one interface."""
    arguments = cli._parser().parse_args(["run", "--pipeline", pipeline])

    assert arguments.pipeline == pipeline
    assert arguments.ground_truth.as_posix() == "data/benchmark_synthesis/groundtruth.jsonl"
    assert arguments.limit == 10


def test_adaptive_run_identity_contains_effective_planner_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Adaptive resumes are tied to effective defaults as well as explicit environment values."""
    ground_truth = tmp_path / "groundtruth.jsonl"
    ground_truth.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("ADAPTIVE_RAG_RRF_K", "30")
    arguments = cli._parser().parse_args(["run", "--pipeline", "adaptive"])

    identity = cli._run_identity(arguments, ground_truth)

    assert identity["adaptive_config"] == {
        "base_search_mode": "hybrid_rerank",
        "max_subqueries": 3,
        "per_leg_limit": 10,
        "rrf_k": 30,
        "planner_max_output_tokens": 2048,
        "planner_temperature": 0.0,
        "planner_max_response_bytes": 262144,
    }


def test_run_identity_fingerprints_provider_and_resolved_embedding_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Run identity distinguishes endpoint routing and immutable embedding model details."""
    ground_truth = tmp_path / "groundtruth.jsonl"
    ground_truth.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.example/v1/")
    monkeypatch.setenv("EMBEDDING_PROFILE", "bge_m3")
    arguments = cli._parser().parse_args(["run", "--pipeline", "naive"])

    identity = cli._run_identity(arguments, ground_truth)

    assert identity["llm_base_url_sha256"] == hashlib.sha256(b"https://provider.example/v1/").hexdigest()
    assert identity["embedding_profile_sha256"] == identity["effective_config"]["embedding_profile_sha256"]
    assert identity["embedding_profile_sha256"] is not None
    assert identity["external_evidence_policy"] == "closed_corpus"
    assert identity["dependency_lock_sha256"] == cli._sha256_file(Path(cli.__file__).resolve().parents[3] / "uv.lock")
    assert {
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
    } <= set(identity["source_sha256"])


def test_external_evidence_policy_marks_live_web_corrective_runs() -> None:
    """CRAG web evidence is kept out of closed-corpus comparisons."""
    assert cli._external_evidence_policy("corrective", {"corrective_rag": {"web_search_enabled": True}}) == "live_web"
    assert cli._external_evidence_policy("corrective", {"corrective_rag": {"web_search_enabled": False}}) == (
        "closed_corpus"
    )


def test_ontology_rewrite_policy_records_always_on_query_planner() -> None:
    """Ontology retrieval records that its planner rewrites even without conversation history."""
    arguments = cli._parser().parse_args(["run", "--pipeline", "ontology", "--rewrite-query"])

    assert cli._query_rewrite_policy(arguments) == "ontology_planner_always"


def test_corrective_run_identity_resolves_default_profile_and_live_web_without_profile_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CRAG defaults remain identity controls even when EMBEDDING_PROFILE is omitted."""
    from legal_rag.corrective_rag.generate import config as corrective_config_module

    ground_truth = tmp_path / "groundtruth.jsonl"
    ground_truth.write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("CRAG_WEB_SEARCH_ENABLED", "true")
    monkeypatch.delenv("EMBEDDING_PROFILE", raising=False)
    monkeypatch.setattr(corrective_config_module, "load_dotenv", lambda: False)
    arguments = cli._parser().parse_args(["run", "--pipeline", "corrective"])

    identity = cli._run_identity(arguments, ground_truth)

    assert identity["external_evidence_policy"] == "live_web"
    assert identity["embedding_profile"] == "vietnamese_embedding"
    assert identity["embedding_profile_sha256"] is not None
    assert identity["effective_config"]["corrective_rag"]["web_search_enabled"] is True


def test_graphiti_run_identity_hashes_only_its_selected_chunk_corpus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A Graphiti subset cannot share the full-corpus compatibility digest."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunk_rows = [
        {"chunk_id": "chunk-a", "source_name": "a.docx", "text": "A"},
        {"chunk_id": "chunk-b", "source_name": "b.docx", "text": "B"},
    ]
    (chunks_dir / "corpus.chunks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunk_rows),
        encoding="utf-8",
    )
    corpus_config = tmp_path / "corpus.json"
    corpus_config.write_text(
        json.dumps({"group_id": "test-group", "documents": [{"file": "a.docx"}]}),
        encoding="utf-8",
    )
    ground_truth = tmp_path / "groundtruth.jsonl"
    ground_truth.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("NAIVE_RAG_CHUNKS_DIR", str(chunks_dir))
    monkeypatch.delenv("EMBEDDING_PROFILE", raising=False)
    arguments = cli._parser().parse_args(
        [
            "run",
            "--pipeline",
            "graphiti",
            "--group-id",
            "test-group",
            "--corpus-config",
            str(corpus_config),
        ]
    )

    identity = cli._run_identity(arguments, ground_truth)

    selected_digest = cli._chunk_corpus_sha256(chunks_dir, source_names=frozenset({"a.docx"}))
    assert identity["corpus_chunks_sha256"] == selected_digest
    assert identity["graphiti_corpus"]["chunks_sha256"] == selected_digest
    assert selected_digest != cli._chunk_corpus_sha256(chunks_dir)


@pytest.mark.parametrize("pipeline", ["adaptive", "ontology"])
def test_runtime_prompt_identity_includes_enabled_history_rewriter(pipeline: str) -> None:
    """Planner pipelines also fingerprint the shared query-rewrite prompt used for multi-turn rows."""
    arguments = cli._parser().parse_args(["run", "--pipeline", pipeline])

    prompt_hashes = cli._runtime_prompt_hashes(arguments)

    assert "query_rewrite.baml" in prompt_hashes


def test_ontology_run_identity_hashes_runtime_prompts_outside_repo_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ontology resume identity hashes installed BAML sources without relying on repository cwd."""
    from legal_rag.baml_client.inlinedbaml import get_baml_files

    ground_truth = tmp_path / "groundtruth.jsonl"
    ontology = tmp_path / "ontology.ttl"
    outside_repo = tmp_path / "outside"
    ground_truth.write_text("{}\n", encoding="utf-8")
    ontology.write_text("@prefix test: <https://example.test/> .\n", encoding="utf-8")
    outside_repo.mkdir()
    arguments = cli._parser().parse_args(["run", "--pipeline", "ontology", "--ontology-paths", str(ontology)])
    monkeypatch.chdir(outside_repo)

    identity = cli._run_identity(arguments, ground_truth)

    runtime_sources = get_baml_files()
    assert (
        identity["ontology_runtime"]["query_planner_sha256"]
        == hashlib.sha256(runtime_sources["ontology_query_planning.baml"].encode()).hexdigest()
    )
    assert (
        identity["ontology_runtime"]["generation_prompt_sha256"]
        == hashlib.sha256(runtime_sources["naive_rag_generation.baml"].encode()).hexdigest()
    )


def test_corrective_run_identity_contains_effective_secret_free_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Corrective resumes are tied to effective grading, refinement, and retrieval settings."""
    from dataclasses import replace

    from legal_rag.corrective_rag.generate.config import RAGConfig

    ground_truth = tmp_path / "groundtruth.jsonl"
    ground_truth.write_text("{}\n", encoding="utf-8")
    private_value = "private-test-value"
    config = replace(
        RAGConfig.from_env(
            {
                "LLM_BASE_URL": "https://provider.example/v1",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "test-model",
                "LLM_TIMEOUT_SECONDS": "30",
            }
        ),
        llm_model="provider/model",
        llm_api_key=private_value,
        milvus_token=private_value,
        embedding_profile="bge_m3",
        milvus_database="legal_rag",
        milvus_collection_name="legal_chunks",
        milvus_index_type="IVF_FLAT",
        milvus_metric_type="COSINE",
        milvus_nlist=128,
        milvus_nprobe=8,
        evaluator_max_output_tokens=512,
        evaluator_temperature=0.0,
        evaluator_upper_threshold=0.8,
        evaluator_lower_threshold=0.3,
        max_response_bytes=262_144,
        refinement_max_strips_per_chunk=6,
        refinement_strip_max_chars=700,
        refinement_top_internal_strips=4,
        refinement_top_external_strips=3,
        refinement_decompose_mode="excerption",
        web_search_enabled=False,
        web_search_max_results=3,
        web_search_allowed_domains=("moj.gov.vn",),
    )
    monkeypatch.setattr(RAGConfig, "from_env", classmethod(lambda _cls: config))
    arguments = cli._parser().parse_args(["run", "--pipeline", "corrective"])

    identity = cli._run_identity(arguments, ground_truth)

    expected = {
        "llm_model": "provider/model",
        "embedding_profile": "bge_m3",
        "milvus_database": "legal_rag",
        "milvus_collection_name": "legal_chunks",
        "milvus_index_type": "IVF_FLAT",
        "milvus_metric_type": "COSINE",
        "milvus_nlist": 128,
        "milvus_nprobe": 8,
        "evaluator_max_output_tokens": 512,
        "evaluator_temperature": 0.0,
        "evaluator_upper_threshold": 0.8,
        "evaluator_lower_threshold": 0.3,
        "max_response_bytes": 262_144,
        "refinement_max_strips_per_chunk": 6,
        "refinement_strip_max_chars": 700,
        "refinement_top_internal_strips": 4,
        "refinement_top_external_strips": 3,
        "refinement_decompose_mode": "excerption",
        "web_search_enabled": False,
        "web_search_max_results": 3,
        "web_search_allowed_domains": ["moj.gov.vn"],
    }
    assert {key: identity["corrective_config"][key] for key in expected} == expected
    json.dumps(identity)
    assert "llm_api_key" not in identity["corrective_config"]
    assert "milvus_token" not in identity["corrective_config"]


def test_builds_adaptive_backend_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adaptive CLI branch wires all four settings objects into its pipeline."""
    from legal_rag.adaptive_rag import pipeline as pipeline_module
    from legal_rag.settings import AdaptiveRagSettings, MilvusSettings, NaiveRagSettings, Settings

    created: list[tuple[object, ...]] = []

    class FakePipeline:
        def __init__(self, *settings: object) -> None:
            self.settings = settings[-1]
            created.append(settings)

    app_settings = object()
    milvus_settings = object()
    naive_settings = object()
    adaptive_settings = AdaptiveRagSettings()
    monkeypatch.setattr(Settings, "from_environment", classmethod(lambda _cls: app_settings))
    monkeypatch.setattr(MilvusSettings, "from_environment", classmethod(lambda _cls: milvus_settings))
    monkeypatch.setattr(NaiveRagSettings, "from_environment", classmethod(lambda _cls: naive_settings))
    monkeypatch.setattr(AdaptiveRagSettings, "from_environment", classmethod(lambda _cls: adaptive_settings))
    monkeypatch.setattr(pipeline_module, "AdaptiveRAGPipeline", FakePipeline)
    arguments = cli._parser().parse_args(["run", "--pipeline", "adaptive", "--rewrite-query"])

    backend = cli._build_evaluation_backend(arguments)

    assert created == [(app_settings, milvus_settings, naive_settings, adaptive_settings)]
    assert backend.name == "adaptive"
    assert backend._rewrite_query is True


def test_run_parses_ontology_paths() -> None:
    """Ontology TTL paths are collected as a list for the ontology pipeline."""
    arguments = cli._parser().parse_args(["run", "--pipeline", "ontology", "--ontology-paths", "a.ttl", "b.ttl"])

    assert arguments.ontology_paths == [Path("a.ttl"), Path("b.ttl")]


def test_run_parses_a_frozen_reference_date() -> None:
    """Temporal evaluation accepts one canonical date shared by the run."""
    arguments = cli._parser().parse_args(["run", "--reference-date", "2026-08-31"])

    assert arguments.reference_date.isoformat() == "2026-08-31"


@pytest.mark.parametrize(
    "rows",
    [
        [
            {"question_id": "duplicate", "question": "first"},
            {"question_id": "duplicate", "question": "second"},
        ],
        [
            {"question": "same fallback identity"},
            {"question": "same fallback identity"},
        ],
    ],
)
def test_load_run_records_rejects_duplicate_checkpoint_identities(tmp_path: Path, rows: list[dict[str, str]]) -> None:
    """Fresh runs fail before preflight instead of collapsing duplicate samples in a map."""
    path = tmp_path / "groundtruth.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(SystemExit, match="Duplicate evaluation record id"):
        cli._load_run_records(path, max_samples=None)

    with pytest.raises(SystemExit):
        cli._parser().parse_args(["run", "--reference-date", "20260831"])


def test_run_resolves_history_by_default_and_allows_an_explicit_ablation() -> None:
    """Comparable multi-turn runs resolve follow-ups unless the ablation is requested explicitly."""
    enabled = cli._parser().parse_args(["run", "--pipeline", "naive"])
    disabled = cli._parser().parse_args(["run", "--pipeline", "naive", "--no-rewrite-query"])
    corrective = cli._parser().parse_args(["run", "--pipeline", "corrective", "--no-rewrite-query"])

    assert enabled.rewrite_query is True
    assert cli._history_resolution_enabled(enabled) is True
    assert cli._query_rewrite_policy(enabled) == "enabled_when_history_present"
    assert disabled.rewrite_query is False
    assert cli._history_resolution_enabled(disabled) is False
    assert cli._history_resolution_enabled(corrective) is True
    assert cli._query_rewrite_policy(corrective) == "intrinsic_when_history_present"


def test_run_rejects_ontology_without_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the ontology backend without paths fails with a clear message."""
    monkeypatch.delenv("ONTOLOGY_PATHS", raising=False)
    arguments = cli._parser().parse_args(["run", "--pipeline", "ontology"])

    with pytest.raises(SystemExit, match="ONTOLOGY_PATHS"):
        cli._build_evaluation_backend(arguments)


def test_graph_evaluation_requires_a_matching_corpus_config(tmp_path: Path) -> None:
    """A Graphiti group cannot be evaluated without a versioned description of its corpus."""
    missing = cli._parser().parse_args(["run", "--pipeline", "graphiti"])
    with pytest.raises(SystemExit, match="corpus-config"):
        cli._validate_graph_corpus_config(missing)

    corpus_config = tmp_path / "corpus.json"
    corpus_config.write_text(
        json.dumps({"group_id": "different-group", "documents": []}),
        encoding="utf-8",
    )
    mismatch = cli._parser().parse_args(["run", "--pipeline", "graphiti", "--corpus-config", str(corpus_config)])
    with pytest.raises(SystemExit, match="does not match"):
        cli._validate_graph_corpus_config(mismatch)


def test_ontology_coverage_reports_partial_gold_provenance_by_bundle() -> None:
    """Ontology comparisons disclose when structured provenance covers only part of the benchmark."""
    records = [
        EvaluationRecord(
            raw={},
            sample=EvaluationSample(
                question="full",
                bundle_kind="single_hop",
                expected_chunk_ids=("chunk-a", "chunk-b"),
            ),
        ),
        EvaluationRecord(
            raw={},
            sample=EvaluationSample(
                question="partial",
                bundle_kind="multi_hop",
                expected_chunk_ids=("chunk-b", "chunk-c"),
            ),
        ),
        EvaluationRecord(raw={}, sample=EvaluationSample(question="no chunk", bundle_kind="multi_hop")),
    ]

    coverage = cli._ontology_coverage_from_chunk_ids(records, frozenset(("chunk-a", "chunk-b")))

    assert coverage["sample_count"] == 3
    assert coverage["samples_with_gold"] == 2
    assert coverage["gold_chunk_reference_count"] == 4
    assert coverage["covered_gold_chunk_reference_count"] == 3
    assert coverage["gold_chunk_reference_coverage"] == 0.75
    assert coverage["unique_gold_chunk_count"] == 3
    assert coverage["covered_unique_gold_chunk_count"] == 2
    assert coverage["sample_any_coverage"] == 1.0
    assert coverage["sample_complete_coverage"] == 0.5
    assert coverage["by_bundle_kind"]["multi_hop"]["sample_count"] == 2
    assert coverage["by_bundle_kind"]["multi_hop"]["sample_complete_coverage"] == 0.0


def test_run_parses_generation_options() -> None:
    """The judge options are only wired when --generation is requested."""
    default_arguments = cli._parser().parse_args(["run", "--pipeline", "naive"])
    arguments = cli._parser().parse_args(
        ["run", "--pipeline", "naive", "--generation", "--judge-batch-size", "3", "--judge-concurrency", "2"]
    )

    assert default_arguments.generation is False
    assert arguments.generation is True
    assert arguments.judge_batch_size == 3
    assert arguments.judge_concurrency == 2


def test_run_rejects_a_non_positive_shared_limit() -> None:
    """Every pipeline receives a positive retrieval budget."""
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["run", "--limit", "0"])


@pytest.mark.parametrize(
    ("command", "handler"),
    [("run", cli._handle_run), ("retrieve", cli._handle_retrieve)],
)
def test_evaluation_rejects_metric_cutoffs_above_the_retrieval_cap(
    command: str,
    handler: object,
) -> None:
    """An @k label cannot claim a cutoff the pipeline was configured never to return."""
    arguments = cli._parser().parse_args([command, "--limit", "3", "--k-values", "1,5"])

    with pytest.raises(SystemExit, match=r"--limit \(3\).*largest --k-values cutoff \(5\)"):
        handler(arguments)  # type: ignore[operator]


@pytest.mark.parametrize(
    ("command", "selector"),
    [("run", ["--pipeline", "naive"]), ("retrieve", ["--retriever", "naive"])],
)
def test_evaluation_rejects_output_path_collisions_before_backend_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    selector: list[str],
) -> None:
    """Predictions and metrics can never overwrite each other under a valid manifest."""
    ground_truth = tmp_path / "groundtruth.jsonl"
    output = tmp_path / "shared.json"
    ground_truth.write_text(json.dumps({"question": "question"}) + "\n", encoding="utf-8")

    def unexpected_backend(*_args: object, **_kwargs: object) -> object:
        raise AssertionError

    monkeypatch.setattr(cli, "_build_preflight_backend", unexpected_backend)
    monkeypatch.setattr(cli, "_retrieve_with_naive", unexpected_backend)
    arguments = cli._parser().parse_args(
        [
            command,
            *selector,
            "--ground-truth",
            str(ground_truth),
            "--predictions-output",
            str(output),
            "--metrics-output",
            str(output),
        ]
    )

    with pytest.raises(SystemExit, match="artifact paths must be distinct"):
        {"run": cli._handle_run, "retrieve": cli._handle_retrieve}[command](arguments)

    assert not output.exists()


@pytest.mark.parametrize(
    ("command", "selector"),
    [("run", ["--pipeline", "naive"]), ("retrieve", ["--retriever", "naive"])],
)
def test_evaluation_rejects_ground_truth_output_collision_before_backend_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    selector: list[str],
) -> None:
    """Evaluation results cannot atomically replace the dataset they were loaded from."""
    ground_truth = tmp_path / "groundtruth.jsonl"
    original = json.dumps({"question": "question"}) + "\n"
    ground_truth.write_text(original, encoding="utf-8")

    def unexpected_backend(*_args: object, **_kwargs: object) -> object:
        raise AssertionError

    monkeypatch.setattr(cli, "_build_preflight_backend", unexpected_backend)
    monkeypatch.setattr(cli, "_retrieve_with_naive", unexpected_backend)
    arguments = cli._parser().parse_args(
        [
            command,
            *selector,
            "--ground-truth",
            str(ground_truth),
            "--predictions-output",
            str(ground_truth),
        ]
    )

    with pytest.raises(SystemExit, match="must not overwrite inputs"):
        {"run": cli._handle_run, "retrieve": cli._handle_retrieve}[command](arguments)

    assert ground_truth.read_text(encoding="utf-8") == original


def test_score_and_compare_reject_overwriting_their_inputs(tmp_path: Path) -> None:
    """Read-only evaluation inputs remain intact when an output path aliases them."""
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"question": "question"}) + "\n", encoding="utf-8")
    score_arguments = cli._parser().parse_args(["score", str(predictions), "--output", str(predictions)])

    with pytest.raises(SystemExit, match="must not overwrite inputs"):
        cli._handle_score(score_arguments)

    first_metrics = tmp_path / "first-metrics.json"
    second_metrics = tmp_path / "second-metrics.json"
    first_metrics.write_text("{}", encoding="utf-8")
    second_metrics.write_text("{}", encoding="utf-8")
    compare_arguments = cli._parser().parse_args(
        ["compare", str(first_metrics), str(second_metrics), "--output", str(first_metrics)]
    )

    with pytest.raises(SystemExit, match="must not overwrite inputs"):
        cli._handle_compare(compare_arguments)


@pytest.mark.parametrize("target_name", ["manifest", "predictions", "generation"])
def test_compare_rejects_overwriting_adjacent_manifest_artifacts(
    tmp_path: Path,
    target_name: str,
) -> None:
    """Comparison output cannot replace a manifest or any artifact it declares."""
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    first_metrics = first_dir / "metrics.json"
    second_metrics = tmp_path / "second-metrics.json"
    first_metrics.write_text("{}", encoding="utf-8")
    second_metrics.write_text("{}", encoding="utf-8")

    predictions = first_dir / "artifacts" / "predictions.jsonl"
    generation = first_dir / "artifacts" / "generation.json"
    predictions.parent.mkdir()
    predictions.write_text("prediction", encoding="utf-8")
    generation.write_text("generation", encoding="utf-8")
    manifest = first_metrics.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "outputs": {
                    "metrics": "metrics.json",
                    "predictions": "artifacts/predictions.jsonl",
                    "generation": "artifacts/generation.json",
                }
            }
        ),
        encoding="utf-8",
    )
    target = {
        "manifest": manifest,
        "predictions": predictions,
        "generation": generation,
    }[target_name]
    original = target.read_text(encoding="utf-8")
    arguments = cli._parser().parse_args(["compare", str(first_metrics), str(second_metrics), "--output", str(target)])

    with pytest.raises(SystemExit, match="must not overwrite inputs"):
        cli._handle_compare(arguments)

    assert target.read_text(encoding="utf-8") == original


def test_evaluation_rejects_outputs_that_overwrite_versioned_runtime_inputs(tmp_path: Path) -> None:
    """Graph corpus manifests and ontology releases are protected like ground truth."""
    ground_truth = tmp_path / "groundtruth.jsonl"
    ground_truth.write_text(json.dumps({"question": "question"}) + "\n", encoding="utf-8")
    corpus_config = tmp_path / "corpus.json"
    corpus_config.write_text("{}", encoding="utf-8")
    graph_arguments = cli._parser().parse_args(
        [
            "retrieve",
            "--retriever",
            "graph",
            "--ground-truth",
            str(ground_truth),
            "--corpus-config",
            str(corpus_config),
            "--metrics-output",
            str(corpus_config),
        ]
    )

    with pytest.raises(SystemExit, match="must not overwrite inputs"):
        cli._handle_retrieve(graph_arguments)

    ontology = tmp_path / "ontology.ttl"
    ontology.write_text("@prefix : <https://example.test/> .", encoding="utf-8")
    ontology_arguments = cli._parser().parse_args(
        [
            "run",
            "--pipeline",
            "ontology",
            "--ground-truth",
            str(ground_truth),
            "--ontology-paths",
            str(ontology),
            "--metrics-output",
            str(ontology),
        ]
    )

    with pytest.raises(SystemExit, match="must not overwrite inputs"):
        cli._handle_run(ontology_arguments)


def test_evaluation_protects_selected_chunk_corpus_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Outputs cannot replace chunk files that define the evaluated corpus."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    selected_chunks = chunks_dir / "selected.chunks.jsonl"
    selected_chunks.write_text(
        json.dumps({"chunk_id": "selected", "source_name": "selected.docx", "text": "Selected"}) + "\n",
        encoding="utf-8",
    )
    unrelated_chunks = chunks_dir / "unrelated.chunks.jsonl"
    unrelated_chunks.write_text(
        json.dumps({"chunk_id": "other", "source_name": "other.docx", "text": "Other"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NAIVE_RAG_CHUNKS_DIR", str(chunks_dir))
    ground_truth = tmp_path / "groundtruth.jsonl"
    ground_truth.write_text(json.dumps({"question": "question"}) + "\n", encoding="utf-8")
    arguments = cli._parser().parse_args(
        [
            "retrieve",
            "--retriever",
            "naive",
            "--ground-truth",
            str(ground_truth),
            "--metrics-output",
            str(selected_chunks),
        ]
    )

    with pytest.raises(SystemExit, match="must not overwrite inputs"):
        cli._handle_retrieve(arguments)

    corpus_config = tmp_path / "corpus.json"
    corpus_config.write_text(
        json.dumps({"group_id": "test-group", "documents": [{"file": "selected.docx"}]}),
        encoding="utf-8",
    )
    graph_arguments = cli._parser().parse_args(
        [
            "run",
            "--pipeline",
            "graphiti",
            "--group-id",
            "test-group",
            "--corpus-config",
            str(corpus_config),
        ]
    )

    graph_inputs = set(cli._evaluation_input_paths(graph_arguments).values())

    assert selected_chunks.resolve() in graph_inputs
    assert unrelated_chunks.resolve() not in graph_inputs


def test_k_values_are_canonicalized_for_stable_identity_and_comparison() -> None:
    """Equivalent cutoff sets cannot create incompatible ordering or duplicate keys."""
    arguments = cli._parser().parse_args(["run", "--k-values", "10,3,3,1"])

    assert arguments.k_values == (1, 3, 10)


def test_run_parses_a_sample_limit_separately_from_retrieval_limit() -> None:
    """Smoke runs bound dataset rows without changing each query's retrieval budget."""
    arguments = cli._parser().parse_args(["run", "--max-samples", "12", "--limit", "7"])

    assert arguments.max_samples == 12
    assert arguments.limit == 7


def test_retrieve_parses_naive_mode_and_sample_limit() -> None:
    """Retrieval-only evaluation exposes every Naive search strategy and bounded smoke tests."""
    arguments = cli._parser().parse_args(
        [
            "retrieve",
            "--retriever",
            "naive",
            "--naive-search-mode",
            "hybrid",
            "--max-samples",
            "12",
        ]
    )

    assert arguments.retriever == "naive"
    assert arguments.naive_search_mode == "hybrid"
    assert arguments.max_samples == 12


def test_compare_writes_a_validated_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The comparison subcommand delegates validation and persists its JSON report."""
    from legal_rag.evaluation import comparison

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "comparison.json"
    expected = {"schema_version": "1.0.0", "compatibility": {"status": "compatible"}, "runs": []}
    captured: dict[str, object] = {}

    def compare(paths: object, *, allow_incompatible: bool) -> dict[str, object]:
        captured["paths"] = paths
        captured["allow_incompatible"] = allow_incompatible
        return expected

    monkeypatch.setattr(comparison, "compare_evaluation_runs", compare)
    arguments = cli._parser().parse_args(["compare", str(first), str(second), "--output", str(output)])

    cli._handle_compare(arguments)

    assert captured == {"paths": [first, second], "allow_incompatible": False}
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_retrieve_accepts_adaptive_without_answer_generation() -> None:
    """Adaptive planning and retrieval are exposed independently of answer generation."""
    arguments = cli._parser().parse_args(["retrieve", "--retriever", "adaptive"])

    assert arguments.retriever == "adaptive"


@pytest.mark.parametrize("mode", ["hybrid_rerank", "hierarchical"])
def test_evaluation_parsers_accept_naive_variants(mode: str) -> None:
    """Both retrieval-only and end-to-end evaluation expose each Naive variant."""
    retrieve = cli._parser().parse_args(["retrieve", "--retriever", "naive", "--naive-search-mode", mode])
    run = cli._parser().parse_args(["run", "--pipeline", "naive", "--naive-search-mode", mode])

    assert retrieve.naive_search_mode == mode
    assert run.naive_search_mode == mode


@pytest.mark.parametrize("retriever", ["naive", "adaptive"])
def test_retrieve_writes_and_replays_identity_addressed_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, retriever: str
) -> None:
    """Bounded retrieval validates its backend and replays only identical completed artifacts."""
    ground_truth = tmp_path / "groundtruth.jsonl"
    ground_truth.write_text(
        "".join(
            json.dumps(
                {
                    "question": f"question {index}",
                    "expected_chunk_ids": [f"chunk-{index}"],
                    "bundle_kind": "single_hop",
                    "answerability": "answerable",
                    "reasoning_hops": 1,
                    "query_style": "standard",
                    "required_claims": [{"claim": f"claim {index}", "evidence": [{"chunk_id": f"chunk-{index}"}]}],
                }
            )
            + "\n"
            for index in (1, 2)
        ),
        encoding="utf-8",
    )

    preflight_calls = 0
    retrieval_calls = 0

    def retrieve(records: list[EvaluationRecord], arguments: argparse.Namespace) -> cli._RetrievalExecution:
        nonlocal preflight_calls, retrieval_calls
        assert arguments.naive_search_mode == "hybrid"
        if not records:
            preflight_calls += 1
            return cli._RetrievalExecution(
                rows=[],
                active_backend_fingerprint={"corpus_sha256": "active-corpus"},
            )
        retrieval_calls += 1
        assert len(records) == 1
        record = records[0]
        return cli._RetrievalExecution(
            rows=[
                cli._retrieval_prediction_row(
                    record.sample,
                    [
                        {
                            "chunk_id": "chunk-1",
                            "source_name": "document.docx",
                            "text": f"context-{retrieval_calls}",
                            "score": 1.0,
                        }
                    ],
                    raw=record.raw,
                )
            ],
            active_backend_fingerprint={"corpus_sha256": "active-corpus"},
        )

    monkeypatch.setattr(cli, f"_retrieve_with_{retriever}", retrieve)
    monkeypatch.chdir(tmp_path)
    arguments = cli._parser().parse_args(
        [
            "retrieve",
            "--retriever",
            retriever,
            "--naive-search-mode",
            "hybrid",
            "--max-samples",
            "1",
            "--ground-truth",
            str(ground_truth),
            "--k-values",
            "1",
            "--reference-date",
            "2026-10-01",
        ]
    )

    cli._handle_retrieve(arguments)

    [run_dir] = (tmp_path / "data/evaluation/runs").iterdir()
    predictions = run_dir / "predictions.jsonl"
    metrics = run_dir / "metrics.json"
    manifest = metrics.with_suffix(".manifest.json")
    assert predictions.exists()
    assert manifest.exists()
    first_artifacts = {path: path.read_bytes() for path in (predictions, metrics, manifest)}

    cli._handle_retrieve(arguments)

    assert preflight_calls == 2
    assert retrieval_calls == 1
    assert {path: path.read_bytes() for path in first_artifacts} == first_artifacts
    report = json.loads(metrics.read_text(encoding="utf-8"))
    manifest_report = json.loads(manifest.read_text(encoding="utf-8"))
    assert report["retriever"] == retriever
    assert report["pipeline"] == retriever
    assert report["search_mode"] == ("hybrid" if retriever == "naive" else None)
    assert report["sample_count"] == 1
    assert report["claim_sample_count"] == 1
    assert report["claim_evidence_complete_at_k"] == {"1": 1.0}
    assert report["recall_at_k"] == {"1": 1.0}
    assert report["slices"]["bundle_kind"]["single_hop"]["sample_count"] == 1
    assert report["slices"]["answerability"]["answerable"]["sample_count"] == 1
    assert report["slices"]["reasoning_hops"]["1"]["sample_count"] == 1
    assert report["slices"]["query_style"]["standard"]["sample_count"] == 1
    assert manifest_report["status"] == "complete"
    assert manifest_report["run_id"] == report["run_id"]
    assert manifest_report["identity"]["naive_search_mode"] == ("hybrid" if retriever == "naive" else None)
    if retriever == "adaptive":
        assert "legal_rag.adaptive_rag" in manifest_report["identity"]["source_sha256"]
        assert report["effective_config"]["adaptive_rag"]["base_search_mode"] == "hybrid_rerank"
        assert report["effective_config"]["planner_prompt_sha256"] == cli._baml_runtime_source_sha256(
            "adaptive_query_planning.baml"
        )
    assert manifest_report["output_sha256"] == {
        "predictions": cli._sha256_file(predictions),
        "metrics": cli._sha256_file(metrics),
    }
    assert report["active_backend_validated"] is True
    assert report["active_backend_fingerprint"]["corpus_sha256"] == "active-corpus"
    assert report["reference_date"] == "2026-10-01"


def test_graph_retrieval_defaults_do_not_overwrite_vector_or_graph_ablation_outputs() -> None:
    """Identity-addressed defaults isolate every retrieval experiment."""
    graph = cli._parser().parse_args(
        [
            "retrieve",
            "--retriever",
            "graph",
            "--group-id",
            "traffic/penalty",
            "--cross-encoder",
            "--no-graph-expansion",
        ]
    )
    vector = cli._parser().parse_args(["retrieve", "--retriever", "vector"])

    graph_predictions, graph_metrics = cli._retrieval_output_paths(graph, "graph-run")
    vector_predictions, vector_metrics = cli._retrieval_output_paths(vector, "vector-run")

    assert graph_predictions == Path("data/evaluation/runs/graph-run/predictions.jsonl")
    assert graph_metrics == Path("data/evaluation/runs/graph-run/metrics.json")
    assert vector_predictions == Path("data/evaluation/runs/vector-run/predictions.jsonl")
    assert vector_metrics == Path("data/evaluation/runs/vector-run/metrics.json")


def test_retrieval_replay_refuses_orphaned_output(tmp_path: Path) -> None:
    """A partial prior write is never replaced without an identity-attesting manifest."""
    predictions = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    predictions.write_text("existing\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        cli._replay_completed_retrieval(
            metrics.with_suffix(".manifest.json"),
            predictions,
            metrics,
            {"identity": "test"},
            "run-id",
        )

    assert predictions.read_text(encoding="utf-8") == "existing\n"


def test_retrieve_with_naive_forwards_mode_and_closes_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Naive adapter returns scored chunk dictionaries and releases its Milvus client."""
    from types import SimpleNamespace

    from legal_rag.datastores import milvus as milvus_module
    from legal_rag.embedding import encoder as encoder_module
    from legal_rag.embedding import profiles as profiles_module
    from legal_rag.naive_rag import retriever as retriever_module
    from legal_rag.settings import MilvusSettings, NaiveRagSettings

    fake_settings = SimpleNamespace(embedding_profile="test-profile")

    class FakeRetriever:
        """Record adapter calls without loading an embedding model or Milvus."""

        def __init__(self, encoder: object, client: object, settings: object) -> None:
            assert encoder == "encoder"
            assert client == "milvus-client"
            assert settings is fake_settings
            self.closed = False
            fake_retrievers.append(self)

        def retrieve(self, question: str, top_k: int, mode: retriever_module.SearchMode) -> list[object]:
            assert question == "question"
            assert top_k == 3
            assert mode.value == "keyword"
            return [
                SimpleNamespace(
                    chunk_id="chunk-1",
                    source_name="document.docx",
                    text="context",
                    kind="paragraph_group",
                    score=2.5,
                )
            ]

        def preflight_corpus(self) -> object:
            """Return a manifest-ready active collection fingerprint."""
            return SimpleNamespace(to_dict=lambda: {"corpus_sha256": "active-corpus"})

        def close(self) -> None:
            self.closed = True

    fake_retrievers: list[FakeRetriever] = []
    monkeypatch.setattr(NaiveRagSettings, "from_environment", classmethod(lambda _cls: fake_settings))
    monkeypatch.setattr(MilvusSettings, "from_environment", classmethod(lambda _cls: "milvus-settings"))
    monkeypatch.setattr(profiles_module, "load_configured_embedding_profile", lambda _profile: "profile")
    monkeypatch.setattr(encoder_module.EmbeddingEncoder, "from_profile", classmethod(lambda _cls, _profile: "encoder"))
    monkeypatch.setattr(milvus_module, "build_milvus_client", lambda _settings: "milvus-client")
    monkeypatch.setattr(retriever_module, "Retriever", FakeRetriever)
    arguments = cli._parser().parse_args(
        ["retrieve", "--retriever", "naive", "--naive-search-mode", "keyword", "--limit", "3"]
    )

    execution = cli._retrieve_with_naive([EvaluationSample(question="question")], arguments)
    [row] = execution.rows

    assert row["retrieved_chunks"] == [
        {
            "chunk_id": "chunk-1",
            "source_name": "document.docx",
            "text": "context",
            "kind": "paragraph_group",
            "score": 2.5,
            "unit_id": "chunk-1",
            "unit_kind": "chunk",
            "evidence": [{"chunk_id": "chunk-1", "source_name": "document.docx"}],
        }
    ]
    assert row["retrieved_units"][0]["unit_id"] == "chunk-1"
    assert execution.active_backend_fingerprint == {"corpus_sha256": "active-corpus"}
    assert fake_retrievers[0].closed is True


def test_retrieve_with_adaptive_preserves_plan_metadata_and_closes_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adaptive retrieval emits auditable plans and never invokes its full run method."""
    from types import SimpleNamespace

    from legal_rag.adaptive_rag import pipeline as pipeline_module
    from legal_rag.adaptive_rag.models import AdaptiveRetrievalResult, AdaptiveRoute
    from legal_rag.settings import AdaptiveRagSettings, MilvusSettings, NaiveRagSettings, Settings

    class FakePipeline:
        """Return one retrieval result while rejecting accidental answer generation."""

        def __init__(self, *settings: object) -> None:
            self.settings = SimpleNamespace(base_search_mode="hybrid_rerank")
            self.closed = False
            pipelines.append(self)

        def retrieve(self, question: str, top_k: int, rewrite_query: bool) -> AdaptiveRetrievalResult:
            assert (question, top_k, rewrite_query) == ("question", 3, False)
            return AdaptiveRetrievalResult(
                query=question,
                sources=(
                    SimpleNamespace(
                        chunk_id="chunk-1",
                        source_name="document.docx",
                        text="context",
                        kind="paragraph_group",
                        score=0.5,
                        effective_unit_id="chunk-1",
                        unit_kind="chunk",
                        effective_evidence_chunk_ids=("chunk-1",),
                    ),
                ),
                route=AdaptiveRoute.MULTI_HOP,
                subqueries=("subquery one", "subquery two"),
            )

        def preflight_corpus(self) -> object:
            """Return the validated collection identity."""
            return SimpleNamespace(to_dict=lambda: {"corpus_sha256": "active-corpus"})

        def close(self) -> None:
            self.closed = True

    pipelines: list[FakePipeline] = []
    monkeypatch.setattr(pipeline_module, "AdaptiveRAGPipeline", FakePipeline)
    monkeypatch.setattr(Settings, "from_environment", classmethod(lambda _cls: object()))
    monkeypatch.setattr(MilvusSettings, "from_environment", classmethod(lambda _cls: object()))
    monkeypatch.setattr(NaiveRagSettings, "from_environment", classmethod(lambda _cls: object()))
    monkeypatch.setattr(AdaptiveRagSettings, "from_environment", classmethod(lambda _cls: object()))
    arguments = cli._parser().parse_args(["retrieve", "--retriever", "adaptive", "--limit", "3"])

    preflight = cli._retrieve_with_adaptive((), arguments)
    assert preflight.rows == []
    execution = cli._retrieve_with_adaptive(
        [
            EvaluationRecord(
                raw={"question_id": "q-1", "question": "question"}, sample=EvaluationSample(question="question")
            )
        ],
        arguments,
    )
    [row] = execution.rows
    assert row["question_id"] == "q-1"
    assert row["evaluation_metadata"]["latency_ms"] >= 0
    assert execution.active_backend_fingerprint == preflight.active_backend_fingerprint

    assert row["retrieved_chunk_ids"] == ["chunk-1"]
    assert row["pipeline_metadata"] == {
        "base_search_mode": "hybrid_rerank",
        "route": "multi_hop",
        "subqueries": ["subquery one", "subquery two"],
    }
    assert all(pipeline.closed for pipeline in pipelines)


def test_retrieval_prediction_row_preserves_grouped_evidence() -> None:
    """Retrieval-only output keeps one ranked article unit and flattened compatibility IDs."""
    row = cli._retrieval_prediction_row(
        EvaluationSample(question="question", expected_chunk_ids=("chunk-1", "chunk-2")),
        [
            {
                "chunk_id": "chunk-1",
                "unit_id": "article:1",
                "unit_kind": "article",
                "source_name": "document.docx",
                "text": "expanded",
                "score": 0.5,
                "evidence": [
                    {"chunk_id": "chunk-1", "source_name": "document.docx"},
                    {"chunk_id": "chunk-2", "source_name": "document.docx"},
                ],
            }
        ],
    )

    assert row["retrieved_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert row["retrieved_units"][0]["unit_id"] == "article:1"
    assert len(row["retrieved_units"][0]["evidence"]) == 2


def test_graph_fact_mapping_preserves_one_ranked_fact_with_all_provenance() -> None:
    """Graph retrieval-only evaluation uses the same native-unit semantics as end-to-end evaluation."""
    from legal_rag.graphiti_rag.retriever import FactSource, RetrievedFact

    mapped = cli._graph_fact_mapping(
        RetrievedFact(
            "fact",
            (
                FactSource("chunk-1", "one.docx"),
                FactSource("chunk-2", "two.docx"),
            ),
        )
    )
    row = cli._retrieval_prediction_row(
        EvaluationSample(question="question", expected_chunk_ids=("chunk-1", "chunk-2")),
        [mapped],
        latency_ms=12.5,
    )

    assert row["retrieved_units"][0]["unit_kind"] == "graph_fact"
    assert row["retrieved_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert row["evaluation_metadata"] == {
        "latency_ms": 12.5,
        "retrieved_unit_count": 1,
        "unique_evidence_chunk_count": 2,
    }


def test_retrieve_with_graph_matches_pipeline_candidate_pool_then_applies_unit_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Retrieval-only Graphiti uses the same expansion pool and final fact budget as the pipeline."""
    from types import SimpleNamespace

    from legal_rag.datastores import neo4j as neo4j_module
    from legal_rag.embedding import profiles as profiles_module
    from legal_rag.graphiti_rag import client as client_module
    from legal_rag.graphiti_rag import index_metadata as index_metadata_module
    from legal_rag.graphiti_rag import indexer as indexer_module
    from legal_rag.graphiti_rag import retriever as retriever_module
    from legal_rag.graphiti_rag import snapshot as snapshot_module
    from legal_rag.naive_rag import indexer as naive_indexer_module
    from legal_rag.settings import GraphitiSettings, Neo4jSettings, Settings

    calls: list[dict[str, object]] = []
    snapshot_calls: list[dict[str, object]] = []

    class FakeGraphiti:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeNeo4jDriver:
        """Record direct snapshot connection cleanup."""

        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeSnapshot:
        """Return one valid deterministic group fingerprint."""

        def validate_for_evaluation(self) -> None:
            return None

        def to_dict(self) -> dict[str, object]:
            return {"group_id": "traffic_penalty_groundtruth_v1", "sha256": "graph-digest"}

    async def search_facts(
        _client: object,
        _question: str,
        _group_id: str,
        num_results: int,
        **options: object,
    ) -> list[retriever_module.RetrievedFact]:
        calls.append({"num_results": num_results, **options})
        return [
            retriever_module.RetrievedFact(
                f"fact {index}",
                (retriever_module.FactSource(f"chunk-{index}", "document.docx"),),
            )
            for index in range(3)
        ]

    async def snapshot_graphiti_group(*_args: object, **kwargs: object) -> FakeSnapshot:
        """Return the valid fake active-group snapshot."""
        snapshot_calls.append(kwargs)
        return FakeSnapshot()

    async def validate_graphiti_index_metadata(*_args: object) -> index_metadata_module.GraphitiIndexMetadata:
        """Return the embedding identity attested before graph retrieval."""
        return index_metadata_module.GraphitiIndexMetadata(
            metadata_schema_version="1.0.0",
            embedding_preprocessing_version="1.0.0",
            embedding_profile_id="profile",
            embedding_model_id="model",
            embedding_model_revision="revision",
            embedding_dimensions=1024,
        )

    fake_client = FakeGraphiti()
    fake_driver = FakeNeo4jDriver()
    corpus_config = tmp_path / "corpus.json"
    corpus_config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Settings, "from_environment", classmethod(lambda _cls: "settings"))
    monkeypatch.setattr(Neo4jSettings, "from_environment", classmethod(lambda _cls: "neo4j"))
    monkeypatch.setattr(
        GraphitiSettings,
        "from_environment",
        classmethod(lambda _cls: SimpleNamespace(embedding_profile="profile", search_num_results=30)),
    )
    monkeypatch.setattr(profiles_module, "load_configured_embedding_profile", lambda _profile: "embedding")
    monkeypatch.setattr(client_module, "build_graphiti", lambda *_args: fake_client)
    monkeypatch.setattr(
        index_metadata_module,
        "validate_graphiti_index_metadata",
        validate_graphiti_index_metadata,
    )
    monkeypatch.setattr(neo4j_module, "build_neo4j_driver", lambda _settings: fake_driver)
    monkeypatch.setattr(
        indexer_module,
        "load_corpus_metadata",
        lambda _path: SimpleNamespace(documents={"document.docx": SimpleNamespace(effective_date=date(2024, 7, 1))}),
    )
    monkeypatch.setattr(
        naive_indexer_module,
        "load_chunks",
        lambda _path: [SimpleNamespace(source_name="document.docx")],
    )
    monkeypatch.setattr(
        snapshot_module,
        "snapshot_graphiti_group",
        snapshot_graphiti_group,
    )
    monkeypatch.setattr(retriever_module, "search_facts", search_facts)
    arguments = cli._parser().parse_args(
        [
            "retrieve",
            "--retriever",
            "graph",
            "--limit",
            "2",
            "--corpus-config",
            str(corpus_config),
            "--reference-date",
            "2026-10-01",
        ]
    )

    record = EvaluationRecord(
        raw={"question_id": "q-1", "question": "question", "query_date": "2025-06-01"},
        sample=EvaluationSample(question="question"),
        query_date=date(2025, 6, 1),
    )
    execution = cli._retrieve_with_graph([record], arguments)
    [row] = execution.rows

    assert calls == [
        {
            "num_results": 30,
            "expand_around_top_fact": True,
            "rerank_with_cross_encoder": False,
            "as_of_date": date(2025, 6, 1),
        }
    ]
    assert len(snapshot_calls) == 1
    assert len(snapshot_calls[0]["expected_chunks"]) == 1
    assert snapshot_calls[0]["expected_source_dates"] == {"document.docx": date(2024, 7, 1)}
    assert len(row["retrieved_units"]) == 2
    assert row["question_id"] == "q-1"
    assert row["query_date"] == "2025-06-01"
    assert fake_client.closed is True
    assert fake_driver.closed is True
    assert execution.active_backend_fingerprint["sha256"] == "graph-digest"
    assert execution.active_backend_fingerprint["embedding_metadata"]["embedding_profile_id"] == "profile"


def test_run_writes_pipeline_specific_predictions_and_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The common CLI scores normalized output without backend-specific metric paths."""
    ground_truth = tmp_path / "groundtruth.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    ground_truth.write_text(
        json.dumps(
            {
                "question_id": "q-1",
                "question": "question",
                "expected_answer": "answer",
                "expected_chunk_ids": ["chunk-1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FakeBackend()
    monkeypatch.setattr(cli, "_build_evaluation_backend", lambda _arguments: backend)
    arguments = cli._parser().parse_args(
        [
            "run",
            "--pipeline",
            "naive",
            "--ground-truth",
            str(ground_truth),
            "--predictions-output",
            str(predictions),
            "--metrics-output",
            str(metrics),
            "--run-dir",
            str(tmp_path / "run"),
            "--k-values",
            "1",
        ]
    )

    cli._handle_run(arguments)

    [prediction] = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    metric_report = json.loads(metrics.read_text(encoding="utf-8"))
    assert prediction["pipeline"] == "naive"
    assert prediction["question_id"] == "q-1"
    assert metric_report["pipeline"] == "naive"
    assert metric_report["precision_at_k"] == {"1": 1.0}
    assert "mean_reciprocal_rank" not in metric_report
    assert backend.closed is True
    run_manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    sidecar_manifest = json.loads(metrics.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert sidecar_manifest == run_manifest

    def fail_if_backend_is_built(_arguments: argparse.Namespace) -> _FakeBackend:
        raise AssertionError

    monkeypatch.setattr(cli, "_build_evaluation_backend", fail_if_backend_is_built)
    arguments.resume = True
    cli._handle_run(arguments)

    changed_metrics = tmp_path / "changed-metrics.json"
    arguments.metrics_output = changed_metrics
    with pytest.raises(SystemExit, match="different metrics output"):
        cli._handle_run(arguments)
    assert not changed_metrics.exists()

    arguments.metrics_output = metrics
    arguments.limit = 2
    with pytest.raises(SystemExit, match="different experiment"):
        cli._handle_run(arguments)


def test_run_sample_limit_is_frozen_in_manifest_and_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A resumable smoke run evaluates exactly the selected deterministic prefix."""
    ground_truth = tmp_path / "groundtruth.jsonl"
    run_dir = tmp_path / "run"
    ground_truth.write_text(
        "".join(json.dumps({"question_id": f"q-{index}", "question": f"question {index}"}) + "\n" for index in (1, 2)),
        encoding="utf-8",
    )
    backend = _FakeBackend()
    monkeypatch.setattr(cli, "_build_evaluation_backend", lambda _arguments: backend)
    arguments = cli._parser().parse_args(
        [
            "run",
            "--pipeline",
            "naive",
            "--ground-truth",
            str(ground_truth),
            "--run-dir",
            str(run_dir),
            "--max-samples",
            "1",
        ]
    )

    cli._handle_run(arguments)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert backend.calls == ["question 1"]
    assert len(rows) == 1
    assert manifest["identity"]["max_samples"] == 1


def test_run_with_generation_merges_judge_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--generation judges the answers and merges aggregate judge metrics into the metrics report."""
    ground_truth = tmp_path / "groundtruth.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    generation_report = tmp_path / "generation.json"
    ground_truth.write_text(
        json.dumps(
            {
                "question": "question",
                "expected_answer": "answer",
                "expected_chunk_ids": ["chunk-1"],
                "query_date": "2025-01-01",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_build_evaluation_backend", lambda _arguments: _FakeBackend())
    monkeypatch.setattr("legal_rag.evaluation.generation.judge.BamlJudgeClient", _FakeJudgeClient)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    arguments = cli._parser().parse_args(
        [
            "run",
            "--pipeline",
            "naive",
            "--ground-truth",
            str(ground_truth),
            "--predictions-output",
            str(predictions),
            "--metrics-output",
            str(metrics),
            "--generation-output",
            str(generation_report),
            "--run-dir",
            str(tmp_path / "run"),
            "--generation",
            "--reference-date",
            "2026-08-31",
            "--k-values",
            "1",
        ]
    )

    cli._handle_run(arguments)

    metric_report = json.loads(metrics.read_text(encoding="utf-8"))
    judge_report = json.loads(generation_report.read_text(encoding="utf-8"))
    assert metric_report["generation"] == {
        "answer_relevancy": 5.0,
        "answerability_correctness": 1.0,
        "correctness": 5.0,
        "faithfulness": 1.0,
        "required_claim_coverage": 1.0,
        "temporal_correctness": 1.0,
    }
    assert metric_report["generation_metric_coverage"] == {
        "answer_relevancy": {"coverage": 1.0, "null_count": 0, "scored_count": 1},
        "answerability_correctness": {"coverage": 1.0, "null_count": 0, "scored_count": 1},
        "correctness": {"coverage": 1.0, "null_count": 0, "scored_count": 1},
        "faithfulness": {"coverage": 1.0, "null_count": 0, "scored_count": 1},
        "required_claim_coverage": {"coverage": 1.0, "null_count": 0, "scored_count": 1},
        "temporal_correctness": {"coverage": 1.0, "null_count": 0, "scored_count": 1},
    }
    assert judge_report["sample_count"] == 1
    assert judge_report["per_sample"] == [
        {
            "sample_id": cli._record_id({"question": "question"}),
            "answer_relevancy": 5.0,
            "answerability_correctness": 1.0,
            "correctness": 5.0,
            "faithfulness": 1.0,
            "required_claim_coverage": 1.0,
            "temporal_correctness": 1.0,
        }
    ]
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_sha256"] == {
        "predictions": cli._sha256_file(predictions),
        "metrics": cli._sha256_file(metrics),
        "generation": cli._sha256_file(generation_report),
    }
    assert "legal_rag.evaluation.generation" in manifest["identity"]["source_sha256"]
    [judged_sample] = _FakeJudgeClient.last_samples
    assert judged_sample.query_date.isoformat() == "2025-01-01"
    assert judged_sample.reference_date.isoformat() == "2026-08-31"


def test_generation_scoring_uses_query_date_then_frozen_reference_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generation samples keep row dates while every sample receives the campaign date."""
    monkeypatch.setattr("legal_rag.evaluation.generation.judge.BamlJudgeClient", _FakeJudgeClient)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    arguments = cli._parser().parse_args(["run", "--generation", "--reference-date", "2026-08-31"])
    _FakeJudgeClient.last_samples = []

    cli._score_generation(
        [
            {
                "question": "dated",
                "actual_answer": "answer",
                "query_date": "2025-01-01",
                "query_dates": ["2025-01-01", "2026-03-01"],
            },
            {"question": "current", "actual_answer": "answer"},
        ],
        arguments,
    )

    dated, current = _FakeJudgeClient.last_samples
    assert dated.query_date == date(2025, 1, 1)
    assert dated.query_dates == (date(2025, 1, 1), date(2026, 3, 1))
    assert current.query_date is None
    assert dated.reference_date == current.reference_date == date(2026, 8, 31)


def test_run_resume_skips_checkpointed_rows_and_restores_dataset_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A compatible run resumes only missing records and writes final rows in source order."""
    ground_truth = tmp_path / "groundtruth.jsonl"
    run_dir = tmp_path / "run"
    ground_truth.write_text(
        "".join(
            json.dumps(
                {
                    "question_id": f"q-{index}",
                    "question": f"question {index}",
                    "expected_chunk_ids": [f"chunk-{index}"],
                }
            )
            + "\n"
            for index in (1, 2)
        ),
        encoding="utf-8",
    )
    arguments = cli._parser().parse_args(
        [
            "run",
            "--pipeline",
            "naive",
            "--ground-truth",
            str(ground_truth),
            "--run-dir",
            str(run_dir),
            "--resume",
            "--k-values",
            "1",
        ]
    )
    identity = cli._run_identity(arguments, ground_truth)
    identity["active_backend_fingerprint"] = {
        "fingerprint_version": "test",
        "corpus_sha256": "active-corpus",
    }
    identity["active_backend_validated"] = True
    cli._prepare_run(
        run_dir / "manifest.json",
        run_dir / "predictions.progress.jsonl",
        identity,
        cli._identity_hash(identity),
        resume=False,
    )
    checkpoint_record = cli._load_run_records(ground_truth, max_samples=None)[1]
    (run_dir / "predictions.progress.jsonl").write_text(
        json.dumps(
            {
                "evaluation_record_id": "q-2",
                "evaluation_workload_sha256": cli._evaluation_workload_sha256(
                    checkpoint_record.raw,
                    "naive",
                ),
                "prediction_schema_version": "2.0.0",
                "pipeline": "naive",
                "question_id": "q-2",
                "question": "question 2",
                "expected_chunk_ids": ["chunk-2"],
                "actual_answer": "answer",
                "retrieved_chunk_ids": ["chunk-2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FakeBackend()
    monkeypatch.setattr(cli, "_build_evaluation_backend", lambda _arguments: backend)

    cli._handle_run(arguments)

    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert backend.calls == ["question 1"]
    assert [row["question_id"] for row in rows] == ["q-1", "q-2"]
    assert manifest["status"] == "complete"
    assert "started_at" in manifest
    assert "completed_at" in manifest


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"evaluation_record_id": "unexpected"}, "Unexpected evaluation record"),
        ({"evaluation_workload_sha256": "stale"}, "does not match the current evaluation input"),
        ({"pipeline": "graphiti"}, "different pipeline"),
        ({"prediction_schema_version": "1.0.0"}, "incompatible prediction schema"),
    ],
)
def test_read_progress_rejects_unbound_or_incompatible_rows(
    tmp_path: Path,
    changes: dict[str, str],
    message: str,
) -> None:
    """A stale checkpoint cannot suppress work merely by claiming an expected record ID."""
    progress = tmp_path / "predictions.progress.jsonl"
    input_row = {"question_id": "q-1", "question": "question"}
    workload_sha256 = cli._evaluation_workload_sha256(input_row, "naive")
    checkpoint = {
        **input_row,
        "evaluation_record_id": "q-1",
        "evaluation_workload_sha256": workload_sha256,
        "prediction_schema_version": "2.0.0",
        "pipeline": "naive",
        "actual_answer": "answer",
    }
    checkpoint.update(changes)
    progress.write_text(json.dumps(checkpoint) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        cli._read_progress(
            progress,
            expected_workloads={"q-1": workload_sha256},
            pipeline="naive",
        )


def test_prepare_run_rejects_existing_or_mismatched_experiments(tmp_path: Path) -> None:
    """Run directories cannot silently overwrite results or mix experiment configurations."""
    manifest = tmp_path / "run" / "manifest.json"
    progress = tmp_path / "run" / "predictions.progress.jsonl"
    cli._prepare_run(manifest, progress, {"pipeline": "naive"}, "run-1", resume=False)

    with pytest.raises(SystemExit, match="pass --resume"):
        cli._prepare_run(manifest, progress, {"pipeline": "naive"}, "run-1", resume=False)
    with pytest.raises(SystemExit, match="different experiment"):
        cli._prepare_run(manifest, progress, {"pipeline": "graphiti"}, "run-2", resume=True)


@pytest.mark.parametrize(
    ("field", "mismatched_value", "message"),
    [
        ("run_id", "run-2", "metrics ID"),
        ("active_backend_validated", False, "backend validation"),
        ("active_backend_fingerprint", {"corpus_sha256": "other"}, "backend fingerprint"),
    ],
)
def test_completed_metrics_must_match_manifest_identity(
    tmp_path: Path,
    field: str,
    mismatched_value: object,
    message: str,
) -> None:
    """Resume cannot replay a metrics file produced by another run or active backend."""
    metrics_path = tmp_path / "metrics.json"
    metrics = {
        "run_id": "run-1",
        "active_backend_validated": True,
        "active_backend_fingerprint": {"corpus_sha256": "expected"},
    }
    metrics[field] = mismatched_value
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "run_id": "run-1",
        "identity": {
            "active_backend_validated": True,
            "active_backend_fingerprint": {"corpus_sha256": "expected"},
        },
        "outputs": {"metrics": str(metrics_path), "predictions": str(predictions_path)},
        "output_sha256": {
            "metrics": cli._sha256_file(metrics_path),
            "predictions": cli._sha256_file(predictions_path),
        },
    }

    with pytest.raises(SystemExit, match=message):
        cli._completed_metrics(manifest, tmp_path / "manifest.json", metrics_path)


def test_completed_metrics_rejects_tampered_output(tmp_path: Path) -> None:
    """Resume verifies completed output bytes before replaying cached metrics."""
    metrics_path = tmp_path / "metrics.json"
    predictions_path = tmp_path / "predictions.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "active_backend_validated": True,
                "active_backend_fingerprint": {"corpus_sha256": "expected"},
            }
        ),
        encoding="utf-8",
    )
    predictions_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "run_id": "run-1",
        "identity": {
            "active_backend_validated": True,
            "active_backend_fingerprint": {"corpus_sha256": "expected"},
        },
        "outputs": {"metrics": str(metrics_path), "predictions": str(predictions_path)},
        "output_sha256": {
            "metrics": cli._sha256_file(metrics_path),
            "predictions": cli._sha256_file(predictions_path),
        },
    }
    predictions_path.write_text('{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(SystemExit, match=r"predictions output does not match.*SHA-256"):
        cli._completed_metrics(manifest, tmp_path / "manifest.json", metrics_path)


def test_record_id_distinguishes_turns_even_when_the_parent_question_id_is_shared() -> None:
    """Nested conversations checkpoint each turn independently."""
    first = cli._record_id({"question_id": "parent", "conversation_id": "conversation-1", "turn_number": 1})
    second = cli._record_id({"question_id": "parent", "conversation_id": "conversation-1", "turn_number": 2})

    assert first == "conversation-1:turn:1"
    assert second == "conversation-1:turn:2"


def test_failed_run_is_recorded_in_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An interrupted or failed pipeline remains explicitly resumable instead of looking active."""
    ground_truth = tmp_path / "groundtruth.jsonl"
    run_dir = tmp_path / "run"
    ground_truth.write_text(json.dumps({"question_id": "q-1", "question": "question"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_evaluation_backend", lambda _arguments: _FailingBackend())
    arguments = cli._parser().parse_args(
        ["run", "--pipeline", "naive", "--ground-truth", str(ground_truth), "--run-dir", str(run_dir)]
    )

    with pytest.raises(_PipelineFailureError):
        cli._handle_run(arguments)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "_PipelineFailureError"
    assert "failed_at" in manifest


class _FakeJudgeClient:
    """Return perfect generation scores for every judged batch."""

    last_samples: ClassVar[list[Any]] = []

    def __init__(self, _settings: object) -> None:
        return None

    async def score_batch(self, samples: object) -> list[dict[str, float | None]]:
        """Score every sample with the full metric set."""
        type(self).last_samples = list(samples)  # type: ignore[arg-type]
        return [
            {
                "faithfulness": 1.0,
                "correctness": 5.0,
                "answer_relevancy": 5.0,
                "temporal_correctness": 1.0,
                "answerability_correctness": 1.0,
                "required_claim_coverage": 1.0,
            }
            for _ in samples  # type: ignore[arg-type]
        ]


class _FakeBackend:
    """Return one perfect prediction through the shared backend protocol."""

    name = "naive"

    def __init__(self) -> None:
        self.closed = False
        self.calls: list[str] = []

    async def predict(self, question: str, limit: int) -> PipelinePrediction:
        """Return a result matching the test ground truth."""
        self.calls.append(question)
        return PipelinePrediction(
            answer="answer",
            retrieved_units=(
                RetrievedUnit(
                    unit_id="chunk-1",
                    unit_kind="chunk",
                    text="context",
                    evidence=(PredictionEvidence(chunk_id="chunk-1", source_name="document.docx"),),
                    score=1.0,
                ),
            ),
            retrieved_contexts=(PredictionContext(text="context", chunk_id="chunk-1"),),
            metadata={},
        )

    async def preflight_corpus(self) -> dict[str, object]:
        """Return one deterministic validated datastore identity."""
        return {"fingerprint_version": "test", "corpus_sha256": "active-corpus"}

    async def close(self) -> None:
        """Record cleanup."""
        self.closed = True


class _FailingBackend:
    """Raise after startup to exercise failed-run persistence."""

    name = "naive"

    async def preflight_corpus(self) -> dict[str, object]:
        """Let the run reach the synthetic prediction failure."""
        return {"fingerprint_version": "test", "corpus_sha256": "active-corpus"}

    async def predict(self, question: str, limit: int) -> PipelinePrediction:
        """Simulate a pipeline failure."""
        raise _PipelineFailureError

    async def close(self) -> None:
        """Release the fake backend."""


class _PipelineFailureError(RuntimeError):
    """Synthetic pipeline failure used by manifest tests."""
