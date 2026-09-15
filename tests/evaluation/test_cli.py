"""Tests for the shared Legal RAG evaluation CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from legal_rag.evaluation import cli
from legal_rag.evaluation.metrics import EvaluationSample
from legal_rag.evaluation.runner import (
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


def test_run_rejects_ontology_without_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the ontology backend without paths fails with a clear message."""
    monkeypatch.delenv("ONTOLOGY_PATHS", raising=False)
    arguments = cli._parser().parse_args(["run", "--pipeline", "ontology"])

    with pytest.raises(SystemExit, match="ONTOLOGY_PATHS"):
        cli._build_evaluation_backend(arguments)


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


@pytest.mark.parametrize("mode", ["hybrid_rerank", "hierarchical"])
def test_evaluation_parsers_accept_naive_variants(mode: str) -> None:
    """Both retrieval-only and end-to-end evaluation expose each Naive variant."""
    retrieve = cli._parser().parse_args(["retrieve", "--retriever", "naive", "--naive-search-mode", mode])
    run = cli._parser().parse_args(["run", "--pipeline", "naive", "--naive-search-mode", mode])

    assert retrieve.naive_search_mode == mode
    assert run.naive_search_mode == mode


def test_retrieve_runs_naive_without_llm_and_uses_mode_specific_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bounded Naive run dispatches directly to retrieval and does not overwrite another mode."""
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

    def retrieve(samples: list[EvaluationSample], arguments: argparse.Namespace) -> list[dict[str, object]]:
        assert len(samples) == 1
        assert arguments.naive_search_mode == "hybrid"
        sample = samples[0]
        return [
            cli._retrieval_prediction_row(
                sample,
                [{"chunk_id": "chunk-1", "source_name": "document.docx", "text": "context", "score": 1.0}],
            )
        ]

    monkeypatch.setattr(cli, "_retrieve_with_naive", retrieve)
    monkeypatch.chdir(tmp_path)
    arguments = cli._parser().parse_args(
        [
            "retrieve",
            "--retriever",
            "naive",
            "--naive-search-mode",
            "hybrid",
            "--max-samples",
            "1",
            "--ground-truth",
            str(ground_truth),
            "--k-values",
            "1",
        ]
    )

    cli._handle_retrieve(arguments)

    predictions = tmp_path / "data/evaluation/naive_hybrid_retrieval_predictions.jsonl"
    metrics = tmp_path / "data/evaluation/naive_hybrid_retrieval_metrics.json"
    assert predictions.exists()
    report = json.loads(metrics.read_text(encoding="utf-8"))
    assert report["retriever"] == "naive"
    assert report["search_mode"] == "hybrid"
    assert report["sample_count"] == 1
    assert report["claim_sample_count"] == 1
    assert report["claim_evidence_complete_at_k"] == {"1": 1.0}
    assert report["recall_at_k"] == {"1": 1.0}
    assert report["slices"]["bundle_kind"]["single_hop"]["sample_count"] == 1
    assert report["slices"]["answerability"]["answerable"]["sample_count"] == 1
    assert report["slices"]["reasoning_hops"]["1"]["sample_count"] == 1
    assert report["slices"]["query_style"]["standard"]["sample_count"] == 1


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

    [row] = cli._retrieve_with_naive([EvaluationSample(question="question")], arguments)

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
    assert fake_retrievers[0].closed is True


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

    completed_backend = _FakeBackend()
    monkeypatch.setattr(cli, "_build_evaluation_backend", lambda _arguments: completed_backend)
    arguments.resume = True
    cli._handle_run(arguments)

    assert completed_backend.calls == []
    assert completed_backend.closed is False


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
            "--k-values",
            "1",
        ]
    )

    cli._handle_run(arguments)

    metric_report = json.loads(metrics.read_text(encoding="utf-8"))
    judge_report = json.loads(generation_report.read_text(encoding="utf-8"))
    assert metric_report["generation"] == {
        "answer_relevancy": 5.0,
        "correctness": 5.0,
        "faithfulness": 1.0,
        "temporal_correctness": 1.0,
    }
    assert judge_report["sample_count"] == 1
    assert judge_report["per_sample"] == [
        {
            "sample_id": cli._record_id({"question": "question"}),
            "answer_relevancy": 5.0,
            "correctness": 5.0,
            "faithfulness": 1.0,
            "temporal_correctness": 1.0,
        }
    ]


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
    cli._prepare_run(
        run_dir / "manifest.json",
        run_dir / "predictions.progress.jsonl",
        identity,
        cli._identity_hash(identity),
        resume=False,
    )
    (run_dir / "predictions.progress.jsonl").write_text(
        json.dumps(
            {
                "evaluation_record_id": "q-2",
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


def test_prepare_run_rejects_existing_or_mismatched_experiments(tmp_path: Path) -> None:
    """Run directories cannot silently overwrite results or mix experiment configurations."""
    manifest = tmp_path / "run" / "manifest.json"
    progress = tmp_path / "run" / "predictions.progress.jsonl"
    cli._prepare_run(manifest, progress, {"pipeline": "naive"}, "run-1", resume=False)

    with pytest.raises(SystemExit, match="pass --resume"):
        cli._prepare_run(manifest, progress, {"pipeline": "naive"}, "run-1", resume=False)
    with pytest.raises(SystemExit, match="different experiment"):
        cli._prepare_run(manifest, progress, {"pipeline": "graphiti"}, "run-2", resume=True)


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

    def __init__(self, _settings: object) -> None:
        return None

    async def score_batch(self, samples: object) -> list[dict[str, float | None]]:
        """Score every sample with the full metric set."""
        return [
            {"faithfulness": 1.0, "correctness": 5.0, "answer_relevancy": 5.0, "temporal_correctness": 1.0}
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

    async def close(self) -> None:
        """Record cleanup."""
        self.closed = True


class _FailingBackend:
    """Raise after startup to exercise failed-run persistence."""

    name = "naive"

    async def predict(self, question: str, limit: int) -> PipelinePrediction:
        """Simulate a pipeline failure."""
        raise _PipelineFailureError

    async def close(self) -> None:
        """Release the fake backend."""


class _PipelineFailureError(RuntimeError):
    """Synthetic pipeline failure used by manifest tests."""
