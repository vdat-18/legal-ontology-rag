"""Tests for the shared Legal RAG evaluation CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.evaluation import cli
from legal_rag.evaluation.runner import PipelinePrediction, PredictionContext


@pytest.mark.parametrize("pipeline", ["naive", "corrective", "graphiti", "ontology"])
def test_run_accepts_every_comparable_pipeline(pipeline: str) -> None:
    """The end-to-end command exposes all four implementations behind one interface."""
    arguments = cli._parser().parse_args(["run", "--pipeline", pipeline])

    assert arguments.pipeline == pipeline
    assert arguments.ground_truth.as_posix() == "data/benchmark_synthesis/groundtruth.jsonl"


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
    assert metric_report["answer_exact_match"] == 1.0
    assert backend.closed is True


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
        {"answer_relevancy": 5.0, "correctness": 5.0, "faithfulness": 1.0, "temporal_correctness": 1.0}
    ]


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

    async def predict(self, question: str, limit: int) -> PipelinePrediction:
        """Return a result matching the test ground truth."""
        return PipelinePrediction(
            answer="answer",
            retrieved_chunk_ids=("chunk-1",),
            retrieved_source_names=("document.docx",),
            retrieved_contexts=(PredictionContext(text="context", chunk_id="chunk-1"),),
            metadata={},
        )

    async def close(self) -> None:
        """Record cleanup."""
        self.closed = True
