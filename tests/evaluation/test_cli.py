"""Tests for the shared Legal RAG evaluation CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.evaluation import cli
from legal_rag.evaluation.runner import PipelinePrediction, PredictionContext


@pytest.mark.parametrize("pipeline", ["naive", "corrective", "graphiti"])
def test_run_accepts_every_comparable_pipeline(pipeline: str) -> None:
    """The end-to-end command exposes all three implementations behind one interface."""
    arguments = cli._parser().parse_args(["run", "--pipeline", pipeline])

    assert arguments.pipeline == pipeline
    assert arguments.ground_truth.as_posix() == "data/benchmark_synthesis/groundtruth.jsonl"


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
