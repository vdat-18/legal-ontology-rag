"""Tests for offline evaluation metrics."""

from __future__ import annotations

import json

from legal_rag.evaluation.metrics import EvaluationSample, evaluate_samples, load_jsonl_samples


def test_evaluate_samples_computes_retrieval_and_answer_metrics() -> None:
    """Aggregate metrics are averaged over computable samples."""
    samples = [
        EvaluationSample(
            question="q1",
            expected_answer="Xe may phat 4 trieu dong",
            actual_answer="Xe may phat 4 trieu dong",
            expected_chunk_ids=("a", "b"),
            retrieved_chunk_ids=("x", "a", "z"),
        ),
        EvaluationSample(
            question="q2",
            expected_answer="Bi tru diem",
            actual_answer="Khong co thong tin",
            expected_chunk_ids=("c",),
            retrieved_chunk_ids=("x", "y"),
        ),
    ]

    metrics = evaluate_samples(samples, k_values=(1, 2, 3))

    assert metrics.sample_count == 2
    assert metrics.retrieval_sample_count == 2
    assert metrics.k_values == (1, 2, 3)
    assert metrics.hit_rate == 0.5
    assert metrics.hit_rate_at_k == {1: 0.0, 2: 0.5, 3: 0.5}
    assert metrics.precision_at_k == {1: 0.0, 2: 0.25, 3: 1 / 6}
    assert metrics.recall_at_k == {1: 0.0, 2: 0.25, 3: 0.25}
    assert metrics.context_precision == 1 / 6
    assert metrics.context_recall == 0.25


def test_metrics_to_dict_uses_json_friendly_k_keys() -> None:
    """Metric output keeps @k values in JSON-friendly maps."""
    sample = EvaluationSample(
        question="q1",
        expected_chunk_ids=("a",),
        retrieved_chunk_ids=("a", "b"),
    )

    data = evaluate_samples([sample], k_values=(1,)).to_dict()

    assert data["precision_at_k"] == {"1": 1.0}
    assert data["recall_at_k"] == {"1": 1.0}
    assert data["hit_rate_at_k"] == {"1": 1.0}


def test_retrieval_metrics_compare_only_matching_label_types() -> None:
    """Retrieval scoring never compares chunk IDs against source names."""
    samples = [
        EvaluationSample(
            question="q1",
            expected_source_names=("source-a",),
            retrieved_chunk_ids=("chunk-a",),
            retrieved_source_names=("source-a",),
        ),
        EvaluationSample(
            question="q2",
            expected_chunk_ids=("chunk-b",),
            retrieved_source_names=("chunk-b",),
        ),
    ]

    metrics = evaluate_samples(samples, k_values=(1,))

    assert metrics.retrieval_sample_count == 2
    assert metrics.hit_rate == 0.5
    assert metrics.precision_at_k == {1: 0.5}
    assert metrics.recall_at_k == {1: 0.5}


def test_retrieval_metrics_count_empty_retrieval_as_zero() -> None:
    """Ground-truth retrieval samples with empty retrieved IDs remain in evaluation."""
    sample = EvaluationSample(
        question="q1",
        expected_chunk_ids=("chunk-a",),
        retrieved_chunk_ids=(),
    )

    metrics = evaluate_samples([sample], k_values=(1, 3))

    assert metrics.retrieval_sample_count == 1
    assert metrics.hit_rate == 0.0
    assert metrics.precision_at_k == {1: 0.0, 3: 0.0}
    assert metrics.recall_at_k == {1: 0.0, 3: 0.0}
    assert metrics.context_precision == 0.0
    assert metrics.context_recall == 0.0


def test_load_jsonl_samples_accepts_pipeline_result_shape(tmp_path) -> None:
    """Loader extracts answer and retrieval fields from a CRAG result-shaped row."""
    path = tmp_path / "predictions.jsonl"
    row = {
        "query": "q1",
        "ground_truth": "answer",
        "result": {"answer": "answer"},
        "expected_source_names": ["source-a"],
        "graded_chunks": [{"chunk_id": "chunk-a", "source_name": "source-a"}],
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    samples = load_jsonl_samples(path)

    assert samples == [
        EvaluationSample(
            question="q1",
            expected_answer="answer",
            actual_answer="answer",
            retrieved_chunk_ids=("chunk-a",),
            expected_source_names=("source-a",),
            retrieved_source_names=("source-a",),
        )
    ]
