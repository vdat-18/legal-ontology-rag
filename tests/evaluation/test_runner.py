"""Tests for the shared end-to-end RAG evaluation contract."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from legal_rag.evaluation.runner import (
    CorrectiveEvaluationBackend,
    EvaluationRecord,
    GraphitiEvaluationBackend,
    NaiveEvaluationBackend,
    load_evaluation_records,
    run_evaluation,
)
from legal_rag.graphiti_rag.retriever import FactSource, RetrievedFact
from legal_rag.naive_rag.pipeline import RAGResult
from legal_rag.naive_rag.retriever import RetrievedChunk


def test_naive_backend_emits_shared_prediction_schema(tmp_path) -> None:
    """Naive RAG preserves ground-truth metadata and exposes its generation chunks."""
    records = _records(tmp_path)
    pipeline = _FakeNaivePipeline()

    rows = asyncio.run(run_evaluation(records, NaiveEvaluationBackend(pipeline, "hybrid"), limit=5))

    assert rows[0]["question_id"] == "question-1"
    assert rows[0]["pipeline"] == "naive"
    assert rows[0]["actual_answer"] == "naive answer"
    assert rows[0]["retrieved_chunk_ids"] == ["chunk-1"]
    assert rows[0]["retrieved_contexts"][0]["text"] == "naive context"
    assert rows[0]["pipeline_metadata"] == {"search_mode": "hybrid"}
    assert pipeline.calls == [("question", 5, "hybrid")]
    assert pipeline.closed is True


def test_corrective_backend_separates_initial_retrieval_from_generation_context(tmp_path) -> None:
    """CRAG retrieval metrics use initial chunks while faithfulness uses corrected knowledge."""
    records = _records(tmp_path)
    pipeline = _FakeCorrectivePipeline()

    rows = asyncio.run(run_evaluation(records, CorrectiveEvaluationBackend(pipeline), limit=5))

    assert rows[0]["retrieved_chunk_ids"] == ["initial-chunk"]
    assert rows[0]["retrieved_contexts"] == [
        {
            "text": "refined context",
            "chunk_id": "initial-chunk",
            "source_name": "document.docx",
            "kind": "internal",
            "url": None,
        }
    ]
    assert rows[0]["pipeline_metadata"]["crag_action"] == "CORRECT"
    assert pipeline.closed is True


def test_graphiti_backend_maps_ranked_fact_provenance_to_gold_chunk_ids(tmp_path) -> None:
    """Graph facts are scored through their unique supporting chunk provenance."""
    records = _records(tmp_path)
    pipeline = _FakeGraphitiPipeline()

    rows = asyncio.run(run_evaluation(records, GraphitiEvaluationBackend(pipeline), limit=5))

    assert rows[0]["retrieved_chunk_ids"] == ["chunk-1"]
    assert rows[0]["retrieved_source_names"] == ["document.docx"]
    assert rows[0]["retrieved_contexts"][0]["text"] == "source passage\n\n[Trích xuất] extracted fact"
    assert rows[0]["pipeline_metadata"]["rewritten_query"] == "rewritten question"
    assert pipeline.calls == [("question", 5)]
    assert pipeline.closed is True


def test_run_evaluation_bounds_concurrency_and_preserves_order(tmp_path) -> None:
    """No more than *concurrency* predictions run at once and rows keep input order."""
    records = _many_records(tmp_path, 6)
    backend = _TrackingBackend(max_concurrency=2)

    rows = asyncio.run(run_evaluation(records, backend, limit=5, concurrency=2))

    assert backend.max_active == 2
    assert [row["question_id"] for row in rows] == [f"question-{index}" for index in range(6)]
    assert backend.closed is True


def test_run_evaluation_emits_rows_incrementally_before_close(tmp_path) -> None:
    """on_row receives each row in input order while the backend is still open."""
    records = _many_records(tmp_path, 3)
    backend = _TrackingBackend(max_concurrency=3)
    emitted: list[tuple[str, bool]] = []

    def on_row(row: dict[str, Any]) -> None:
        emitted.append((row["question_id"], backend.closed))

    rows = asyncio.run(run_evaluation(records, backend, limit=5, concurrency=3, on_row=on_row))

    assert [question_id for question_id, _ in emitted] == ["question-0", "question-1", "question-2"]
    assert all(not closed for _, closed in emitted)
    assert len(rows) == 3


def _many_records(tmp_path: Path, count: int) -> list[EvaluationRecord]:
    """Write and load *count* minimal ground-truth rows."""
    path = tmp_path / "groundtruth.jsonl"
    path.write_text(
        "".join(
            json.dumps({"question_id": f"question-{index}", "question": f"q{index}"}) + "\n" for index in range(count)
        ),
        encoding="utf-8",
    )
    return load_evaluation_records(path)


class _TrackingBackend:
    """Count in-flight predictions to verify the concurrency bound."""

    name = "tracking"

    def __init__(self, max_concurrency: int) -> None:
        self._limit = max_concurrency
        self.active = 0
        self.max_active = 0
        self.closed = False

    async def predict(self, question: str, limit: int) -> Any:
        """Yield once so sibling predictions interleave while tracking peak concurrency."""
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return SimpleNamespace(
            answer="answer",
            retrieved_chunk_ids=(),
            retrieved_source_names=(),
            retrieved_contexts=(),
            metadata={},
        )

    async def close(self) -> None:
        """Record cleanup."""
        self.closed = True


def _records(tmp_path: Path) -> list[EvaluationRecord]:
    """Write and load one representative synthesized ground-truth row."""
    path = tmp_path / "groundtruth.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "question-1",
                "question": "question",
                "expected_answer": "expected answer",
                "expected_chunk_ids": ["chunk-1"],
                "required_claims": [{"claim": "expected answer"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return load_evaluation_records(path)


class _FakeNaivePipeline:
    """Return one fixed naive result."""

    def __init__(self) -> None:
        self.settings = SimpleNamespace(search_mode="semantic")
        self.calls: list[tuple[str, int, str | None]] = []
        self.closed = False

    def run(self, question: str, top_k: int, search_mode: str | None) -> RAGResult:
        """Record the request and return one retrieved chunk."""
        self.calls.append((question, top_k, search_mode))
        return RAGResult(
            query=question,
            answer="naive answer",
            sources=(RetrievedChunk("chunk-1", "naive context", "document.docx", "paragraph", 0.9),),
        )

    def close(self) -> None:
        """Record resource cleanup."""
        self.closed = True


class _FakeCorrectivePipeline:
    """Return initial retrieval and distinct refined generation knowledge."""

    def __init__(self) -> None:
        self.closed = False

    def run(self, question: str, limit: int) -> dict[str, Any]:
        """Return a representative Corrective RAG result."""
        return {
            "answer": "corrective answer",
            "status": "SUCCESS_CORRECT",
            "crag_action": "CORRECT",
            "reformulated_query": None,
            "web_search_count": 0,
            "graded_chunks": [{"chunk_id": "initial-chunk", "source_name": "document.docx"}],
            "generation_contexts": [
                {
                    "chunk_id": "initial-chunk",
                    "source_name": "document.docx",
                    "text": "refined context",
                    "kind": "internal",
                    "url": None,
                }
            ],
        }

    def close(self) -> None:
        """Record resource cleanup."""
        self.closed = True


class _FakeGraphitiPipeline:
    """Return one fact with exact chunk provenance."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.closed = False

    async def run(self, question: str, max_facts: int) -> SimpleNamespace:
        """Record the request and return one graph fact."""
        self.calls.append((question, max_facts))
        return SimpleNamespace(
            answer="graph answer",
            rewritten_query="rewritten question",
            sources=(
                RetrievedFact(
                    fact="extracted fact",
                    sources=(FactSource("chunk-1", "document.docx", "source passage"),),
                ),
            ),
            unindexed_citations=(),
            suggestions=(),
        )

    async def close(self) -> None:
        """Record resource cleanup."""
        self.closed = True
