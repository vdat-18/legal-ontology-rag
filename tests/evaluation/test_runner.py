"""Tests for the shared end-to-end RAG evaluation contract."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from legal_rag.adaptive_rag.models import AdaptiveRAGResult, AdaptiveRoute
from legal_rag.evaluation.metrics import EvaluationSample
from legal_rag.evaluation.runner import (
    AdaptiveEvaluationBackend,
    CorrectiveEvaluationBackend,
    EvaluationRecord,
    GraphitiEvaluationBackend,
    NaiveEvaluationBackend,
    OntologyEvaluationBackend,
    PipelinePrediction,
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
    assert rows[0]["prediction_schema_version"] == "2.0.0"
    assert rows[0]["retrieved_units"] == [
        {
            "unit_id": "chunk-1",
            "unit_kind": "chunk",
            "text": "naive context",
            "evidence": [{"chunk_id": "chunk-1", "source_name": "document.docx"}],
            "score": 0.9,
        }
    ]
    assert rows[0]["retrieved_chunk_ids"] == ["chunk-1"]
    assert rows[0]["retrieved_contexts"][0]["text"] == "naive context"
    assert rows[0]["pipeline_metadata"] == {"search_mode": "hybrid"}
    assert pipeline.calls == [("question", 5, "hybrid")]
    assert pipeline.closed is True


def test_naive_backend_preserves_hierarchical_unit_evidence(tmp_path) -> None:
    """One expanded article remains one ranked unit with all source chunks."""
    records = _records(tmp_path)
    pipeline = _FakeNaivePipeline()
    pipeline.result = RAGResult(
        query="question",
        answer="naive answer",
        sources=(
            RetrievedChunk(
                "seed",
                "expanded article",
                "document.docx",
                "paragraph_group",
                0.9,
                unit_id="article:1",
                unit_kind="article",
                evidence_chunk_ids=("chunk-1", "chunk-2"),
            ),
        ),
    )

    rows = asyncio.run(run_evaluation(records, NaiveEvaluationBackend(pipeline, "hierarchical"), limit=5))

    assert rows[0]["retrieved_units"] == [
        {
            "unit_id": "article:1",
            "unit_kind": "article",
            "text": "expanded article",
            "evidence": [
                {"chunk_id": "chunk-1", "source_name": "document.docx"},
                {"chunk_id": "chunk-2", "source_name": "document.docx"},
            ],
            "score": 0.9,
        }
    ]


def test_naive_backend_passes_multi_turn_history_to_pipeline(tmp_path: Path) -> None:
    """Multi-turn evaluation uses prior question-answer pairs for supported backends."""
    records = [
        EvaluationRecord(
            raw={"question": "follow-up"},
            sample=EvaluationSample(question="follow-up", history=(("first", "first answer"),)),
        )
    ]
    pipeline = _HistoryNaivePipeline()

    asyncio.run(run_evaluation(records, NaiveEvaluationBackend(pipeline), limit=5))

    assert pipeline.history == (("first", "first answer"),)


def test_adaptive_backend_emits_plan_metadata_and_grouped_evidence(tmp_path: Path) -> None:
    """Adaptive plans remain auditable while retrieval units keep chunk provenance."""
    records = _records(tmp_path)
    pipeline = _FakeAdaptivePipeline()

    rows = asyncio.run(run_evaluation(records, AdaptiveEvaluationBackend(pipeline), limit=5))

    assert rows[0]["pipeline"] == "adaptive"
    assert rows[0]["pipeline_metadata"] == {
        "route": "multi_hop",
        "subqueries": ["first", "second"],
        "base_search_mode": "hybrid_rerank",
        "rewritten_query": "rewritten question",
    }
    assert rows[0]["retrieved_units"][0]["unit_id"] == "article:1"
    assert rows[0]["retrieved_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert pipeline.calls == [("question", 5, (), False)]
    assert pipeline.closed is True


def test_adaptive_backend_passes_multi_turn_history(tmp_path: Path) -> None:
    """Adaptive evaluation forwards earlier conversation turns to query rewriting."""
    records = [
        EvaluationRecord(
            raw={"question": "follow-up"},
            sample=EvaluationSample(question="follow-up", history=(("first", "answer"),)),
        )
    ]
    pipeline = _FakeAdaptivePipeline()

    asyncio.run(run_evaluation(records, AdaptiveEvaluationBackend(pipeline, rewrite_query=True), limit=3))

    assert pipeline.calls == [("follow-up", 3, (("first", "answer"),), True)]


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
    assert rows[0]["retrieved_units"][0]["unit_kind"] == "graph_fact"
    assert rows[0]["retrieved_units"][0]["evidence"] == [{"chunk_id": "chunk-1", "source_name": "document.docx"}]
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


def test_run_evaluation_checkpoints_completion_order_but_returns_input_order(tmp_path) -> None:
    """Checkpointing is immediate even when an earlier input is still running."""
    records = _many_records(tmp_path, 3)
    backend = _TrackingBackend(max_concurrency=3, delays={"q0": 0.02})
    emitted: list[tuple[str, bool]] = []

    def on_row(row: dict[str, Any]) -> None:
        emitted.append((row["question_id"], backend.closed))

    rows = asyncio.run(run_evaluation(records, backend, limit=5, concurrency=3, on_row=on_row))

    assert emitted[0][0] != "question-0"
    assert all(not closed for _, closed in emitted)
    assert [row["question_id"] for row in rows] == ["question-0", "question-1", "question-2"]


def test_ontology_backend_preserves_rule_units_and_source_chunk_provenance(tmp_path) -> None:
    """Ontology rules remain ranked units while their chunks provide comparable evidence."""
    records = _records(tmp_path)
    pipeline = _FakeOntologyPipeline()

    rows = asyncio.run(run_evaluation(records, OntologyEvaluationBackend(pipeline), limit=10))

    assert rows[0]["pipeline"] == "ontology"
    assert rows[0]["retrieved_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert rows[0]["retrieved_source_names"] == ["document.docx"]
    assert rows[0]["retrieved_units"] == [
        {
            "unit_id": "https://example.test/rule-1",
            "unit_kind": "ontology_rule",
            "text": "Không đội mũ bảo hiểm (Điều 5)",
            "evidence": [
                {"chunk_id": "chunk-1", "source_name": "document.docx"},
                {"chunk_id": "chunk-2", "source_name": "document.docx"},
            ],
            "score": None,
        }
    ]
    assert rows[0]["retrieved_contexts"] == [
        {
            "text": "Không đội mũ bảo hiểm: Phạt 400000-600000 VND",
            "chunk_id": "rule-1",
            "source_name": "Điều 5",
            "kind": "ontology",
            "url": None,
        }
    ]
    assert rows[0]["pipeline_metadata"] == {"rule_count": 1}
    assert pipeline.calls == [("question", 10)]
    assert pipeline.closed is True


def test_nested_multi_turn_rows_are_flattened_with_history(tmp_path: Path) -> None:
    """Nested ground truth becomes one evaluation record per turn with prior answers."""
    path = tmp_path / "multi-turn.jsonl"
    path.write_text(
        json.dumps(
            {
                "conversation_id": "conversation-1",
                "interaction_type": "multi_turn",
                "turns": [
                    {"turn_number": 1, "question": "first", "expected_answer": "first answer"},
                    {"turn_number": 2, "question": "follow-up", "expected_answer": "follow-up answer"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_evaluation_records(path)

    assert [record.sample.question for record in records] == ["first", "follow-up"]
    assert records[0].raw["conversation_id"] == "conversation-1"
    assert records[0].sample.history == ()
    assert records[1].sample.history == (("first", "first answer"),)


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

    def __init__(self, max_concurrency: int, delays: dict[str, float] | None = None) -> None:
        self._limit = max_concurrency
        self._delays = delays or {}
        self.active = 0
        self.max_active = 0
        self.closed = False

    async def predict(self, question: str, limit: int) -> Any:
        """Yield once so sibling predictions interleave while tracking peak concurrency."""
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(self._delays.get(question, 0))
        self.active -= 1
        return PipelinePrediction(
            answer="answer",
            retrieved_units=(),
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
        self.result = RAGResult(
            query="question",
            answer="naive answer",
            sources=(RetrievedChunk("chunk-1", "naive context", "document.docx", "paragraph", 0.9),),
        )

    def run(self, question: str, top_k: int, search_mode: str | None) -> RAGResult:
        """Record the request and return one retrieved chunk."""
        self.calls.append((question, top_k, search_mode))
        return self.result

    def close(self) -> None:
        """Record resource cleanup."""
        self.closed = True


class _HistoryNaivePipeline:
    """Record conversation history passed to a naive pipeline."""

    settings = SimpleNamespace(search_mode="semantic")

    def __init__(self) -> None:
        self.history: tuple[tuple[str, str], ...] = ()

    def run(self, question: str, top_k: int, history: tuple[tuple[str, str], ...]) -> RAGResult:
        """Record history and return a minimal result."""
        self.history = history
        return RAGResult(query=question, answer="answer")

    def close(self) -> None:
        """Release the fake pipeline."""


class _FakeAdaptivePipeline:
    """Return one fixed adaptive result and record evaluation arguments."""

    settings = SimpleNamespace(base_search_mode="hybrid_rerank")

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, tuple[tuple[str, str], ...], bool]] = []
        self.closed = False

    def run(
        self,
        question: str,
        top_k: int,
        history: tuple[tuple[str, str], ...],
        rewrite_query: bool,
    ) -> AdaptiveRAGResult:
        """Record one call and return grouped adaptive evidence."""
        self.calls.append((question, top_k, history, rewrite_query))
        return AdaptiveRAGResult(
            query=question,
            rewritten_query="rewritten question",
            answer="adaptive answer",
            sources=(
                RetrievedChunk(
                    "seed",
                    "expanded article",
                    "document.docx",
                    "paragraph_group",
                    0.9,
                    unit_id="article:1",
                    unit_kind="article",
                    evidence_chunk_ids=("chunk-1", "chunk-2"),
                ),
            ),
            route=AdaptiveRoute.MULTI_HOP,
            subqueries=("first", "second"),
        )

    def close(self) -> None:
        """Record resource cleanup."""
        self.closed = True


class _FakeOntologyPipeline:
    """Return one ontology result with distinct chunk sources and rule contexts."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.closed = False

    def run(self, question: str, top_k: int) -> SimpleNamespace:
        """Record the request and return chunks plus one rule-summary context."""
        self.calls.append((question, top_k))
        return SimpleNamespace(
            answer="ontology answer",
            rules=(
                SimpleNamespace(
                    uri="https://example.test/rule-1",
                    conduct_label_vi="Không đội mũ bảo hiểm",
                    provision_path="Điều 5",
                    source_chunk_ids=("chunk-1", "chunk-2"),
                ),
            ),
            sources=(
                RetrievedChunk("chunk-1", "chunk one", "document.docx", "paragraph", 0.9),
                RetrievedChunk("chunk-2", "chunk two", "document.docx", "paragraph", 0.8),
                RetrievedChunk("chunk-1", "chunk one duplicate", "document.docx", "paragraph", 0.7),
            ),
            answer_contexts=(
                RetrievedChunk("rule-1", "Không đội mũ bảo hiểm: Phạt 400000-600000 VND", "Điều 5", "ontology", 1.0),
            ),
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
            "graded_chunks": [
                {
                    "chunk_id": "initial-chunk",
                    "source_name": "document.docx",
                    "text": "initial context",
                    "score": 0.9,
                }
            ],
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
