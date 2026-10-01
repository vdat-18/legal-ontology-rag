"""Tests for the shared end-to-end RAG evaluation contract."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from time import sleep
from types import SimpleNamespace
from typing import Any

import pytest

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


def test_runner_passes_conversation_history_alias_to_pipeline(tmp_path: Path) -> None:
    """Legacy conversation_history rows keep their context through the shared runner."""
    path = tmp_path / "legacy-history.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "follow-up",
                "conversation_history": [
                    {"user_input": "first", "response": "first answer"},
                    ["second", "second answer"],
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    [record] = load_evaluation_records(path)
    pipeline = _HistoryNaivePipeline()

    asyncio.run(run_evaluation([record], NaiveEvaluationBackend(pipeline), limit=5))

    assert pipeline.history == (("first", "first answer"), ("second", "second answer"))


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
    assert rows[0]["evaluation_metadata"]["retrieved_unit_count"] == 1
    assert rows[0]["evaluation_metadata"]["unique_evidence_chunk_count"] == 1
    assert rows[0]["evaluation_metadata"]["generation_context_count"] == 1
    assert rows[0]["evaluation_metadata"]["generation_context_characters"] == len("refined context")
    assert rows[0]["evaluation_metadata"]["latency_ms"] >= 0
    assert rows[0]["evaluation_metadata"]["usage"] == {
        "call_count": 0,
        "usage_available_call_count": 0,
        "usage_unavailable_call_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "models": {},
        "partial": False,
    }
    assert pipeline.closed is True


def test_corrective_backend_passes_multi_turn_history(tmp_path: Path) -> None:
    """CRAG evaluation resolves follow-up questions through its shared history contract."""
    records = [
        EvaluationRecord(
            raw={"question": "follow-up"},
            sample=EvaluationSample(question="follow-up", history=(("first", "answer"),)),
        )
    ]
    pipeline = _FakeCorrectivePipeline()

    [row] = asyncio.run(run_evaluation(records, CorrectiveEvaluationBackend(pipeline), limit=4))

    assert pipeline.calls == [("follow-up", 4, (("first", "answer"),))]
    assert row["pipeline_metadata"]["rewritten_query"] == "standalone question"


def test_corrective_backend_preflights_the_identity_corpus_path(tmp_path: Path) -> None:
    """CRAG validates the same local corpus path fingerprinted by the run identity."""
    preflight_paths: list[Path] = []

    class _PreflightPipeline:
        def preflight_corpus(self, chunks_dir: Path) -> SimpleNamespace:
            preflight_paths.append(chunks_dir)
            return SimpleNamespace(to_dict=lambda: {"corpus_sha256": "verified"})

    fingerprint = asyncio.run(CorrectiveEvaluationBackend(_PreflightPipeline(), tmp_path).preflight_corpus())

    assert fingerprint == {"corpus_sha256": "verified"}
    assert preflight_paths == [tmp_path]


def test_graphiti_backend_maps_ranked_fact_provenance_to_gold_chunk_ids(tmp_path) -> None:
    """Graph facts are scored through their unique supporting chunk provenance."""
    records = _records(tmp_path)
    pipeline = _FakeGraphitiPipeline()

    rows = asyncio.run(run_evaluation(records, GraphitiEvaluationBackend(pipeline), limit=5))

    assert rows[0]["retrieved_chunk_ids"] == ["chunk-1"]
    assert rows[0]["retrieved_source_names"] == ["document.docx"]
    assert rows[0]["retrieved_units"][0]["unit_kind"] == "graph_fact"
    assert rows[0]["retrieved_units"][0]["evidence"] == [{"chunk_id": "chunk-1", "source_name": "document.docx"}]
    assert rows[0]["retrieved_contexts"][0]["text"] == (
        "[Có hiệu lực tại mốc truy hồi: 2025-06-01]\nsource passage\n\n[Trích xuất] extracted fact"
    )
    assert rows[0]["pipeline_metadata"]["rewritten_query"] == "rewritten question"
    assert pipeline.calls == [("question", 5, False, (date(2025, 6, 1),))]
    assert rows[0]["pipeline_metadata"]["as_of_date"] == "2025-06-01"
    assert rows[0]["pipeline_metadata"]["as_of_dates"] == ["2025-06-01"]
    assert pipeline.closed is True


def test_graphiti_preflight_validates_authoritative_episode_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end Graphiti preflight sends document validity dates to the active snapshot validator."""
    from legal_rag.datastores import neo4j as neo4j_module
    from legal_rag.graphiti_rag import index_metadata as index_metadata_module
    from legal_rag.graphiti_rag import snapshot as snapshot_module

    calls: list[dict[str, object]] = []

    class FakeDriver:
        closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeSnapshot:
        def validate_for_evaluation(self) -> None:
            return None

        def to_dict(self) -> dict[str, object]:
            return {"sha256": "graph-snapshot"}

    async def validate_metadata(*_args: object) -> index_metadata_module.GraphitiIndexMetadata:
        return index_metadata_module.GraphitiIndexMetadata(
            metadata_schema_version="1.0.0",
            embedding_preprocessing_version="1.0.0",
            embedding_profile_id="profile",
            embedding_model_id="model",
            embedding_model_revision="revision",
            embedding_dimensions=1024,
        )

    async def snapshot_group(*_args: object, **kwargs: object) -> FakeSnapshot:
        calls.append(kwargs)
        return FakeSnapshot()

    driver = FakeDriver()
    monkeypatch.setattr(neo4j_module, "build_neo4j_driver", lambda _settings: driver)
    monkeypatch.setattr(index_metadata_module, "validate_graphiti_index_metadata", validate_metadata)
    monkeypatch.setattr(snapshot_module, "snapshot_graphiti_group", snapshot_group)
    chunk = SimpleNamespace(chunk_id="chunk-1", source_name="document.docx")
    backend = GraphitiEvaluationBackend(
        SimpleNamespace(),
        neo4j_settings="settings",
        group_id="group",
        expected_chunks=(chunk,),
        expected_source_dates={"document.docx": date(2024, 7, 1)},
        embedding_profile="profile",
    )

    fingerprint = asyncio.run(backend.preflight_corpus())

    assert calls == [
        {
            "expected_chunks": (chunk,),
            "expected_source_dates": {"document.docx": date(2024, 7, 1)},
        }
    ]
    assert fingerprint["sha256"] == "graph-snapshot"
    assert driver.closed is True


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


def test_run_evaluation_waits_for_started_workers_before_closing_backend() -> None:
    """A peer failure cannot close resources still used by a thread-backed prediction."""

    class _FailingPeerBackend:
        name = "test"

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.closed = False
            self.closed_while_active: list[bool] = []

        async def predict(self, question: str, _limit: int) -> PipelinePrediction:
            if question == "fail":
                await self.started.wait()
                msg = "simulated peer failure"
                raise RuntimeError(msg)
            self.started.set()
            return await asyncio.to_thread(self._slow_prediction)

        def _slow_prediction(self) -> PipelinePrediction:
            sleep(0.05)
            self.closed_while_active.append(self.closed)
            return PipelinePrediction(answer="answer", retrieved_units=(), retrieved_contexts=(), metadata={})

        async def close(self) -> None:
            self.closed = True

    backend = _FailingPeerBackend()
    records = [
        EvaluationRecord(raw={"question": question}, sample=EvaluationSample(question=question))
        for question in ("slow", "fail")
    ]

    with pytest.raises(RuntimeError, match="simulated peer failure"):
        asyncio.run(run_evaluation(records, backend, limit=1, concurrency=2))

    assert backend.closed_while_active == [False]
    assert backend.closed is True


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
    assert rows[0]["pipeline_metadata"] == {
        "candidate_count": 4,
        "indeterminate_rule_count": 1,
        "invalid_rule_count": 2,
        "query_dates": ["2025-06-01"],
        "requested_query_date": "2025-06-01",
        "rule_count": 1,
        "temporal_mode": "as_of",
        "temporal_status": {"https://example.test/rule-1": "valid"},
    }
    assert pipeline.calls == [("question", 10, date(2025, 6, 1), ())]
    assert pipeline.closed is True


def test_ontology_backend_forwards_authoritative_plural_query_dates() -> None:
    """Ontology receives the same benchmark version targets as Graphiti and the judge."""
    targets = (date(2025, 1, 1), date(2026, 3, 1))
    record = EvaluationRecord(
        raw={"question": "compare", "query_dates": [item.isoformat() for item in targets]},
        sample=EvaluationSample(question="compare"),
        query_dates=targets,
        temporal_comparison=True,
    )
    pipeline = _FakeOntologyPipeline()

    [row] = asyncio.run(run_evaluation([record], OntologyEvaluationBackend(pipeline), limit=10))

    assert pipeline.calls == [("compare", 10, None, targets)]
    assert row["pipeline_metadata"]["requested_query_dates"] == ["2025-01-01", "2026-03-01"]
    assert row["pipeline_metadata"]["query_dates"] == ["2025-01-01", "2026-03-01"]


def test_ontology_backend_omits_redundant_scalar_when_plural_dates_are_present() -> None:
    """Accepted legacy-plus-plural rows cannot violate the ontology pipeline's exclusive API."""
    targets = (date(2025, 1, 1), date(2026, 3, 1))
    record = EvaluationRecord(
        raw={
            "question": "compare",
            "query_date": targets[0].isoformat(),
            "query_dates": [item.isoformat() for item in targets],
        },
        sample=EvaluationSample(question="compare"),
        query_date=targets[0],
        query_dates=targets,
        temporal_comparison=True,
    )
    pipeline = _FakeOntologyPipeline()

    asyncio.run(run_evaluation([record], OntologyEvaluationBackend(pipeline), limit=10))

    assert pipeline.calls == [("compare", 10, None, targets)]


def test_load_evaluation_records_rejects_noncanonical_query_date(tmp_path: Path) -> None:
    """Evaluation dates must use a calendar date rather than a datetime or compact form."""
    path = tmp_path / "groundtruth.jsonl"
    path.write_text(json.dumps({"question": "question", "query_date": "20250601"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected YYYY-MM-DD"):
        load_evaluation_records(path)


def test_load_evaluation_records_preserves_plural_temporal_targets(tmp_path: Path) -> None:
    """Version-comparison dates remain typed and the temporal bundle is marked explicitly."""
    path = tmp_path / "groundtruth.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "compare",
                "bundle_kind": "temporal",
                "query_dates": ["2025-01-01", "2026-03-01"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    [record] = load_evaluation_records(path)

    assert record.query_dates == (date(2025, 1, 1), date(2026, 3, 1))
    assert record.temporal_comparison is True


@pytest.mark.parametrize(
    ("query_dates", "error"),
    [
        ("2025-01-01", TypeError),
        (["2025-1-1"], ValueError),
        (["2025-01-01", "2025-01-01"], ValueError),
        ([None], ValueError),
    ],
)
def test_load_evaluation_records_rejects_invalid_query_dates(
    tmp_path: Path,
    query_dates: object,
    error: type[Exception],
) -> None:
    """Plural targets use the same strict persisted-date contract as the legacy field."""
    path = tmp_path / "groundtruth.jsonl"
    path.write_text(json.dumps({"question": "question", "query_dates": query_dates}) + "\n", encoding="utf-8")

    with pytest.raises(error):
        load_evaluation_records(path)


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
                "query_date": "2025-06-01",
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
        self.calls: list[tuple[str, int, date | None, tuple[date, ...]]] = []
        self.closed = False

    def run(
        self,
        question: str,
        top_k: int,
        query_date: date | None = None,
        query_dates: tuple[date, ...] = (),
    ) -> SimpleNamespace:
        """Record the request and return chunks plus one rule-summary context."""
        self.calls.append((question, top_k, query_date, query_dates))
        resolved_dates = query_dates or ((query_date,) if query_date is not None else (date(2025, 6, 1),))
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
            temporal_mode="as_of",
            query_dates=resolved_dates,
            temporal_status={"https://example.test/rule-1": "valid"},
            candidate_count=4,
            invalid_rule_count=2,
            indeterminate_rule_count=1,
        )

    def close(self) -> None:
        """Record resource cleanup."""
        self.closed = True


class _FakeCorrectivePipeline:
    """Return initial retrieval and distinct refined generation knowledge."""

    def __init__(self) -> None:
        self.closed = False
        self.calls: list[tuple[str, int, tuple[tuple[str, str], ...]]] = []

    def run(
        self,
        question: str,
        limit: int,
        history: tuple[tuple[str, str], ...] = (),
    ) -> dict[str, Any]:
        """Return a representative Corrective RAG result."""
        self.calls.append((question, limit, history))
        return {
            "answer": "corrective answer",
            "rewritten_query": "standalone question" if history else None,
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
        self.calls: list[tuple[str, int, bool, tuple[date, ...]]] = []
        self.closed = False

    async def run(
        self,
        question: str,
        max_facts: int,
        *,
        as_of_dates: tuple[date, ...] = (),
        include_auxiliary_outputs: bool = True,
    ) -> SimpleNamespace:
        """Record the request and return one graph fact."""
        self.calls.append((question, max_facts, include_auxiliary_outputs, as_of_dates))
        return SimpleNamespace(
            answer="graph answer",
            rewritten_query="rewritten question",
            sources=(
                RetrievedFact(
                    fact="extracted fact",
                    sources=(FactSource("chunk-1", "document.docx", "source passage"),),
                    as_of_dates=as_of_dates,
                ),
            ),
            unindexed_citations=(),
            suggestions=(),
        )

    async def close(self) -> None:
        """Record resource cleanup."""
        self.closed = True
