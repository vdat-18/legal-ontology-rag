"""Shared prediction contract for end-to-end Legal RAG evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from loguru import logger

from legal_rag.baml_transport import LLMUsage, capture_llm_usage
from legal_rag.evaluation.metrics import EvaluationSample, expand_evaluation_row
from legal_rag.temporal import resolve_temporal_targets


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One original ground-truth row and its normalized scoring fields."""

    raw: Mapping[str, Any]
    sample: EvaluationSample
    query_date: date | None = None
    query_dates: tuple[date, ...] = ()
    temporal_comparison: bool = False


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """One context item actually supplied to answer generation."""

    text: str
    chunk_id: str | None = None
    source_name: str | None = None
    kind: str | None = None
    url: str | None = None


@dataclass(frozen=True, slots=True)
class PredictionEvidence:
    """One source chunk supporting a pipeline-native retrieval unit."""

    chunk_id: str
    source_name: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievedUnit:
    """One ranked retrieval unit with chunk-level provenance for shared scoring."""

    unit_id: str
    unit_kind: str
    text: str
    evidence: tuple[PredictionEvidence, ...]
    score: float | None = None


@dataclass(frozen=True, slots=True)
class PipelinePrediction:
    """Pipeline-neutral answer and retrieval result."""

    answer: str
    retrieved_units: tuple[RetrievedUnit, ...]
    retrieved_contexts: tuple[PredictionContext, ...]
    metadata: Mapping[str, Any]

    @property
    def retrieved_chunk_ids(self) -> tuple[str, ...]:
        """Flatten ranked unit provenance for compatibility with v1 prediction files."""
        return _unique(evidence.chunk_id for unit in self.retrieved_units for evidence in unit.evidence)

    @property
    def retrieved_source_names(self) -> tuple[str, ...]:
        """Flatten ranked unit source names for compatibility with v1 prediction files."""
        return _unique(
            evidence.source_name for unit in self.retrieved_units for evidence in unit.evidence if evidence.source_name
        )


class EvaluationBackend(Protocol):
    """Boundary implemented by each evaluated RAG pipeline."""

    name: str

    async def predict(self, question: str, limit: int) -> PipelinePrediction:
        """Run one question and return the shared prediction contract."""

    async def preflight_corpus(self) -> Mapping[str, Any]:
        """Validate and fingerprint the active retrieval backend before evaluation."""

    async def close(self) -> None:
        """Release external resources owned by the pipeline."""


class NaiveEvaluationBackend:
    """Adapt the naive retrieve-then-generate pipeline for shared evaluation."""

    name = "naive"

    def __init__(self, pipeline: Any, search_mode: str | None = None, rewrite_query: bool = False) -> None:
        """Store the pipeline and optional retrieval-mode override.

        Args:
            pipeline: The NaiveRAGPipeline instance.
            search_mode: Optional retrieval mode override.
            rewrite_query: Whether to rewrite queries before retrieval.
        """
        self._pipeline = pipeline
        self._search_mode = search_mode
        self._rewrite_query = rewrite_query

    async def predict(self, question: str, limit: int) -> PipelinePrediction:
        """Run naive RAG and expose its retrieved chunks as generation context."""
        return await self._predict(question, limit)

    async def predict_with_history(
        self, question: str, limit: int, history: tuple[tuple[str, str], ...]
    ) -> PipelinePrediction:
        """Run naive RAG with earlier conversation turns."""
        return await self._predict(question, limit, history=history)

    async def _predict(
        self, question: str, limit: int, history: tuple[tuple[str, str], ...] = ()
    ) -> PipelinePrediction:
        """Run the pipeline while passing only arguments it supports."""
        run_kwargs = {"top_k": limit, "search_mode": self._search_mode, "rewrite_query": self._rewrite_query}
        if history:
            run_kwargs["history"] = history
        supported_kwargs = {
            key: value for key, value in run_kwargs.items() if key in inspect.signature(self._pipeline.run).parameters
        }
        result = await asyncio.to_thread(
            self._pipeline.run,
            question,
            **supported_kwargs,
        )
        return _retrieved_chunks_prediction(
            result,
            _with_optional_value(
                {"search_mode": self._search_mode or self._pipeline.settings.search_mode},
                "rewritten_query",
                result.rewritten_query,
            ),
        )

    async def close(self) -> None:
        """Close the naive pipeline's Milvus connection."""
        self._pipeline.close()

    async def preflight_corpus(self) -> Mapping[str, Any]:
        """Verify and fingerprint the active Naive Milvus collection."""
        fingerprint = await asyncio.to_thread(self._pipeline.preflight_corpus)
        return fingerprint.to_dict()


class AdaptiveEvaluationBackend:
    """Adapt one-shot adaptive RAG to the shared evaluation contract."""

    name = "adaptive"

    def __init__(self, pipeline: Any, rewrite_query: bool = False) -> None:
        """Store the adaptive pipeline and query-rewriting policy."""
        self._pipeline = pipeline
        self._rewrite_query = rewrite_query

    async def predict(self, question: str, limit: int) -> PipelinePrediction:
        """Run adaptive RAG without conversation history."""
        return await self._predict(question, limit)

    async def predict_with_history(
        self, question: str, limit: int, history: tuple[tuple[str, str], ...]
    ) -> PipelinePrediction:
        """Run adaptive RAG with earlier conversation turns."""
        return await self._predict(question, limit, history)

    async def _predict(
        self,
        question: str,
        limit: int,
        history: tuple[tuple[str, str], ...] = (),
    ) -> PipelinePrediction:
        """Run the blocking adaptive pipeline off the event loop."""
        result = await asyncio.to_thread(
            self._pipeline.run,
            question,
            top_k=limit,
            history=history,
            rewrite_query=self._rewrite_query,
        )
        metadata = {
            "route": result.route.value,
            "subqueries": list(result.subqueries),
            "base_search_mode": self._pipeline.settings.base_search_mode,
        }
        return _retrieved_chunks_prediction(
            result,
            _with_optional_value(metadata, "rewritten_query", result.rewritten_query),
        )

    async def close(self) -> None:
        """Close the adaptive pipeline's Milvus connection."""
        self._pipeline.close()

    async def preflight_corpus(self) -> Mapping[str, Any]:
        """Verify and fingerprint the active shared Naive Milvus collection."""
        fingerprint = await asyncio.to_thread(self._pipeline.preflight_corpus)
        return fingerprint.to_dict()


class CorrectiveEvaluationBackend:
    """Adapt Corrective RAG while separating retrieval labels from generation context."""

    name = "corrective"

    def __init__(self, pipeline: Any, chunks_dir: Path = Path("data/processed/groundtruth")) -> None:
        """Store the Corrective RAG pipeline and the corpus its active index must match."""
        self._pipeline = pipeline
        self._chunks_dir = chunks_dir

    async def predict(self, question: str, limit: int) -> PipelinePrediction:
        """Run CRAG and retain both initial retrieval and corrected knowledge."""
        return await self._predict(question, limit)

    async def predict_with_history(
        self, question: str, limit: int, history: tuple[tuple[str, str], ...]
    ) -> PipelinePrediction:
        """Run CRAG with earlier conversation turns used to resolve the current question."""
        return await self._predict(question, limit, history)

    async def _predict(
        self,
        question: str,
        limit: int,
        history: tuple[tuple[str, str], ...] = (),
    ) -> PipelinePrediction:
        """Run the blocking CRAG pipeline off the event loop."""
        run_kwargs: dict[str, Any] = {"limit": limit}
        if history:
            run_kwargs["history"] = history
        result = await asyncio.to_thread(self._pipeline.run, question, **run_kwargs)
        graded_chunks = _mapping_sequence(result.get("graded_chunks"))
        generation_contexts = _mapping_sequence(result.get("generation_contexts"))
        contexts = tuple(
            PredictionContext(
                chunk_id=_optional_string(item.get("chunk_id")),
                source_name=_optional_string(item.get("source_name")),
                text=_required_string(item.get("text"), "CRAG generation context text"),
                kind=_optional_string(item.get("kind")),
                url=_optional_string(item.get("url")),
            )
            for item in generation_contexts
        )
        units = tuple(
            RetrievedUnit(
                unit_id=_required_string(item.get("chunk_id"), "CRAG retrieved chunk id"),
                unit_kind="chunk",
                text=_required_string(item.get("text"), "CRAG retrieved chunk text"),
                evidence=(
                    PredictionEvidence(
                        chunk_id=_required_string(item.get("chunk_id"), "CRAG retrieved chunk id"),
                        source_name=_optional_string(item.get("source_name")),
                    ),
                ),
                score=_optional_float(item.get("score")),
            )
            for item in graded_chunks
        )
        return PipelinePrediction(
            answer=_required_string(result.get("answer"), "CRAG answer"),
            retrieved_units=units,
            retrieved_contexts=contexts,
            metadata={
                "status": result.get("status"),
                "crag_action": result.get("crag_action"),
                "rewritten_query": result.get("rewritten_query"),
                "reformulated_query": result.get("reformulated_query"),
                "web_search_count": result.get("web_search_count"),
                "graded_chunks": graded_chunks,
            },
        )

    async def close(self) -> None:
        """Close all Corrective RAG clients."""
        self._pipeline.close()

    async def preflight_corpus(self) -> Mapping[str, Any]:
        """Verify and fingerprint the active Corrective Milvus collection."""
        fingerprint = await asyncio.to_thread(self._pipeline.preflight_corpus, self._chunks_dir)
        return fingerprint.to_dict()


class OntologyEvaluationBackend:
    """Adapt the ontology-enhanced RAG pipeline for shared evaluation.

    Ranked retrieval units are selected ontology rules, with their source chunks retained as
    comparable evidence. Generation context is the rule summaries supplied to answer generation.
    """

    name = "ontology"

    def __init__(
        self,
        pipeline: Any,
        rewrite_query: bool = False,
        reference_date: date | None = None,
    ) -> None:
        """Store the ontology RAG pipeline.

        Args:
            pipeline: The OntologyRAGPipeline instance.
            rewrite_query: Whether to rewrite queries before retrieval.
            reference_date: Optional fixed current date for reproducible evaluation.
        """
        self._pipeline = pipeline
        self._rewrite_query = rewrite_query
        self._reference_date = reference_date

    async def predict(
        self,
        question: str,
        limit: int,
        query_date: date | None = None,
        query_dates: tuple[date, ...] = (),
    ) -> PipelinePrediction:
        """Run ontology RAG and expose selected rules with source-chunk provenance."""
        return await self._predict(question, limit, query_date=query_date, query_dates=query_dates)

    async def predict_with_history(
        self,
        question: str,
        limit: int,
        history: tuple[tuple[str, str], ...],
        query_date: date | None = None,
        query_dates: tuple[date, ...] = (),
    ) -> PipelinePrediction:
        """Run ontology RAG with earlier conversation turns."""
        return await self._predict(
            question,
            limit,
            history=history,
            query_date=query_date,
            query_dates=query_dates,
        )

    async def _predict(
        self,
        question: str,
        limit: int,
        history: tuple[tuple[str, str], ...] = (),
        query_date: date | None = None,
        query_dates: tuple[date, ...] = (),
    ) -> PipelinePrediction:
        """Run ontology RAG while passing only arguments it supports."""
        run_kwargs = {"top_k": limit, "rewrite_query": self._rewrite_query}
        if history:
            run_kwargs["history"] = history
        if query_date is not None:
            run_kwargs["query_date"] = query_date
        if query_dates:
            run_kwargs["query_dates"] = query_dates
        if self._reference_date is not None:
            run_kwargs["reference_date"] = self._reference_date
        supported_kwargs = {
            key: value for key, value in run_kwargs.items() if key in inspect.signature(self._pipeline.run).parameters
        }
        result = await asyncio.to_thread(
            self._pipeline.run,
            question,
            **supported_kwargs,
        )
        contexts = tuple(
            PredictionContext(
                chunk_id=context.chunk_id,
                source_name=context.source_name,
                text=context.text,
                kind=context.kind,
            )
            for context in result.answer_contexts
        )
        source_names = {
            context.chunk_id: context.source_name
            for context in (*result.sources, *result.answer_contexts)
            if context.chunk_id and context.source_name
        }
        units = tuple(
            RetrievedUnit(
                unit_id=str(rule.uri),
                unit_kind="ontology_rule",
                text=_ontology_rule_text(rule),
                evidence=tuple(
                    PredictionEvidence(chunk_id=chunk_id, source_name=source_names.get(chunk_id))
                    for chunk_id in rule.source_chunk_ids
                ),
            )
            for rule in result.rules
        )
        if any(not unit.evidence for unit in units):
            msg = "Ontology evaluation requires source_chunk_ids on every selected rule"
            raise ValueError(msg)
        return PipelinePrediction(
            answer=result.answer,
            retrieved_units=units,
            retrieved_contexts=contexts,
            metadata=_ontology_result_metadata(result, query_date, query_dates),
        )

    async def close(self) -> None:
        """Close the ontology pipeline's Milvus connection."""
        await asyncio.to_thread(self._pipeline.close)

    async def preflight_corpus(self) -> Mapping[str, Any]:
        """Verify and fingerprint the active fallback Milvus collection."""
        fingerprint = await asyncio.to_thread(self._pipeline.preflight_corpus)
        return fingerprint.to_dict()


class GraphitiEvaluationBackend:
    """Adapt Graphiti facts and their chunk provenance for shared evaluation."""

    name = "graphiti"

    def __init__(
        self,
        pipeline: Any,
        *,
        neo4j_settings: Any | None = None,
        group_id: str | None = None,
        expected_chunks: Sequence[Any] = (),
        expected_source_dates: Mapping[str, date] | None = None,
        embedding_profile: Any | None = None,
        reference_date: date | None = None,
    ) -> None:
        """Store the pipeline and exact local corpus used to validate its graph group."""
        self._pipeline = pipeline
        self._neo4j_settings = neo4j_settings
        self._group_id = group_id
        self._expected_chunks = tuple(expected_chunks)
        self._expected_source_dates = dict(expected_source_dates or {})
        self._embedding_profile = embedding_profile
        self._reference_date = reference_date

    async def predict(
        self,
        question: str,
        limit: int,
        query_date: date | None = None,
        query_dates: tuple[date, ...] = (),
        temporal_comparison: bool = False,
    ) -> PipelinePrediction:
        """Run Graphiti RAG and preserve ranked facts with chunk provenance."""
        return await self._predict(
            question,
            limit,
            query_date=query_date,
            query_dates=query_dates,
            temporal_comparison=temporal_comparison,
        )

    async def predict_with_history(
        self,
        question: str,
        limit: int,
        history: tuple[tuple[str, str], ...],
        query_date: date | None = None,
        query_dates: tuple[date, ...] = (),
        temporal_comparison: bool = False,
    ) -> PipelinePrediction:
        """Run Graphiti RAG with earlier conversation turns."""
        return await self._predict(
            question,
            limit,
            history=history,
            query_date=query_date,
            query_dates=query_dates,
            temporal_comparison=temporal_comparison,
        )

    async def _predict(
        self,
        question: str,
        limit: int,
        history: tuple[tuple[str, str], ...] = (),
        query_date: date | None = None,
        query_dates: tuple[date, ...] = (),
        temporal_comparison: bool = False,
    ) -> PipelinePrediction:
        """Run Graphiti RAG with typed conversation history."""
        from legal_rag.baml_client.types import ConversationTurn
        from legal_rag.graphiti_rag.generator import format_fact_context

        typed_history = tuple(ConversationTurn(question=item[0], answer=item[1]) for item in history)
        run_kwargs = {
            "max_facts": limit,
            "history": typed_history,
            "include_auxiliary_outputs": False,
        }
        as_of_dates = resolve_temporal_targets(
            question,
            query_dates=query_dates,
            query_date=query_date,
            reference_date=self._reference_date,
            temporal_comparison=temporal_comparison,
        )
        run_parameters = inspect.signature(self._pipeline.run).parameters
        if as_of_dates:
            if "as_of_dates" in run_parameters:
                run_kwargs["as_of_dates"] = as_of_dates
            elif len(as_of_dates) == 1 and "as_of_date" in run_parameters:
                run_kwargs["as_of_date"] = as_of_dates[0]
        supported_kwargs = {key: value for key, value in run_kwargs.items() if key in run_parameters}
        result = await self._pipeline.run(question, **supported_kwargs)
        contexts = tuple(
            PredictionContext(
                chunk_id=source.chunk_id,
                source_name=source.source_name,
                text=format_fact_context(fact.fact, source.text, fact.as_of_dates),
                kind="graph_fact",
            )
            for fact in result.sources
            for source in fact.sources
        )
        units = tuple(
            RetrievedUnit(
                unit_id=_graph_fact_id(fact),
                unit_kind="graph_fact",
                text=fact.fact,
                evidence=tuple(
                    PredictionEvidence(chunk_id=source.chunk_id, source_name=source.source_name)
                    for source in fact.sources
                ),
            )
            for fact in result.sources
        )
        metadata = {
            "rewritten_query": result.rewritten_query,
            "unindexed_citations": list(result.unindexed_citations),
            "suggestions": list(result.suggestions),
        }
        if as_of_dates:
            metadata["as_of_dates"] = [item.isoformat() for item in as_of_dates]
            if len(as_of_dates) == 1:
                metadata["as_of_date"] = as_of_dates[0].isoformat()
        return PipelinePrediction(
            answer=result.answer,
            retrieved_units=units,
            retrieved_contexts=contexts,
            metadata=metadata,
        )

    async def close(self) -> None:
        """Close the Graphiti/Neo4j connection."""
        await self._pipeline.close()

    async def preflight_corpus(self) -> Mapping[str, Any]:
        """Validate the active Neo4j group and return its deterministic snapshot."""
        if (
            self._neo4j_settings is None
            or self._group_id is None
            or not self._expected_chunks
            or self._embedding_profile is None
        ):
            msg = "Graphiti evaluation requires Neo4j settings, a group id, expected local chunks, and an embedding profile"
            raise ValueError(msg)
        from legal_rag.datastores.neo4j import build_neo4j_driver
        from legal_rag.graphiti_rag.index_metadata import validate_graphiti_index_metadata
        from legal_rag.graphiti_rag.snapshot import snapshot_graphiti_group

        driver = build_neo4j_driver(self._neo4j_settings)
        try:
            embedding_metadata = await validate_graphiti_index_metadata(
                driver,
                self._group_id,
                self._embedding_profile,
            )
            snapshot = await snapshot_graphiti_group(
                driver,
                self._group_id,
                expected_chunks=self._expected_chunks,
                expected_source_dates=self._expected_source_dates,
            )
            snapshot.validate_for_evaluation()
            return {**snapshot.to_dict(), "embedding_metadata": asdict(embedding_metadata)}
        finally:
            await driver.close()


def load_evaluation_records(path: Path) -> list[EvaluationRecord]:
    """Load ground truth without discarding synthesis metadata."""
    records: list[EvaluationRecord] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                msg = f"Expected a JSON object at {path}:{line_number}"
                raise TypeError(msg)
            records.extend(
                _evaluation_record(item, path=path, line_number=line_number) for item in expand_evaluation_row(row)
            )
    return records


async def run_evaluation(
    records: Sequence[EvaluationRecord],
    backend: EvaluationBackend,
    *,
    limit: int,
    concurrency: int = 1,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run one backend over every record and emit a shared prediction schema.

    At most *concurrency* questions are evaluated at once; a semaphore bounds how many pipeline
    calls (each including one LLM answer generation) run at the same time so the shared provider
    is not overwhelmed. When *on_row* is given it is invoked immediately in completion order so
    a caller can durably persist progress without waiting for slower earlier rows. The returned
    list keeps the input record order.
    """
    semaphore = asyncio.Semaphore(concurrency)
    active_predictions: set[asyncio.Task[Any]] = set()

    async def _predict_record(index: int, record: EvaluationRecord) -> dict[str, Any]:
        async with semaphore:
            current_task = asyncio.current_task()
            if current_task is not None:
                active_predictions.add(current_task)
            try:
                logger.info("Evaluating {} sample {}/{}", backend.name, index + 1, len(records))
                started_at = perf_counter()
                with capture_llm_usage() as usage:
                    predict_with_history = getattr(backend, "predict_with_history", None)
                    if record.sample.history and callable(predict_with_history):
                        prediction = await _predict_with_optional_temporal_scope(
                            predict_with_history,
                            record.sample.question,
                            limit,
                            record.sample.history,
                            query_date=record.query_date,
                            query_dates=record.query_dates,
                            temporal_comparison=record.temporal_comparison,
                        )
                    else:
                        prediction = await _predict_with_optional_temporal_scope(
                            backend.predict,
                            record.sample.question,
                            limit,
                            query_date=record.query_date,
                            query_dates=record.query_dates,
                            temporal_comparison=record.temporal_comparison,
                        )
                latency_ms = (perf_counter() - started_at) * 1000
            finally:
                if current_task is not None:
                    active_predictions.discard(current_task)
        row = dict(record.raw)
        row.update(
            {
                "prediction_schema_version": "2.0.0",
                "pipeline": backend.name,
                "actual_answer": prediction.answer,
                "retrieved_units": [_retrieved_unit_dict(unit) for unit in prediction.retrieved_units],
                "retrieved_chunk_ids": list(prediction.retrieved_chunk_ids),
                "retrieved_source_names": list(prediction.retrieved_source_names),
                "retrieved_contexts": [asdict(context) for context in prediction.retrieved_contexts],
                "pipeline_metadata": dict(prediction.metadata),
                "evaluation_metadata": _evaluation_metadata(prediction, latency_ms, usage),
            }
        )
        return row

    rows: list[dict[str, Any]] = []
    finished: dict[int, dict[str, Any]] = {}
    next_index = 0
    indexed: dict[asyncio.Future[dict[str, Any]], int] = {
        asyncio.ensure_future(_predict_record(index, record)): index for index, record in enumerate(records)
    }
    pending: set[asyncio.Future[dict[str, Any]]] = set(indexed)
    try:
        while pending:
            done_futures, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for future in done_futures:
                pending.remove(future)
                index = indexed[future]
                row = future.result()
                finished[index] = row
                if on_row is not None:
                    on_row(row)
            while next_index in finished:
                row = finished.pop(next_index)
                rows.append(row)
                next_index += 1
    finally:
        await _finish_evaluation_tasks(pending, active_predictions, backend)
    return rows


async def _finish_evaluation_tasks(
    pending: set[asyncio.Future[dict[str, Any]]],
    active_predictions: set[asyncio.Task[Any]],
    backend: EvaluationBackend,
) -> None:
    """Stop queued work and let started pipeline calls finish before closing resources."""
    active = [future for future in pending if future in active_predictions]
    waiting = [future for future in pending if future not in active_predictions]
    for future in waiting:
        future.cancel()
    if waiting:
        await asyncio.gather(*waiting, return_exceptions=True)
    if active:
        await asyncio.gather(*active, return_exceptions=True)
    await backend.close()


def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    """Return mapping items from an external sequence response."""
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _retrieved_unit_dict(unit: RetrievedUnit) -> dict[str, Any]:
    """Serialize one unit into the exact JSON-compatible prediction schema."""
    return {
        "unit_id": unit.unit_id,
        "unit_kind": unit.unit_kind,
        "text": unit.text,
        "evidence": [asdict(item) for item in unit.evidence],
        "score": unit.score,
    }


def _evaluation_metadata(prediction: PipelinePrediction, latency_ms: float, usage: LLMUsage) -> dict[str, Any]:
    """Describe the measured work behind one prediction without changing retrieval scores."""
    evidence_ids = {evidence.chunk_id for unit in prediction.retrieved_units for evidence in unit.evidence}
    return {
        "latency_ms": latency_ms,
        "retrieved_unit_count": len(prediction.retrieved_units),
        "unique_evidence_chunk_count": len(evidence_ids),
        "generation_context_count": len(prediction.retrieved_contexts),
        "generation_context_characters": sum(len(context.text) for context in prediction.retrieved_contexts),
        "usage": usage.to_dict(),
    }


def _optional_float(value: Any) -> float | None:
    """Normalize one optional numeric score."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _optional_string(value: Any) -> str | None:
    """Normalize one optional external string."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _with_optional_value(metadata: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Add optional pipeline metadata only when the pipeline returned it."""
    if value is not None:
        metadata[key] = value
    return metadata


async def _predict_with_optional_temporal_scope(
    predict: Callable[..., Any],
    *args: Any,
    query_date: date | None,
    query_dates: tuple[date, ...],
    temporal_comparison: bool,
) -> PipelinePrediction:
    """Pass ground-truth temporal controls only to backends that expose them."""
    kwargs = {}
    parameters = inspect.signature(predict).parameters
    if query_date is not None and not query_dates and "query_date" in parameters:
        kwargs["query_date"] = query_date
    if query_dates and "query_dates" in parameters:
        kwargs["query_dates"] = query_dates
    if temporal_comparison and "temporal_comparison" in parameters:
        kwargs["temporal_comparison"] = True
    return await predict(*args, **kwargs)


def _evaluation_record(item: Mapping[str, Any], *, path: Path, line_number: int) -> EvaluationRecord:
    """Build one typed record, letting canonical plural dates supersede the legacy scalar."""
    query_date = _optional_query_date(item.get("query_date"), path=path, line_number=line_number)
    query_dates = _optional_query_dates(item.get("query_dates"), path=path, line_number=line_number)
    if query_date is not None and query_dates and query_date not in query_dates:
        msg = f"query_date must be included in query_dates at {path}:{line_number}"
        raise ValueError(msg)
    return EvaluationRecord(
        raw=dict(item),
        sample=EvaluationSample.from_mapping(item),
        query_date=None if query_dates else query_date,
        query_dates=query_dates,
        temporal_comparison=item.get("bundle_kind") == "temporal",
    )


def _optional_query_date(value: Any, *, path: Path, line_number: int) -> date | None:
    """Parse an optional ground-truth date using the exact persisted ISO format."""
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"query_date must be an ISO date string at {path}:{line_number}"
        raise TypeError(msg)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        msg = f"Invalid query_date at {path}:{line_number}: {value!r}; expected YYYY-MM-DD"
        raise ValueError(msg) from error
    if parsed.isoformat() != value:
        msg = f"Invalid query_date at {path}:{line_number}: {value!r}; expected YYYY-MM-DD"
        raise ValueError(msg)
    return parsed


def _optional_query_dates(value: Any, *, path: Path, line_number: int) -> tuple[date, ...]:
    """Parse unique plural temporal targets from the persisted evaluation schema."""
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        msg = f"query_dates must be an array of ISO date strings at {path}:{line_number}"
        raise TypeError(msg)
    parsed = tuple(_optional_query_date(item, path=path, line_number=line_number) for item in value)
    if any(item is None for item in parsed):
        msg = f"query_dates must not contain null at {path}:{line_number}"
        raise ValueError(msg)
    dates = tuple(item for item in parsed if item is not None)
    if len(set(dates)) != len(dates):
        msg = f"query_dates must not contain duplicates at {path}:{line_number}"
        raise ValueError(msg)
    return dates


def _ontology_result_metadata(
    result: Any,
    requested_query_date: date | None,
    requested_query_dates: Sequence[date] = (),
) -> dict[str, Any]:
    """Serialize temporal decisions and candidate accounting returned by ontology RAG."""
    metadata: dict[str, Any] = {"rule_count": len(result.rules)}
    _with_optional_value(metadata, "rewritten_query", getattr(result, "rewritten_query", None))
    if requested_query_date is not None:
        metadata["requested_query_date"] = requested_query_date.isoformat()
    if requested_query_dates:
        metadata["requested_query_dates"] = [item.isoformat() for item in requested_query_dates]

    temporal_mode = getattr(result, "temporal_mode", None)
    if temporal_mode is not None:
        metadata["temporal_mode"] = str(temporal_mode)
    query_dates = getattr(result, "query_dates", None)
    if query_dates is not None:
        metadata["query_dates"] = [item.isoformat() if isinstance(item, date) else str(item) for item in query_dates]
    temporal_status = getattr(result, "temporal_status", None)
    if isinstance(temporal_status, Mapping):
        metadata["temporal_status"] = {str(key): str(value) for key, value in temporal_status.items()}
    for field_name in ("candidate_count", "invalid_rule_count", "indeterminate_rule_count"):
        value = getattr(result, field_name, None)
        if value is not None:
            metadata[field_name] = value
    return metadata


def _retrieved_chunks_prediction(result: Any, metadata: Mapping[str, Any]) -> PipelinePrediction:
    """Map Naive-compatible retrieved chunks into the shared prediction schema."""
    contexts = tuple(
        PredictionContext(
            chunk_id=source.chunk_id,
            source_name=source.source_name,
            text=source.text,
            kind=source.kind,
        )
        for source in result.sources
    )
    units = tuple(
        RetrievedUnit(
            unit_id=source.effective_unit_id,
            unit_kind=source.unit_kind,
            text=source.text,
            evidence=tuple(
                PredictionEvidence(chunk_id=chunk_id, source_name=source.source_name)
                for chunk_id in source.effective_evidence_chunk_ids
            ),
            score=source.score,
        )
        for source in result.sources
    )
    return PipelinePrediction(
        answer=result.answer,
        retrieved_units=units,
        retrieved_contexts=contexts,
        metadata=metadata,
    )


def _required_string(value: Any, label: str) -> str:
    """Require a non-empty string at an external pipeline boundary."""
    normalized = _optional_string(value)
    if normalized is None:
        msg = f"{label} must be a non-empty string"
        raise ValueError(msg)
    return normalized


def _ontology_rule_text(rule: Any) -> str:
    """Render stable text for one selected ontology rule."""
    return f"{rule.conduct_label_vi} ({rule.provision_path})"


def _graph_fact_id(fact: Any) -> str:
    """Build a deterministic identity for a graph fact and its provenance."""
    evidence = "\n".join(sorted({source.chunk_id for source in fact.sources}))
    return hashlib.sha256(f"{fact.fact}\n{evidence}".encode()).hexdigest()


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate ranked identifiers without changing their order."""
    return tuple(dict.fromkeys(values))
