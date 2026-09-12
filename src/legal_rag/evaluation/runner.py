"""Shared prediction contract for end-to-end Legal RAG evaluation."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from legal_rag.evaluation.metrics import EvaluationSample, expand_evaluation_row


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One original ground-truth row and its normalized scoring fields."""

    raw: Mapping[str, Any]
    sample: EvaluationSample


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """One context item actually supplied to answer generation."""

    text: str
    chunk_id: str | None = None
    source_name: str | None = None
    kind: str | None = None
    url: str | None = None


@dataclass(frozen=True, slots=True)
class PipelinePrediction:
    """Pipeline-neutral answer and retrieval result."""

    answer: str
    retrieved_chunk_ids: tuple[str, ...]
    retrieved_source_names: tuple[str, ...]
    retrieved_contexts: tuple[PredictionContext, ...]
    metadata: Mapping[str, Any]


class EvaluationBackend(Protocol):
    """Boundary implemented by each evaluated RAG pipeline."""

    name: str

    async def predict(self, question: str, limit: int) -> PipelinePrediction:
        """Run one question and return the shared prediction contract."""

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
        contexts = tuple(
            PredictionContext(
                chunk_id=source.chunk_id,
                source_name=source.source_name,
                text=source.text,
                kind=source.kind,
            )
            for source in result.sources
        )
        return PipelinePrediction(
            answer=result.answer,
            retrieved_chunk_ids=tuple(context.chunk_id for context in contexts if context.chunk_id),
            retrieved_source_names=tuple(context.source_name for context in contexts if context.source_name),
            retrieved_contexts=contexts,
            metadata=_with_optional_value(
                {"search_mode": self._search_mode or self._pipeline.settings.search_mode},
                "rewritten_query",
                result.rewritten_query,
            ),
        )

    async def close(self) -> None:
        """Close the naive pipeline's Milvus connection."""
        self._pipeline.close()


class CorrectiveEvaluationBackend:
    """Adapt Corrective RAG while separating retrieval labels from generation context."""

    name = "corrective"

    def __init__(self, pipeline: Any) -> None:
        """Store the Corrective RAG pipeline."""
        self._pipeline = pipeline

    async def predict(self, question: str, limit: int) -> PipelinePrediction:
        """Run CRAG and retain both initial retrieval and corrected knowledge."""
        result = await asyncio.to_thread(self._pipeline.run, question, limit=limit)
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
        return PipelinePrediction(
            answer=_required_string(result.get("answer"), "CRAG answer"),
            retrieved_chunk_ids=_field_values(graded_chunks, "chunk_id"),
            retrieved_source_names=_field_values(graded_chunks, "source_name"),
            retrieved_contexts=contexts,
            metadata={
                "status": result.get("status"),
                "crag_action": result.get("crag_action"),
                "reformulated_query": result.get("reformulated_query"),
                "web_search_count": result.get("web_search_count"),
                "graded_chunks": graded_chunks,
            },
        )

    async def close(self) -> None:
        """Close all Corrective RAG clients."""
        self._pipeline.close()


class OntologyEvaluationBackend:
    """Adapt the ontology-enhanced RAG pipeline for shared evaluation.

    Retrieval labels come from the Milvus chunks the pipeline retrieved, while the generation
    context is the rule summaries the pipeline actually supplied to answer generation.
    """

    name = "ontology"

    def __init__(self, pipeline: Any, rewrite_query: bool = False) -> None:
        """Store the ontology RAG pipeline.

        Args:
            pipeline: The OntologyRAGPipeline instance.
            rewrite_query: Whether to rewrite queries before retrieval.
        """
        self._pipeline = pipeline
        self._rewrite_query = rewrite_query

    async def predict(self, question: str, limit: int) -> PipelinePrediction:
        """Run ontology RAG and expose both chunk labels and rule-summary contexts."""
        return await self._predict(question, limit)

    async def predict_with_history(
        self, question: str, limit: int, history: tuple[tuple[str, str], ...]
    ) -> PipelinePrediction:
        """Run ontology RAG with earlier conversation turns."""
        return await self._predict(question, limit, history=history)

    async def _predict(
        self, question: str, limit: int, history: tuple[tuple[str, str], ...] = ()
    ) -> PipelinePrediction:
        """Run ontology RAG while passing only arguments it supports."""
        run_kwargs = {"top_k": limit, "rewrite_query": self._rewrite_query}
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
        contexts = tuple(
            PredictionContext(
                chunk_id=context.chunk_id,
                source_name=context.source_name,
                text=context.text,
                kind=context.kind,
            )
            for context in result.answer_contexts
        )
        return PipelinePrediction(
            answer=result.answer,
            retrieved_chunk_ids=_unique(source.chunk_id for source in result.sources if source.chunk_id),
            retrieved_source_names=_unique(source.source_name for source in result.sources if source.source_name),
            retrieved_contexts=contexts,
            metadata=_with_optional_value(
                {"rule_count": len(result.rules)}, "rewritten_query", getattr(result, "rewritten_query", None)
            ),
        )

    async def close(self) -> None:
        """Close the ontology pipeline's Milvus connection."""
        await asyncio.to_thread(self._pipeline.close)


class GraphitiEvaluationBackend:
    """Adapt Graphiti facts and their chunk provenance for shared evaluation."""

    name = "graphiti"

    def __init__(self, pipeline: Any) -> None:
        """Store the Graphiti RAG pipeline."""
        self._pipeline = pipeline

    async def predict(self, question: str, limit: int) -> PipelinePrediction:
        """Run Graphiti RAG and flatten ranked fact provenance into chunk labels."""
        return await self._predict(question, limit)

    async def predict_with_history(
        self, question: str, limit: int, history: tuple[tuple[str, str], ...]
    ) -> PipelinePrediction:
        """Run Graphiti RAG with earlier conversation turns."""
        return await self._predict(question, limit, history=history)

    async def _predict(
        self, question: str, limit: int, history: tuple[tuple[str, str], ...] = ()
    ) -> PipelinePrediction:
        """Run Graphiti RAG with typed conversation history."""
        from legal_rag.baml_client.types import ConversationTurn
        from legal_rag.graphiti_rag.generator import format_fact_context

        typed_history = tuple(ConversationTurn(question=item[0], answer=item[1]) for item in history)
        run_kwargs = {"max_facts": limit, "history": typed_history}
        supported_kwargs = {
            key: value for key, value in run_kwargs.items() if key in inspect.signature(self._pipeline.run).parameters
        }
        result = await self._pipeline.run(question, **supported_kwargs)
        contexts = tuple(
            PredictionContext(
                chunk_id=source.chunk_id,
                source_name=source.source_name,
                text=format_fact_context(fact.fact, source.text),
                kind="graph_fact",
            )
            for fact in result.sources
            for source in fact.sources
        )
        return PipelinePrediction(
            answer=result.answer,
            retrieved_chunk_ids=_unique(context.chunk_id for context in contexts if context.chunk_id),
            retrieved_source_names=_unique(context.source_name for context in contexts if context.source_name),
            retrieved_contexts=contexts,
            metadata={
                "rewritten_query": result.rewritten_query,
                "unindexed_citations": list(result.unindexed_citations),
                "suggestions": list(result.suggestions),
            },
        )

    async def close(self) -> None:
        """Close the Graphiti/Neo4j connection."""
        await self._pipeline.close()


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
                EvaluationRecord(raw=dict(item), sample=EvaluationSample.from_mapping(item))
                for item in expand_evaluation_row(row)
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
    is not overwhelmed. When *on_row* is given it is invoked with each finished row, in input
    order, as soon as every earlier row has also finished — so a caller can persist progress
    without waiting for the whole dataset. The returned list keeps the input record order.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _predict_record(index: int, record: EvaluationRecord) -> dict[str, Any]:
        async with semaphore:
            logger.info("Evaluating {} sample {}/{}", backend.name, index + 1, len(records))
            predict_with_history = getattr(backend, "predict_with_history", None)
            if record.sample.history and callable(predict_with_history):
                prediction = await predict_with_history(record.sample.question, limit, record.sample.history)
            else:
                prediction = await backend.predict(record.sample.question, limit)
        row = dict(record.raw)
        row.update(
            {
                "pipeline": backend.name,
                "actual_answer": prediction.answer,
                "retrieved_chunk_ids": list(prediction.retrieved_chunk_ids),
                "retrieved_source_names": list(prediction.retrieved_source_names),
                "retrieved_contexts": [asdict(context) for context in prediction.retrieved_contexts],
                "pipeline_metadata": dict(prediction.metadata),
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
                finished[indexed[future]] = future.result()
            while next_index in finished:
                row = finished.pop(next_index)
                rows.append(row)
                if on_row is not None:
                    on_row(row)
                next_index += 1
    finally:
        for future in pending:
            future.cancel()
        await backend.close()
    return rows


def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    """Return mapping items from an external sequence response."""
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _field_values(items: Sequence[Mapping[str, Any]], key: str) -> tuple[str, ...]:
    """Read non-empty string values from result mappings."""
    return tuple(value for item in items if (value := _optional_string(item.get(key))))


def _optional_string(value: Any) -> str | None:
    """Normalize one optional external string."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _with_optional_value(metadata: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Add optional pipeline metadata only when the pipeline returned it."""
    if value is not None:
        metadata[key] = value
    return metadata


def _required_string(value: Any, label: str) -> str:
    """Require a non-empty string at an external pipeline boundary."""
    normalized = _optional_string(value)
    if normalized is None:
        msg = f"{label} must be a non-empty string"
        raise ValueError(msg)
    return normalized


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate ranked identifiers without changing their order."""
    return tuple(dict.fromkeys(values))
