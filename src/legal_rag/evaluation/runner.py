"""Shared prediction contract for end-to-end Legal RAG evaluation."""

from __future__ import annotations

import asyncio
import hashlib
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

    Ranked retrieval units are selected ontology rules, with their source chunks retained as
    comparable evidence. Generation context is the rule summaries supplied to answer generation.
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
        """Run ontology RAG and expose selected rules with source-chunk provenance."""
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
        source_names = {source.chunk_id: source.source_name for source in result.sources}
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
        """Run Graphiti RAG and preserve ranked facts with chunk provenance."""
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
        return PipelinePrediction(
            answer=result.answer,
            retrieved_units=units,
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
    is not overwhelmed. When *on_row* is given it is invoked immediately in completion order so
    a caller can durably persist progress without waiting for slower earlier rows. The returned
    list keeps the input record order.
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
                "prediction_schema_version": "2.0.0",
                "pipeline": backend.name,
                "actual_answer": prediction.answer,
                "retrieved_units": [_retrieved_unit_dict(unit) for unit in prediction.retrieved_units],
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
        for future in pending:
            future.cancel()
        await backend.close()
    return rows


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
