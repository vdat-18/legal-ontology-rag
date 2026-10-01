"""Tests for action-gated CRAG knowledge selection before generation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from legal_rag.corrective_rag.generate.actions import ActionTrigger
from legal_rag.corrective_rag.generate.grader import RelevanceEvaluation
from legal_rag.corrective_rag.generate.models import KnowledgeItem
from legal_rag.corrective_rag.generate.pipeline import CorrectiveRAGPipeline
from legal_rag.corrective_rag.generate.refiner import KnowledgeRefiner
from legal_rag.query_rewriting import RewrittenQuery


def test_correct_branch_generates_from_only_correct_internal_knowledge() -> None:
    """CORRECT retrieval refines only CORRECT chunks and only usable strips."""
    pipeline = _pipeline(
        chunks=[
            _chunk("correct internal", "correct-source"),
            _chunk("ambiguous internal", "ambiguous-source"),
            _chunk("irrelevant internal", "irrelevant-source"),
        ],
        scores={
            "correct internal": 0.9,
            "ambiguous internal": 0.5,
            "irrelevant internal": 0.0,
        },
    )

    result = pipeline.run("q")

    assert result["crag_action"] == "CORRECT"
    assert result["internal_knowledge_count"] == 1
    assert result["external_knowledge_count"] == 0
    assert result["generation_contexts"] == [
        {
            "text": "correct internal",
            "source_name": "correct-source",
            "score": 0.9,
            "kind": "internal",
            "chunk_id": "correct-source",
            "url": None,
        }
    ]
    assert _knowledge_sources(pipeline.generator.knowledge) == ["correct-source"]
    assert pipeline.web_search.queries == []


def test_ambiguous_branch_generates_from_ambiguous_internal_and_usable_external_knowledge() -> None:
    """AMBIGUOUS retrieval excludes incorrect chunks and low-scored external strips."""
    pipeline = _pipeline(
        chunks=[
            _chunk("ambiguous internal", "ambiguous-source"),
            _chunk("irrelevant internal", "irrelevant-source"),
        ],
        web_documents=[
            _web_document("strong external", "external-strong"),
            _web_document("weak external", "external-weak"),
            _web_document("zero external", "external-zero"),
        ],
        scores={
            "ambiguous internal": 0.5,
            "irrelevant internal": 0.0,
            "strong external": 0.9,
            "weak external": 0.2,
            "zero external": 0.0,
        },
    )

    result = pipeline.run("q")

    assert result["crag_action"] == "AMBIGUOUS"
    assert result["internal_knowledge_count"] == 1
    assert result["external_knowledge_count"] == 1
    assert _knowledge_sources(pipeline.generator.knowledge) == ["ambiguous-source", "external-strong"]
    assert pipeline.web_search.queries == ["rewritten q"]


def test_incorrect_branch_generates_from_only_usable_external_knowledge() -> None:
    """INCORRECT retrieval discards internal chunks and never keeps zero-score strips."""
    pipeline = _pipeline(
        chunks=[_chunk("irrelevant internal", "irrelevant-source")],
        web_documents=[
            _web_document("partial external", "external-partial"),
            _web_document("zero external", "external-zero"),
        ],
        scores={
            "irrelevant internal": 0.0,
            "partial external": 0.5,
            "zero external": 0.0,
        },
    )

    result = pipeline.run("q")

    assert result["crag_action"] == "INCORRECT"
    assert result["internal_knowledge_count"] == 0
    assert result["external_knowledge_count"] == 1
    assert _knowledge_sources(pipeline.generator.knowledge) == ["external-partial"]
    assert pipeline.web_search.queries == ["rewritten q"]


def test_multi_turn_uses_standalone_question_across_the_crag_pipeline() -> None:
    """A follow-up is resolved before retrieval, grading, refinement, and generation."""
    pipeline = _pipeline(
        chunks=[_chunk("correct internal", "correct-source")],
        scores={"correct internal": 0.9},
    )
    rewriter = _FakeQueryRewriter("standalone legal question")
    pipeline._query_rewriter = rewriter
    history = (("previous question", "previous answer"),)

    result = pipeline.run("follow-up", history=history)

    assert rewriter.calls == [("follow-up", history)]
    assert pipeline.db.queries == [("standalone legal question", 5)]
    assert pipeline.grader.queries == ["standalone legal question", "standalone legal question"]
    assert pipeline.generator.answer_query == "standalone legal question"
    assert pipeline.generator.original_query == "follow-up"
    assert result["query"] == "follow-up"
    assert result["rewritten_query"] == "standalone legal question"


def test_single_turn_keeps_the_original_question_without_rewriting() -> None:
    """Existing single-turn CRAG behavior does not add an unnecessary LLM call."""
    pipeline = _pipeline(
        chunks=[_chunk("correct internal", "correct-source")],
        scores={"correct internal": 0.9},
    )
    rewriter = _FakeQueryRewriter("should not be used")
    pipeline._query_rewriter = rewriter

    result = pipeline.run("original question")

    assert rewriter.calls == []
    assert pipeline.db.queries == [("original question", 5)]
    assert pipeline.generator.answer_query == "original question"
    assert pipeline.generator.original_query is None
    assert result["rewritten_query"] is None


def test_multi_turn_rejects_an_empty_standalone_question() -> None:
    """Malformed rewrite output fails before retrieval rather than querying with empty text."""
    pipeline = _pipeline(
        chunks=[_chunk("correct internal", "correct-source")],
        scores={"correct internal": 0.9},
    )
    pipeline._query_rewriter = _FakeQueryRewriter("   ")

    with pytest.raises(ValueError, match="empty standalone question"):
        pipeline.run("follow-up", history=(("previous question", "previous answer"),))

    assert pipeline.db.queries == []


def _pipeline(
    *,
    chunks: list[dict[str, Any]],
    scores: dict[str, float],
    web_documents: list[dict[str, str]] | None = None,
) -> CorrectiveRAGPipeline:
    """Build a CRAG pipeline with fake side-effecting adapters."""
    config = _config()
    pipeline = object.__new__(CorrectiveRAGPipeline)
    pipeline.config = config
    pipeline.db = _FakeVectorDb(chunks)
    pipeline.grader = _FakeGrader(scores)
    pipeline.generator = _FakeGenerator()
    pipeline.action_trigger = ActionTrigger(
        lower_threshold=config.evaluator_lower_threshold,
        upper_threshold=config.evaluator_upper_threshold,
    )
    pipeline.refiner = KnowledgeRefiner(config, pipeline.grader)
    pipeline.web_search = _FakeWebSearch(web_documents or [])
    return pipeline


def _config() -> SimpleNamespace:
    """Return config fields used by the pipeline and refiner."""
    return SimpleNamespace(
        evaluator_lower_threshold=0.35,
        evaluator_upper_threshold=0.7,
        refinement_max_strips_per_chunk=4,
        refinement_strip_max_chars=1000,
        refinement_top_internal_strips=10,
        refinement_top_external_strips=10,
        refinement_decompose_mode="selection",
    )


def _chunk(text: str, source_name: str) -> dict[str, Any]:
    """Build one retrieved chunk row."""
    return {
        "chunk_id": source_name,
        "source_name": source_name,
        "text": text,
        "score": 0.42,
    }


def _web_document(text: str, source_name: str) -> dict[str, str]:
    """Build one web document row."""
    return {
        "source_name": source_name,
        "url": f"https://example.test/{source_name}",
        "text": text,
    }


def _knowledge_sources(items: list[KnowledgeItem]) -> list[str]:
    """Return the source names passed into answer generation."""
    return [item.source_name for item in items]


class _FakeVectorDb:
    """Vector DB fake returning configured chunks."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Return configured chunks up to the requested limit."""
        self.queries.append((query, limit))
        return self.chunks[:limit]


class _FakeGrader:
    """Grader fake returning text-keyed scores."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.queries: list[str] = []

    def score_document(self, query: str, document_text: str) -> RelevanceEvaluation:
        """Return the configured score for one document text."""
        self.queries.append(query)
        return RelevanceEvaluation(score=self.scores[document_text], rationale="")


class _FakeGenerator:
    """Generator fake capturing generation knowledge."""

    def __init__(self) -> None:
        self.knowledge: list[KnowledgeItem] = []
        self.answer_query = ""
        self.original_query: str | None = None

    def reformulate_query(self, query: str) -> str:
        """Return a stable rewritten query."""
        return f"rewritten {query}"

    def generate_answer(
        self,
        query: str,
        relevant_chunks: list[KnowledgeItem],
        *,
        original_query: str | None = None,
    ) -> str:
        """Capture knowledge passed to generation."""
        self.answer_query = query
        self.original_query = original_query
        self.knowledge = list(relevant_chunks)
        return "answer"


class _FakeQueryRewriter:
    """Query rewriter fake returning one standalone question."""

    def __init__(self, rewritten: str) -> None:
        self.rewritten = rewritten
        self.calls: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    def rewrite(self, query: str, history: tuple[tuple[str, str], ...]) -> RewrittenQuery:
        """Record structured history and return the configured rewrite."""
        self.calls.append((query, history))
        return RewrittenQuery(original=query, rewritten=self.rewritten)


class _FakeWebSearch:
    """Web-search fake returning configured external documents."""

    def __init__(self, documents: list[dict[str, str]]) -> None:
        self.documents = documents
        self.queries: list[str] = []

    def search(self, query: str) -> list[dict[str, str]]:
        """Return configured documents and record the query."""
        self.queries.append(query)
        return self.documents
