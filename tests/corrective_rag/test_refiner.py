"""Tests for CRAG knowledge refinement."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from legal_rag.corrective_rag.generate.grader import RelevanceEvaluation
from legal_rag.corrective_rag.generate.models import GradedChunk, KnowledgeItem
from legal_rag.corrective_rag.generate.refiner import KnowledgeRefiner


def test_refine_internal_skips_disallowed_chunk_actions_before_scoring() -> None:
    """Internal refinement only evaluates chunks allowed by the current CRAG branch."""
    grader = _FakeGrader({"correct strip": 0.9, "ambiguous strip": 0.9})
    refiner = KnowledgeRefiner(_config(), grader)

    knowledge = refiner.refine_internal(
        "query",
        [
            GradedChunk(_chunk("correct strip", "correct-source"), score=0.9, action="CORRECT"),
            GradedChunk(_chunk("ambiguous strip", "ambiguous-source"), score=0.5, action="AMBIGUOUS"),
        ],
        allowed_actions={"CORRECT"},
    )

    assert _knowledge_sources(knowledge) == ["correct-source"]
    assert grader.scored_texts == ["correct strip"]


def test_refine_internal_filters_by_strip_action() -> None:
    """Strip-level scores must pass the action threshold requested by the branch."""
    grader = _FakeGrader(
        {
            "high strip": 0.9,
            "middle strip": 0.5,
            "zero strip": 0.0,
        }
    )
    refiner = KnowledgeRefiner(_config(), grader)

    knowledge = refiner.refine_internal(
        "query",
        [
            GradedChunk(_chunk("high strip", "high-source"), score=0.9, action="CORRECT"),
            GradedChunk(_chunk("middle strip", "middle-source"), score=0.9, action="CORRECT"),
            GradedChunk(_chunk("zero strip", "zero-source"), score=0.9, action="CORRECT"),
        ],
        allowed_actions={"CORRECT"},
        allowed_strip_actions={"AMBIGUOUS"},
    )

    assert _knowledge_sources(knowledge) == ["middle-source"]


def test_refine_external_does_not_select_zero_or_incorrect_strips() -> None:
    """External refinement must not fill top-N with score-0 or below-threshold knowledge."""
    grader = _FakeGrader(
        {
            "strong web strip": 0.8,
            "weak web strip": 0.2,
            "zero web strip": 0.0,
        }
    )
    refiner = KnowledgeRefiner(_config(), grader)

    knowledge = refiner.refine_external(
        "query",
        [
            _web_document("strong web strip", "strong-source"),
            _web_document("weak web strip", "weak-source"),
            _web_document("zero web strip", "zero-source"),
        ],
    )

    assert _knowledge_sources(knowledge) == ["strong-source"]


def _config() -> SimpleNamespace:
    """Return config fields used by the refiner."""
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
    """Build one external document row."""
    return {
        "source_name": source_name,
        "url": f"https://example.test/{source_name}",
        "text": text,
    }


def _knowledge_sources(items: list[KnowledgeItem]) -> list[str]:
    """Return selected source names."""
    return [item.source_name for item in items]


class _FakeGrader:
    """Grader fake returning text-keyed scores."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.scored_texts: list[str] = []

    def score_document(self, query: str, document_text: str) -> RelevanceEvaluation:
        """Return the configured score and record scored text."""
        self.scored_texts.append(document_text)
        return RelevanceEvaluation(score=self.scores[document_text], rationale="")
