"""Knowledge refinement for CRAG."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from loguru import logger

from legal_rag.corrective_rag.generate.actions import ActionTrigger
from legal_rag.corrective_rag.generate.config import RAGConfig
from legal_rag.corrective_rag.generate.grader import CorrectiveGrader
from legal_rag.corrective_rag.generate.models import GradedChunk, KnowledgeAction, KnowledgeItem, KnowledgeKind

_USABLE_KNOWLEDGE_ACTIONS: set[KnowledgeAction] = {"CORRECT", "AMBIGUOUS"}


class KnowledgeRefiner:
    """Decompose, evaluator-rank, and recompose retrieved knowledge."""

    def __init__(self, config: RAGConfig, grader: CorrectiveGrader) -> None:
        """Initialize the refiner."""
        self.config = config
        self.grader = grader
        self.action_trigger = ActionTrigger(
            lower_threshold=config.evaluator_lower_threshold,
            upper_threshold=config.evaluator_upper_threshold,
        )

    def refine_internal(
        self,
        query: str,
        graded_chunks: Sequence[GradedChunk],
        *,
        allowed_actions: set[KnowledgeAction] | None = None,
        allowed_strip_actions: set[KnowledgeAction] | None = None,
    ) -> list[KnowledgeItem]:
        """Select top internal knowledge strips from retrieved chunks."""
        if allowed_strip_actions is None:
            allowed_strip_actions = _USABLE_KNOWLEDGE_ACTIONS
        candidates: list[_CandidateStrip] = []
        for chunk_position, graded_chunk in enumerate(graded_chunks):
            if allowed_actions is not None and graded_chunk.action not in allowed_actions:
                continue
            chunk = graded_chunk.chunk
            candidates.extend(
                self._score_text_strips(
                    query=query,
                    text=str(chunk.get("text", "")),
                    source_name=str(chunk.get("source_name", "unknown")),
                    kind="internal",
                    chunk_id=str(chunk.get("chunk_id") or "") or None,
                    url=None,
                    source_position=chunk_position,
                    allowed_actions=allowed_strip_actions,
                )
            )

        knowledge = _select_relevant_knowledge(candidates, top_n=self.config.refinement_top_internal_strips)
        logger.info("Refined {} internal knowledge items.", len(knowledge))
        return knowledge

    def refine_external(
        self,
        query: str,
        documents: Sequence[Mapping[str, str]],
        *,
        allowed_strip_actions: set[KnowledgeAction] | None = None,
    ) -> list[KnowledgeItem]:
        """Select top external knowledge strips from web documents."""
        if allowed_strip_actions is None:
            allowed_strip_actions = _USABLE_KNOWLEDGE_ACTIONS
        candidates: list[_CandidateStrip] = []
        for source_position, document in enumerate(documents):
            candidates.extend(
                self._score_text_strips(
                    query=query,
                    text=document.get("text", ""),
                    source_name=document.get("source_name", document.get("url", "external")),
                    kind="external",
                    chunk_id=None,
                    url=document.get("url"),
                    source_position=source_position,
                    allowed_actions=allowed_strip_actions,
                )
            )

        knowledge = _select_relevant_knowledge(candidates, top_n=self.config.refinement_top_external_strips)
        logger.info("Refined {} external knowledge items.", len(knowledge))
        return knowledge

    def _score_text_strips(
        self,
        *,
        query: str,
        text: str,
        source_name: str,
        kind: KnowledgeKind,
        chunk_id: str | None,
        url: str | None,
        source_position: int,
        allowed_actions: set[KnowledgeAction],
    ) -> list[_CandidateStrip]:
        strips = _split_strips(
            text,
            max_chars=self.config.refinement_strip_max_chars,
            mode=self.config.refinement_decompose_mode,
        )
        strips = strips[: self.config.refinement_max_strips_per_chunk]
        candidates: list[_CandidateStrip] = []
        for strip_position, strip in enumerate(strips):
            evaluation = self.grader.score_document(query, strip)
            action = self.action_trigger.classify_score(evaluation.score)
            if action not in allowed_actions:
                continue
            candidates.append(
                _CandidateStrip(
                    text=strip,
                    source_name=source_name,
                    score=evaluation.score,
                    action=action,
                    kind=kind,
                    source_position=source_position,
                    strip_position=strip_position,
                    chunk_id=chunk_id,
                    url=url,
                )
            )
        return candidates


@dataclass(frozen=True, slots=True)
class _CandidateStrip:
    """One evaluator-scored knowledge strip before top-N selection."""

    text: str
    source_name: str
    score: float
    action: KnowledgeAction
    kind: KnowledgeKind
    source_position: int
    strip_position: int
    chunk_id: str | None = None
    url: str | None = None


def _select_relevant_knowledge(candidates: Sequence[_CandidateStrip], *, top_n: int) -> list[KnowledgeItem]:
    """Return top evaluator-scored strips as CRAG knowledge."""
    if top_n <= 0:
        return []

    relevant_candidates = [item for item in candidates if item.score > 0.0]
    selected = sorted(relevant_candidates, key=lambda item: (-item.score, item.source_position, item.strip_position))[
        :top_n
    ]
    return [
        KnowledgeItem(
            text=item.text,
            source_name=item.source_name,
            score=item.score,
            kind=item.kind,
            chunk_id=item.chunk_id,
            url=item.url,
        )
        for item in selected
    ]


def _split_strips(text: str, *, max_chars: int, mode: str) -> list[str]:
    """Split text into CRAG knowledge strips."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    if mode == "selection":
        return [normalized]
    if mode == "fixed_num":
        return _split_fixed_word_windows(normalized)
    return _split_excerption_strips(normalized, max_chars=max_chars)


def _split_fixed_word_windows(text: str) -> list[str]:
    """Split text into fixed word windows like the CRAG reference implementation."""
    final_strips: list[str] = []
    window_length = 50
    buffer: list[str] = []
    for word in text.split():
        buffer.append(word)
        if len(buffer) == window_length:
            final_strips.append(" ".join(buffer))
            buffer = []
    if buffer:
        if final_strips and len(buffer) < 10:
            final_strips[-1] = f"{final_strips[-1]} {' '.join(buffer)}"
        else:
            final_strips.append(" ".join(buffer))
    return final_strips


def _split_excerption_strips(text: str, *, max_chars: int) -> list[str]:
    """Split text by sentence-like excerpts and recompose nearby excerpts."""
    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]

    strips: list[str] = []
    buffer: list[str] = []
    for sentence in sentences:
        candidate = " ".join((*buffer, sentence))
        if buffer and (len(buffer) == 3 or len(candidate) > max_chars):
            strips.append(" ".join(buffer))
            buffer = []
        buffer.append(sentence)
    if buffer:
        strips.append(" ".join(buffer))
    return strips


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units."""
    return [part.strip() for part in re.split(r"(?<=[.!?;:])\s+", text) if part.strip()]
