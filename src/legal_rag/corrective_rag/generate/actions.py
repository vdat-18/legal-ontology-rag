"""Action trigger for CRAG retrieval confidence."""

from __future__ import annotations

from collections.abc import Sequence

from legal_rag.corrective_rag.generate.models import GradedChunk, KnowledgeAction


class ActionTrigger:
    """Map evaluator scores to CRAG actions."""

    def __init__(self, *, lower_threshold: float, upper_threshold: float) -> None:
        """Initialize score thresholds."""
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def classify_score(self, score: float) -> KnowledgeAction:
        """Classify one evaluator score."""
        if score >= self.upper_threshold:
            return "CORRECT"
        if score < self.lower_threshold:
            return "INCORRECT"
        return "AMBIGUOUS"

    def retrieval_action(self, graded_chunks: Sequence[GradedChunk]) -> KnowledgeAction:
        """Classify the overall retrieval result."""
        if not graded_chunks:
            return "INCORRECT"
        if any(chunk.action == "CORRECT" for chunk in graded_chunks):
            return "CORRECT"
        if any(chunk.action == "AMBIGUOUS" for chunk in graded_chunks):
            return "AMBIGUOUS"
        return "INCORRECT"
