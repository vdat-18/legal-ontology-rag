"""Track one candidate's progress from generation through evolution, trusting LLM output as-is."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from legal_rag.benchmark_synthesis.candidate_generation import CandidateDraft

CANDIDATE_PROGRESS_SCHEMA_VERSION = "3.0.0"


@dataclass(frozen=True, slots=True)
class CandidateProgressRecord:
    """Checkpoint for one scenario attempt across every candidate LLM stage."""

    task_id: str
    scenario_id: str
    attempt_index: int
    stage: str
    draft: CandidateDraft

    @property
    def completed(self) -> bool:
        """Return whether this task has reached a final candidate record."""
        return self.stage == "completed"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable checkpoint record."""
        return {
            "schema_version": CANDIDATE_PROGRESS_SCHEMA_VERSION,
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "attempt_index": self.attempt_index,
            "stage": self.stage,
            "draft": self.draft.to_dict(),
        }
