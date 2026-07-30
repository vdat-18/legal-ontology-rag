"""Combine a candidate draft with its gate outcomes into the final persisted record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from legal_rag.benchmark_synthesis.binding_gate import BindingCheckResult
from legal_rag.benchmark_synthesis.candidate_generation import CandidateDraft
from legal_rag.benchmark_synthesis.verification import VerificationResult

CANDIDATE_PROGRESS_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One candidate combined with its gate 1 outcome and, when reached, its gate 2 outcome."""

    draft: CandidateDraft
    binding: BindingCheckResult
    verification: VerificationResult | None

    @property
    def review_status(self) -> str:
        """Return where this candidate landed in the automated gate pipeline."""
        if not self.binding.passed:
            return "rejected_by_binding_gate"
        if self.verification is None:
            return "binding_verified"
        return "automatically_verified" if self.verification.passed else "rejected_by_verification"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate record with both gate outcomes attached."""
        record = self.draft.to_dict()
        record["requested_query_type"] = self.draft.query_type
        record["requested_answerability"] = self.draft.answerability
        record["requested_reasoning_hops"] = self.draft.reasoning_hops
        if self.verification is not None:
            record["query_type"] = self.verification.observed_query_type
            record["answerability"] = self.verification.observed_answerability
            record["reasoning_hops"] = self.verification.necessary_reasoning_hops
        record["binding_check"] = self.binding.to_dict()
        record["verification"] = self.verification.to_dict() if self.verification is not None else None
        record["review_status"] = self.review_status
        return record


@dataclass(frozen=True, slots=True)
class CandidateProgressRecord:
    """Checkpoint for one scenario attempt across every candidate LLM stage."""

    task_id: str
    scenario_id: str
    attempt_index: int
    stage: str
    draft: CandidateDraft
    binding: BindingCheckResult | None = None
    verification: VerificationResult | None = None

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
            "binding": self.binding.to_dict() if self.binding is not None else None,
            "verification": self.verification.to_dict() if self.verification is not None else None,
        }

    def candidate_record(self) -> CandidateRecord:
        """Return the final candidate represented by a completed checkpoint."""
        if not self.completed or self.binding is None:
            msg = f"Candidate task {self.task_id} is not complete"
            raise ValueError(msg)
        return CandidateRecord(draft=self.draft, binding=self.binding, verification=self.verification)
