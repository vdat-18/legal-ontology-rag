"""Combine a candidate draft with its gate outcomes into the final persisted record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from legal_rag.benchmark_synthesis.binding_gate import BindingCheckResult
from legal_rag.benchmark_synthesis.candidate_generation import CandidateDraft
from legal_rag.benchmark_synthesis.verification import VerificationResult


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
        record["binding_check"] = self.binding.to_dict()
        record["verification"] = self.verification.to_dict() if self.verification is not None else None
        record["review_status"] = self.review_status
        return record
