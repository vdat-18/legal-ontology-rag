"""Typed artifacts for cross-document legal-change resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from legal_rag.ontology_population.models import PopulationEvidence

CHANGE_CANDIDATE_SCHEMA_VERSION = "0.1.0"
CHANGE_PROPOSAL_SCHEMA_VERSION = "0.1.0"
CHANGE_PROGRESS_SCHEMA_VERSION = "0.1.0"
CHANGE_REVIEW_SCHEMA_VERSION = "0.1.0"
LEGAL_CHANGE_EVENT_SCHEMA_VERSION = "0.1.0"

LegalChangeType = Literal["Amendment", "Replacement", "RepealOrSupersession"]
ReviewDecision = Literal["pending", "accepted", "rejected"]


@dataclass(frozen=True, slots=True)
class ChangeCandidateWindow:
    """One bounded cross-document window requiring semantic confirmation."""

    candidate_id: str
    earlier_document_id: str
    later_document_id: str
    earlier_population_ids: tuple[str, ...]
    later_population_ids: tuple[str, ...]
    change_source_chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable candidate window."""
        return {
            "candidate_id": self.candidate_id,
            "earlier_document_id": self.earlier_document_id,
            "later_document_id": self.later_document_id,
            "earlier_population_ids": list(self.earlier_population_ids),
            "later_population_ids": list(self.later_population_ids),
            "change_source_chunk_ids": list(self.change_source_chunk_ids),
        }


@dataclass(frozen=True, slots=True)
class ChangeCandidateSet:
    """One fingerprinted set of bounded cross-document windows."""

    run_id: str
    windows: tuple[ChangeCandidateWindow, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the stable candidate-set schema."""
        return {
            "schema_version": CHANGE_CANDIDATE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "windows": [window.to_dict() for window in self.windows],
        }


@dataclass(frozen=True, slots=True)
class LegalChangeProposal:
    """One model-proposed legal change awaiting human review."""

    proposal_id: str
    candidate_id: str
    earlier_population_id: str
    later_population_id: str | None
    change_type: LegalChangeType
    effective_on: date
    change_document_number: str
    change_provision_path: str
    evidence: tuple[PopulationEvidence, ...]
    rationale_vi: str
    proposed_by_model: str
    prompt_version: str
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return the stable proposal schema."""
        return {
            "schema_version": CHANGE_PROPOSAL_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "candidate_id": self.candidate_id,
            "earlier_population_id": self.earlier_population_id,
            "later_population_id": self.later_population_id,
            "change_type": self.change_type,
            "effective_on": self.effective_on.isoformat(),
            "change_document_number": self.change_document_number,
            "change_provision_path": self.change_provision_path,
            "evidence": [item.to_dict() for item in self.evidence],
            "rationale_vi": self.rationale_vi,
            "proposed_by_model": self.proposed_by_model,
            "prompt_version": self.prompt_version,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class ChangeProposalProgress:
    """One candidate window whose proposal task completed."""

    candidate_id: str
    run_id: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable proposal checkpoint."""
        return {
            "schema_version": CHANGE_PROGRESS_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class ChangeReview:
    """One human decision over a proposed legal change."""

    proposal_id: str
    decision: ReviewDecision
    reviewer: str
    reviewed_on: date | None
    note: str

    def to_dict(self) -> dict[str, object]:
        """Return the stable human-review schema."""
        return {
            "schema_version": CHANGE_REVIEW_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "decision": self.decision,
            "reviewer": self.reviewer,
            "reviewed_on": self.reviewed_on.isoformat() if self.reviewed_on else None,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class LegalChangeEvent:
    """One accepted change linked to canonical rule identities."""

    change_id: str
    change_type: LegalChangeType
    effective_on: date
    earlier_population_id: str
    later_population_id: str | None
    change_document_number: str
    change_provision_path: str
    evidence: tuple[PopulationEvidence, ...]
    proposal_id: str
    reviewer: str
    reviewed_on: date

    def to_dict(self) -> dict[str, Any]:
        """Return the stable accepted-change schema."""
        return {
            "schema_version": LEGAL_CHANGE_EVENT_SCHEMA_VERSION,
            "change_id": self.change_id,
            "change_type": self.change_type,
            "effective_on": self.effective_on.isoformat(),
            "earlier_population_id": self.earlier_population_id,
            "later_population_id": self.later_population_id,
            "change_document_number": self.change_document_number,
            "change_provision_path": self.change_provision_path,
            "evidence": [item.to_dict() for item in self.evidence],
            "proposal_id": self.proposal_id,
            "reviewer": self.reviewer,
            "reviewed_on": self.reviewed_on.isoformat(),
        }
