"""Typed artifacts for cross-chunk rule assembly and human review."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from legal_rag.ontology_population.models import OntologyPopulationRecord

ASSEMBLY_CANDIDATE_SCHEMA_VERSION = "0.1.0"
ASSEMBLY_PROPOSAL_SCHEMA_VERSION = "0.1.0"
ASSEMBLY_PROPOSAL_PROGRESS_SCHEMA_VERSION = "0.1.0"
ASSEMBLY_REVIEW_SCHEMA_VERSION = "0.1.0"

ReviewDecision = Literal["pending", "accepted", "rejected"]


@dataclass(frozen=True, slots=True)
class AssemblyCandidateWindow:
    """One structurally bounded group requiring downstream semantic confirmation."""

    candidate_id: str
    population_ids: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]
    provision_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable candidate window."""
        return {
            "candidate_id": self.candidate_id,
            "population_ids": list(self.population_ids),
            "source_chunk_ids": list(self.source_chunk_ids),
            "provision_paths": list(self.provision_paths),
        }


@dataclass(frozen=True, slots=True)
class AssemblyCandidateSet:
    """One fingerprinted set of candidate windows."""

    run_id: str
    windows: tuple[AssemblyCandidateWindow, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the stable candidate-set schema."""
        return {
            "schema_version": ASSEMBLY_CANDIDATE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "windows": [window.to_dict() for window in self.windows],
        }


@dataclass(frozen=True, slots=True)
class AssemblyProposal:
    """One model-proposed assembly awaiting human review."""

    proposal_id: str
    candidate_id: str
    member_population_ids: tuple[str, ...]
    assembled_record: OntologyPopulationRecord
    rationale_vi: str
    proposed_by_model: str
    prompt_version: str
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return the stable proposal schema."""
        return {
            "schema_version": ASSEMBLY_PROPOSAL_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "candidate_id": self.candidate_id,
            "member_population_ids": list(self.member_population_ids),
            "assembled_record": self.assembled_record.to_dict(),
            "rationale_vi": self.rationale_vi,
            "proposed_by_model": self.proposed_by_model,
            "prompt_version": self.prompt_version,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class AssemblyProposalProgress:
    """One candidate window whose proposal task completed."""

    candidate_id: str
    run_id: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable proposal checkpoint."""
        return {
            "schema_version": ASSEMBLY_PROPOSAL_PROGRESS_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class AssemblyReview:
    """One human decision over a model-proposed assembly."""

    proposal_id: str
    decision: ReviewDecision
    reviewer: str
    reviewed_on: date | None
    note: str

    def to_dict(self) -> dict[str, object]:
        """Return the stable human-review schema."""
        return {
            "schema_version": ASSEMBLY_REVIEW_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "decision": self.decision,
            "reviewer": self.reviewer,
            "reviewed_on": self.reviewed_on.isoformat() if self.reviewed_on else None,
            "note": self.note,
        }
