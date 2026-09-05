"""Typed artifacts for cross-chunk rule assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from legal_rag.ontology_population.models import OntologyPopulationRecord

ASSEMBLY_CANDIDATE_SCHEMA_VERSION = "0.1.0"
ASSEMBLY_PROPOSAL_PROGRESS_SCHEMA_VERSION = "0.1.0"
ASSEMBLY_RESOLUTION_PROGRESS_SCHEMA_VERSION = "0.1.0"
LEGACY_ASSEMBLY_PROPOSAL_SCHEMA_VERSION = "0.1.0"
ASSEMBLY_PROPOSAL_SCHEMA_VERSION = "0.2.0"


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
    """One typed model-proposed assembly."""

    proposal_id: str
    candidate_id: str
    member_population_ids: tuple[str, ...]
    assembled_record: OntologyPopulationRecord
    rationale_vi: str
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
class AssemblyConflictComponent:
    """One connected group of proposals that reuse population members."""

    component_id: str
    proposal_ids: tuple[str, ...]
    member_population_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssemblyResolutionProgress:
    """One conflicting proposal component resolved by the model."""

    component_id: str
    run_id: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable resolution checkpoint."""
        return {
            "schema_version": ASSEMBLY_RESOLUTION_PROGRESS_SCHEMA_VERSION,
            "component_id": self.component_id,
            "run_id": self.run_id,
        }
