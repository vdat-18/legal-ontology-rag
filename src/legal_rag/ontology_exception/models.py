"""Typed artifacts for legal-exception resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from legal_rag.ontology_population.models import PopulationEvidence

EXCEPTION_CANDIDATE_SCHEMA_VERSION = "0.1.0"
LEGACY_EXCEPTION_PROPOSAL_SCHEMA_VERSION = "0.1.0"
EXCEPTION_PROPOSAL_SCHEMA_VERSION = "0.2.0"
EXCEPTION_PROGRESS_SCHEMA_VERSION = "0.1.0"
LEGACY_LEGAL_EXCEPTION_RECORD_SCHEMA_VERSION = "0.1.0"
LEGAL_EXCEPTION_RECORD_SCHEMA_VERSION = "0.2.0"


@dataclass(frozen=True, slots=True)
class ExceptionCandidateWindow:
    """One bounded structural window requiring semantic exception confirmation."""

    candidate_id: str
    source_document_id: str
    general_population_ids: tuple[str, ...]
    target_population_ids: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable candidate window."""
        return {
            "candidate_id": self.candidate_id,
            "source_document_id": self.source_document_id,
            "general_population_ids": list(self.general_population_ids),
            "target_population_ids": list(self.target_population_ids),
            "source_chunk_ids": list(self.source_chunk_ids),
        }


@dataclass(frozen=True, slots=True)
class ExceptionCandidateSet:
    """One fingerprinted set of structural exception windows."""

    run_id: str
    windows: tuple[ExceptionCandidateWindow, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the stable candidate-set schema."""
        return {
            "schema_version": EXCEPTION_CANDIDATE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "windows": [window.to_dict() for window in self.windows],
        }


@dataclass(frozen=True, slots=True)
class ExceptionCitationResolution:
    """One grouped citation wording resolved to canonical target rule versions."""

    citation_text: str
    source_chunk_id: str
    target_population_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable grouped citation."""
        return {
            "citation_text": self.citation_text,
            "source_chunk_id": self.source_chunk_id,
            "target_population_ids": list(self.target_population_ids),
        }


@dataclass(frozen=True, slots=True)
class LegalExceptionProposal:
    """One typed model-proposed exception fragment."""

    proposal_id: str
    candidate_id: str
    general_population_id: str
    source_document_number: str
    exception_provision_path: str
    evidence: tuple[PopulationEvidence, ...]
    citations: tuple[ExceptionCitationResolution, ...]
    rationale_vi: str
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return the stable proposal schema."""
        return {
            "schema_version": EXCEPTION_PROPOSAL_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "candidate_id": self.candidate_id,
            "general_population_id": self.general_population_id,
            "source_document_number": self.source_document_number,
            "exception_provision_path": self.exception_provision_path,
            "evidence": [item.to_dict() for item in self.evidence],
            "citations": [citation.to_dict() for citation in self.citations],
            "rationale_vi": self.rationale_vi,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class ExceptionProposalProgress:
    """One exception candidate whose proposal task completed."""

    candidate_id: str
    run_id: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable proposal checkpoint."""
        return {
            "schema_version": EXCEPTION_PROGRESS_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class LegalExceptionRecord:
    """One canonical exception with grouped citations and redirects."""

    exception_id: str
    general_population_id: str
    source_document_number: str
    exception_provision_path: str
    evidence: tuple[PopulationEvidence, ...]
    citations: tuple[ExceptionCitationResolution, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable canonical exception schema."""
        return {
            "schema_version": LEGAL_EXCEPTION_RECORD_SCHEMA_VERSION,
            "exception_id": self.exception_id,
            "general_population_id": self.general_population_id,
            "source_document_number": self.source_document_number,
            "exception_provision_path": self.exception_provision_path,
            "evidence": [item.to_dict() for item in self.evidence],
            "citations": [citation.to_dict() for citation in self.citations],
        }
