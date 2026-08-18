"""Typed artifacts for reviewed ontology concept resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from legal_rag.ontology_population.models import PopulationEvidence

CONCEPT_CANDIDATE_SCHEMA_VERSION = "0.1.0"
CONCEPT_PROPOSAL_SCHEMA_VERSION = "0.1.0"
CONCEPT_PROGRESS_SCHEMA_VERSION = "0.1.0"
CONCEPT_REVIEW_SCHEMA_VERSION = "0.1.0"
CANONICAL_CONCEPT_SCHEMA_VERSION = "0.1.0"
CANONICAL_RELATION_SCHEMA_VERSION = "0.1.0"

ConceptKind = Literal["RegulatedConduct", "LegalSubjectRole", "VehicleCategory"]
ConceptRelation = Literal["SameConcept", "LeftBroaderThanRight", "LeftNarrowerThanRight", "Related", "Distinct"]
CanonicalRelation = Literal["Broader", "Related", "Distinct"]
ReviewDecision = Literal["pending", "accepted", "rejected"]


@dataclass(frozen=True, slots=True)
class ConceptOccurrence:
    """One rule-scoped concept occurrence with source evidence."""

    occurrence_id: str
    kind: ConceptKind
    label_vi: str
    population_id: str
    document_number: str
    provision_path: str
    evidence: tuple[PopulationEvidence, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable occurrence."""
        return {
            "occurrence_id": self.occurrence_id,
            "kind": self.kind,
            "label_vi": self.label_vi,
            "population_id": self.population_id,
            "document_number": self.document_number,
            "provision_path": self.provision_path,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class ConceptCandidatePair:
    """One same-kind pair requiring semantic relation confirmation."""

    candidate_id: str
    left_occurrence_id: str
    right_occurrence_id: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable candidate pair."""
        return {
            "candidate_id": self.candidate_id,
            "left_occurrence_id": self.left_occurrence_id,
            "right_occurrence_id": self.right_occurrence_id,
        }


@dataclass(frozen=True, slots=True)
class ConceptCandidateSet:
    """One fingerprinted exhaustive set of same-kind concept pairs."""

    run_id: str
    occurrences: tuple[ConceptOccurrence, ...]
    pairs: tuple[ConceptCandidatePair, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the stable candidate-set schema."""
        return {
            "schema_version": CONCEPT_CANDIDATE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "occurrences": [item.to_dict() for item in self.occurrences],
            "pairs": [item.to_dict() for item in self.pairs],
        }


@dataclass(frozen=True, slots=True)
class ConceptRelationProposal:
    """One model-proposed concept relation awaiting human review."""

    proposal_id: str
    candidate_id: str
    left_occurrence_id: str
    right_occurrence_id: str
    relation: ConceptRelation
    preferred_label_vi: str
    left_evidence: tuple[PopulationEvidence, ...]
    right_evidence: tuple[PopulationEvidence, ...]
    rationale_vi: str
    proposed_by_model: str
    prompt_version: str
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return the stable proposal schema."""
        return {
            "schema_version": CONCEPT_PROPOSAL_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "candidate_id": self.candidate_id,
            "left_occurrence_id": self.left_occurrence_id,
            "right_occurrence_id": self.right_occurrence_id,
            "relation": self.relation,
            "preferred_label_vi": self.preferred_label_vi,
            "left_evidence": [item.to_dict() for item in self.left_evidence],
            "right_evidence": [item.to_dict() for item in self.right_evidence],
            "rationale_vi": self.rationale_vi,
            "proposed_by_model": self.proposed_by_model,
            "prompt_version": self.prompt_version,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class ConceptProposalProgress:
    """One candidate pair whose model task completed."""

    candidate_id: str
    run_id: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable checkpoint."""
        return {
            "schema_version": CONCEPT_PROGRESS_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class ConceptReview:
    """One attributable human decision over a concept proposal."""

    proposal_id: str
    decision: ReviewDecision
    reviewer: str
    reviewed_on: date | None
    note: str

    def to_dict(self) -> dict[str, object]:
        """Return the stable review schema."""
        return {
            "schema_version": CONCEPT_REVIEW_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "decision": self.decision,
            "reviewer": self.reviewer,
            "reviewed_on": self.reviewed_on.isoformat() if self.reviewed_on else None,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class CanonicalConceptRecord:
    """One reviewed canonical concept identity and its source occurrences."""

    concept_id: str
    kind: ConceptKind
    preferred_label_vi: str
    alternative_labels_vi: tuple[str, ...]
    occurrence_ids: tuple[str, ...]
    accepted_proposal_ids: tuple[str, ...]
    reviewers: tuple[str, ...]
    reviewed_on: tuple[date, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable canonical-concept schema."""
        return {
            "schema_version": CANONICAL_CONCEPT_SCHEMA_VERSION,
            "concept_id": self.concept_id,
            "kind": self.kind,
            "preferred_label_vi": self.preferred_label_vi,
            "alternative_labels_vi": list(self.alternative_labels_vi),
            "occurrence_ids": list(self.occurrence_ids),
            "accepted_proposal_ids": list(self.accepted_proposal_ids),
            "reviewers": list(self.reviewers),
            "reviewed_on": [value.isoformat() for value in self.reviewed_on],
        }


@dataclass(frozen=True, slots=True)
class CanonicalConceptRelationRecord:
    """One reviewed relation between distinct canonical concepts."""

    relation_id: str
    source_concept_id: str
    target_concept_id: str
    relation: CanonicalRelation
    accepted_proposal_ids: tuple[str, ...]
    reviewers: tuple[str, ...]
    reviewed_on: tuple[date, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable canonical-relation schema."""
        return {
            "schema_version": CANONICAL_RELATION_SCHEMA_VERSION,
            "relation_id": self.relation_id,
            "source_concept_id": self.source_concept_id,
            "target_concept_id": self.target_concept_id,
            "relation": self.relation,
            "accepted_proposal_ids": list(self.accepted_proposal_ids),
            "reviewers": list(self.reviewers),
            "reviewed_on": [value.isoformat() for value in self.reviewed_on],
        }
