"""Persistence helpers for concept-resolution artifacts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from legal_rag.io_utils import write_json_atomic, write_jsonl_atomic
from legal_rag.ontology_concept.models import (
    CANONICAL_CONCEPT_SCHEMA_VERSION,
    CANONICAL_RELATION_SCHEMA_VERSION,
    CONCEPT_CANDIDATE_SCHEMA_VERSION,
    CONCEPT_PROGRESS_SCHEMA_VERSION,
    CONCEPT_PROPOSAL_SCHEMA_VERSION,
    CONCEPT_REVIEW_SCHEMA_VERSION,
    CanonicalConceptRecord,
    CanonicalConceptRelationRecord,
    ConceptCandidatePair,
    ConceptCandidateSet,
    ConceptOccurrence,
    ConceptProposalProgress,
    ConceptRelationProposal,
    ConceptReview,
)
from legal_rag.ontology_population.models import PopulationEvidence

REVIEW_DECISIONS = {"pending", "accepted", "rejected"}
CONCEPT_RELATIONS = {"SameConcept", "LeftBroaderThanRight", "LeftNarrowerThanRight", "Related", "Distinct"}


def write_concept_candidates(path: Path, candidates: ConceptCandidateSet) -> None:
    """Atomically replace the concept candidate set."""
    write_json_atomic(path, candidates.to_dict())


def read_concept_candidates(path: Path) -> ConceptCandidateSet:
    """Read one typed concept candidate set."""
    with path.open(encoding="utf-8") as source_file:
        payload = json.load(source_file)
    if payload["schema_version"] != CONCEPT_CANDIDATE_SCHEMA_VERSION:
        msg = f"Unsupported concept candidate schema: {payload['schema_version']}"
        raise ValueError(msg)
    return ConceptCandidateSet(
        run_id=payload["run_id"],
        occurrences=tuple(
            ConceptOccurrence(
                occurrence_id=item["occurrence_id"],
                kind=item["kind"],
                label_vi=item["label_vi"],
                population_id=item["population_id"],
                document_number=item["document_number"],
                provision_path=item["provision_path"],
                evidence=_evidence(item["evidence"]),
            )
            for item in payload["occurrences"]
        ),
        pairs=tuple(
            ConceptCandidatePair(
                candidate_id=item["candidate_id"],
                left_occurrence_id=item["left_occurrence_id"],
                right_occurrence_id=item["right_occurrence_id"],
            )
            for item in payload["pairs"]
        ),
    )


def write_concept_proposals(path: Path, proposals: Sequence[ConceptRelationProposal]) -> None:
    """Atomically replace reviewable concept proposals."""
    write_jsonl_atomic(path, (item.to_dict() for item in proposals))


def read_concept_proposals(path: Path) -> tuple[ConceptRelationProposal, ...]:
    """Read typed concept proposals."""
    return tuple(_proposal(payload) for payload in _read_jsonl(path, CONCEPT_PROPOSAL_SCHEMA_VERSION))


def write_concept_progress(path: Path, progress: Sequence[ConceptProposalProgress]) -> None:
    """Atomically replace concept proposal checkpoints."""
    write_jsonl_atomic(path, (item.to_dict() for item in progress))


def read_concept_progress(path: Path) -> tuple[ConceptProposalProgress, ...]:
    """Read concept proposal checkpoints."""
    return tuple(
        ConceptProposalProgress(candidate_id=item["candidate_id"], run_id=item["run_id"])
        for item in _read_jsonl(path, CONCEPT_PROGRESS_SCHEMA_VERSION)
    )


def write_concept_reviews(path: Path, reviews: Sequence[ConceptReview]) -> None:
    """Atomically replace human concept reviews."""
    write_jsonl_atomic(path, (item.to_dict() for item in reviews))


def read_concept_reviews(path: Path) -> tuple[ConceptReview, ...]:
    """Read typed human concept reviews."""
    reviews = []
    for item in _read_jsonl(path, CONCEPT_REVIEW_SCHEMA_VERSION):
        if item["decision"] not in REVIEW_DECISIONS:
            msg = f"Unsupported concept review decision: {item['decision']}"
            raise ValueError(msg)
        reviews.append(
            ConceptReview(
                proposal_id=item["proposal_id"],
                decision=item["decision"],
                reviewer=item["reviewer"],
                reviewed_on=date.fromisoformat(item["reviewed_on"]) if item["reviewed_on"] else None,
                note=item["note"],
            )
        )
    return tuple(reviews)


def pending_concept_review_template(proposals: Sequence[ConceptRelationProposal]) -> tuple[ConceptReview, ...]:
    """Return one pending human-review row for every proposal."""
    return tuple(
        ConceptReview(proposal_id=item.proposal_id, decision="pending", reviewer="", reviewed_on=None, note="")
        for item in proposals
    )


def write_canonical_concepts(path: Path, concepts: Sequence[CanonicalConceptRecord]) -> None:
    """Atomically replace canonical concept records."""
    write_jsonl_atomic(path, (item.to_dict() for item in concepts))


def read_canonical_concepts(path: Path) -> tuple[CanonicalConceptRecord, ...]:
    """Read canonical concept records."""
    return tuple(
        CanonicalConceptRecord(
            concept_id=item["concept_id"],
            kind=item["kind"],
            preferred_label_vi=item["preferred_label_vi"],
            alternative_labels_vi=tuple(item["alternative_labels_vi"]),
            occurrence_ids=tuple(item["occurrence_ids"]),
            accepted_proposal_ids=tuple(item["accepted_proposal_ids"]),
            reviewers=tuple(item["reviewers"]),
            reviewed_on=tuple(date.fromisoformat(value) for value in item["reviewed_on"]),
        )
        for item in _read_jsonl(path, CANONICAL_CONCEPT_SCHEMA_VERSION)
    )


def write_canonical_relations(path: Path, relations: Sequence[CanonicalConceptRelationRecord]) -> None:
    """Atomically replace reviewed canonical concept relations."""
    write_jsonl_atomic(path, (item.to_dict() for item in relations))


def read_canonical_relations(path: Path) -> tuple[CanonicalConceptRelationRecord, ...]:
    """Read canonical concept relation records."""
    return tuple(
        CanonicalConceptRelationRecord(
            relation_id=item["relation_id"],
            source_concept_id=item["source_concept_id"],
            target_concept_id=item["target_concept_id"],
            relation=item["relation"],
            accepted_proposal_ids=tuple(item["accepted_proposal_ids"]),
            reviewers=tuple(item["reviewers"]),
            reviewed_on=tuple(date.fromisoformat(value) for value in item["reviewed_on"]),
        )
        for item in _read_jsonl(path, CANONICAL_RELATION_SCHEMA_VERSION)
    )


def _proposal(item: dict[str, object]) -> ConceptRelationProposal:
    """Translate one persisted proposal payload."""
    if item["relation"] not in CONCEPT_RELATIONS:
        msg = f"Unsupported concept proposal relation: {item['relation']}"
        raise ValueError(msg)
    return ConceptRelationProposal(
        proposal_id=str(item["proposal_id"]),
        candidate_id=str(item["candidate_id"]),
        left_occurrence_id=str(item["left_occurrence_id"]),
        right_occurrence_id=str(item["right_occurrence_id"]),
        relation=item["relation"],
        preferred_label_vi=str(item["preferred_label_vi"]),
        left_evidence=_evidence(item["left_evidence"]),
        right_evidence=_evidence(item["right_evidence"]),
        rationale_vi=str(item["rationale_vi"]),
        proposed_by_model=str(item["proposed_by_model"]),
        prompt_version=str(item["prompt_version"]),
        run_id=str(item["run_id"]),
    )


def _evidence(items: Sequence[dict[str, str]]) -> tuple[PopulationEvidence, ...]:
    """Translate persisted exact evidence entries."""
    return tuple(PopulationEvidence(chunk_id=item["chunk_id"], exact_text=item["exact_text"]) for item in items)


def _read_jsonl(path: Path, schema_version: str) -> tuple[dict[str, object], ...]:
    """Read JSONL rows with one expected schema version."""
    rows = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            item = json.loads(line)
            if item["schema_version"] != schema_version:
                msg = f"Unsupported concept artifact schema: {item['schema_version']}"
                raise ValueError(msg)
            rows.append(item)
    return tuple(rows)
