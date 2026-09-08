"""Persistence helpers for concept-resolution artifacts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from legal_rag.io_utils import write_json_atomic, write_jsonl_atomic
from legal_rag.ontology_concept.models import (
    CANONICAL_CONCEPT_SCHEMA_VERSION,
    CANONICAL_RELATION_SCHEMA_VERSION,
    CONCEPT_CANDIDATE_SCHEMA_VERSION,
    CONCEPT_PROGRESS_SCHEMA_VERSION,
    CONCEPT_PROPOSAL_SCHEMA_VERSION,
    LEGACY_CANONICAL_CONCEPT_SCHEMA_VERSION,
    LEGACY_CANONICAL_RELATION_SCHEMA_VERSION,
    LEGACY_CONCEPT_PROPOSAL_SCHEMA_VERSION,
    CanonicalConceptRecord,
    CanonicalConceptRelationRecord,
    ConceptCandidateInventory,
    ConceptCandidateSet,
    ConceptOccurrence,
    ConceptProposalProgress,
    ConceptRelationProposal,
)
from legal_rag.ontology_population.models import PopulationEvidence

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
        inventories=tuple(
            ConceptCandidateInventory(
                candidate_id=item["candidate_id"],
                kind=item["kind"],
                occurrence_ids=tuple(item["occurrence_ids"]),
            )
            for item in payload["inventories"]
        ),
    )


def write_concept_proposals(path: Path, proposals: Sequence[ConceptRelationProposal]) -> None:
    """Atomically replace concept proposals."""
    write_jsonl_atomic(path, (item.to_dict() for item in proposals))


def read_concept_proposals(path: Path) -> tuple[ConceptRelationProposal, ...]:
    """Read typed concept proposals."""
    return tuple(
        _proposal(payload)
        for payload in _read_jsonl(path, {LEGACY_CONCEPT_PROPOSAL_SCHEMA_VERSION, CONCEPT_PROPOSAL_SCHEMA_VERSION})
    )


def write_concept_progress(path: Path, progress: Sequence[ConceptProposalProgress]) -> None:
    """Atomically replace concept proposal checkpoints."""
    write_jsonl_atomic(path, (item.to_dict() for item in progress))


def read_concept_progress(path: Path) -> tuple[ConceptProposalProgress, ...]:
    """Read concept proposal checkpoints."""
    return tuple(
        ConceptProposalProgress(candidate_id=item["candidate_id"], run_id=item["run_id"])
        for item in _read_jsonl(path, {CONCEPT_PROGRESS_SCHEMA_VERSION})
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
        )
        for item in _read_jsonl(path, {LEGACY_CANONICAL_CONCEPT_SCHEMA_VERSION, CANONICAL_CONCEPT_SCHEMA_VERSION})
    )


def write_canonical_relations(path: Path, relations: Sequence[CanonicalConceptRelationRecord]) -> None:
    """Atomically replace canonical concept relations."""
    write_jsonl_atomic(path, (item.to_dict() for item in relations))


def read_canonical_relations(path: Path) -> tuple[CanonicalConceptRelationRecord, ...]:
    """Read canonical concept relation records."""
    return tuple(
        CanonicalConceptRelationRecord(
            relation_id=item["relation_id"],
            source_concept_id=item["source_concept_id"],
            target_concept_id=item["target_concept_id"],
            relation=item["relation"],
        )
        for item in _read_jsonl(path, {LEGACY_CANONICAL_RELATION_SCHEMA_VERSION, CANONICAL_RELATION_SCHEMA_VERSION})
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
        run_id=str(item["run_id"]),
    )


def _evidence(items: Sequence[dict[str, str]]) -> tuple[PopulationEvidence, ...]:
    """Translate persisted exact evidence entries."""
    return tuple(PopulationEvidence(chunk_id=item["chunk_id"], exact_text=item["exact_text"]) for item in items)


def _read_jsonl(path: Path, schema_versions: set[str]) -> tuple[dict[str, object], ...]:
    """Read JSONL rows with supported schema versions."""
    rows = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            item = json.loads(line)
            if item["schema_version"] not in schema_versions:
                msg = f"Unsupported concept artifact schema: {item['schema_version']}"
                raise ValueError(msg)
            rows.append(item)
    return tuple(rows)
