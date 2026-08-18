"""Persistence for legal-change candidates, proposals, reviews, and events."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from legal_rag.io_utils import write_json_atomic, write_jsonl_atomic
from legal_rag.ontology_change.models import (
    CHANGE_CANDIDATE_SCHEMA_VERSION,
    CHANGE_PROGRESS_SCHEMA_VERSION,
    CHANGE_PROPOSAL_SCHEMA_VERSION,
    CHANGE_REVIEW_SCHEMA_VERSION,
    LEGAL_CHANGE_EVENT_SCHEMA_VERSION,
    ChangeCandidateSet,
    ChangeCandidateWindow,
    ChangeProposalProgress,
    ChangeReview,
    LegalChangeEvent,
    LegalChangeProposal,
)
from legal_rag.ontology_population.models import PopulationEvidence

CHANGE_TYPES = {"Amendment", "Replacement", "RepealOrSupersession"}
REVIEW_DECISIONS = {"pending", "accepted", "rejected"}


def write_change_candidates(path: Path, candidates: ChangeCandidateSet) -> None:
    """Atomically replace one legal-change candidate-set artifact."""
    write_json_atomic(path, candidates.to_dict())


def read_change_candidates(path: Path) -> ChangeCandidateSet:
    """Read one fingerprinted legal-change candidate set."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != CHANGE_CANDIDATE_SCHEMA_VERSION:
        msg = f"Unsupported legal-change candidate schema: {payload['schema_version']}"
        raise ValueError(msg)
    return ChangeCandidateSet(
        run_id=payload["run_id"],
        windows=tuple(
            ChangeCandidateWindow(
                candidate_id=item["candidate_id"],
                earlier_document_id=item["earlier_document_id"],
                later_document_id=item["later_document_id"],
                earlier_population_ids=tuple(item["earlier_population_ids"]),
                later_population_ids=tuple(item["later_population_ids"]),
                change_source_chunk_ids=tuple(item["change_source_chunk_ids"]),
            )
            for item in payload["windows"]
        ),
    )


def write_change_proposals(path: Path, proposals: Sequence[LegalChangeProposal]) -> None:
    """Atomically replace reviewable legal-change proposals."""
    write_jsonl_atomic(path, (proposal.to_dict() for proposal in proposals))


def read_change_proposals(path: Path) -> tuple[LegalChangeProposal, ...]:
    """Read typed legal-change proposals."""
    proposals = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != CHANGE_PROPOSAL_SCHEMA_VERSION:
                msg = f"Unsupported legal-change proposal schema: {payload['schema_version']}"
                raise ValueError(msg)
            if payload["change_type"] not in CHANGE_TYPES:
                msg = f"Unsupported legal-change type: {payload['change_type']}"
                raise ValueError(msg)
            proposals.append(
                LegalChangeProposal(
                    proposal_id=payload["proposal_id"],
                    candidate_id=payload["candidate_id"],
                    earlier_population_id=payload["earlier_population_id"],
                    later_population_id=payload["later_population_id"],
                    change_type=payload["change_type"],
                    effective_on=date.fromisoformat(payload["effective_on"]),
                    change_document_number=payload["change_document_number"],
                    change_provision_path=payload["change_provision_path"],
                    evidence=_evidence(payload["evidence"]),
                    rationale_vi=payload["rationale_vi"],
                    proposed_by_model=payload["proposed_by_model"],
                    prompt_version=payload["prompt_version"],
                    run_id=payload["run_id"],
                )
            )
    return tuple(proposals)


def write_change_progress(path: Path, progress: Sequence[ChangeProposalProgress]) -> None:
    """Atomically replace legal-change proposal checkpoints."""
    write_jsonl_atomic(path, (record.to_dict() for record in progress))


def read_change_progress(path: Path) -> tuple[ChangeProposalProgress, ...]:
    """Read legal-change proposal checkpoints."""
    progress = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != CHANGE_PROGRESS_SCHEMA_VERSION:
                msg = f"Unsupported legal-change progress schema: {payload['schema_version']}"
                raise ValueError(msg)
            progress.append(ChangeProposalProgress(candidate_id=payload["candidate_id"], run_id=payload["run_id"]))
    return tuple(progress)


def write_change_reviews(path: Path, reviews: Sequence[ChangeReview]) -> None:
    """Atomically replace human legal-change reviews."""
    write_jsonl_atomic(path, (review.to_dict() for review in reviews))


def read_change_reviews(path: Path) -> tuple[ChangeReview, ...]:
    """Read human legal-change reviews."""
    reviews = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != CHANGE_REVIEW_SCHEMA_VERSION:
                msg = f"Unsupported legal-change review schema: {payload['schema_version']}"
                raise ValueError(msg)
            if payload["decision"] not in REVIEW_DECISIONS:
                msg = f"Unsupported legal-change review decision: {payload['decision']}"
                raise ValueError(msg)
            reviews.append(
                ChangeReview(
                    proposal_id=payload["proposal_id"],
                    decision=payload["decision"],
                    reviewer=payload["reviewer"],
                    reviewed_on=date.fromisoformat(payload["reviewed_on"]) if payload["reviewed_on"] else None,
                    note=payload["note"],
                )
            )
    return tuple(reviews)


def pending_change_review_template(proposals: Sequence[LegalChangeProposal]) -> tuple[ChangeReview, ...]:
    """Return one pending human-review row for every proposal."""
    return tuple(
        ChangeReview(proposal_id=proposal.proposal_id, decision="pending", reviewer="", reviewed_on=None, note="")
        for proposal in proposals
    )


def write_legal_change_events(path: Path, events: Sequence[LegalChangeEvent]) -> None:
    """Atomically replace accepted canonical legal-change events."""
    write_jsonl_atomic(path, (event.to_dict() for event in events))


def read_legal_change_events(path: Path) -> tuple[LegalChangeEvent, ...]:
    """Read canonical accepted legal-change events."""
    events = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != LEGAL_CHANGE_EVENT_SCHEMA_VERSION:
                msg = f"Unsupported legal-change event schema: {payload['schema_version']}"
                raise ValueError(msg)
            if payload["change_type"] not in CHANGE_TYPES:
                msg = f"Unsupported legal-change type: {payload['change_type']}"
                raise ValueError(msg)
            events.append(
                LegalChangeEvent(
                    change_id=payload["change_id"],
                    change_type=payload["change_type"],
                    effective_on=date.fromisoformat(payload["effective_on"]),
                    earlier_population_id=payload["earlier_population_id"],
                    later_population_id=payload["later_population_id"],
                    change_document_number=payload["change_document_number"],
                    change_provision_path=payload["change_provision_path"],
                    evidence=_evidence(payload["evidence"]),
                    proposal_id=payload["proposal_id"],
                    reviewer=payload["reviewer"],
                    reviewed_on=date.fromisoformat(payload["reviewed_on"]),
                )
            )
    return tuple(events)


def _evidence(payload: Sequence[dict[str, str]]) -> tuple[PopulationEvidence, ...]:
    """Translate persisted exact evidence entries."""
    return tuple(PopulationEvidence(chunk_id=item["chunk_id"], exact_text=item["exact_text"]) for item in payload)
