"""Persistence for exception candidates, proposals, reviews, and records."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from legal_rag.io_utils import write_json_atomic, write_jsonl_atomic
from legal_rag.ontology_exception.models import (
    EXCEPTION_CANDIDATE_SCHEMA_VERSION,
    EXCEPTION_PROGRESS_SCHEMA_VERSION,
    EXCEPTION_PROPOSAL_SCHEMA_VERSION,
    EXCEPTION_REVIEW_SCHEMA_VERSION,
    LEGAL_EXCEPTION_RECORD_SCHEMA_VERSION,
    ExceptionCandidateSet,
    ExceptionCandidateWindow,
    ExceptionCitationResolution,
    ExceptionProposalProgress,
    ExceptionReview,
    LegalExceptionProposal,
    LegalExceptionRecord,
)
from legal_rag.ontology_population.models import PopulationEvidence

REVIEW_DECISIONS = {"pending", "accepted", "rejected"}


def write_exception_candidates(path: Path, candidates: ExceptionCandidateSet) -> None:
    """Atomically replace one exception candidate-set artifact."""
    write_json_atomic(path, candidates.to_dict())


def read_exception_candidates(path: Path) -> ExceptionCandidateSet:
    """Read one fingerprinted exception candidate set."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != EXCEPTION_CANDIDATE_SCHEMA_VERSION:
        msg = f"Unsupported exception candidate schema: {payload['schema_version']}"
        raise ValueError(msg)
    return ExceptionCandidateSet(
        run_id=payload["run_id"],
        windows=tuple(
            ExceptionCandidateWindow(
                candidate_id=item["candidate_id"],
                source_document_id=item["source_document_id"],
                general_population_ids=tuple(item["general_population_ids"]),
                target_population_ids=tuple(item["target_population_ids"]),
                source_chunk_ids=tuple(item["source_chunk_ids"]),
            )
            for item in payload["windows"]
        ),
    )


def write_exception_proposals(path: Path, proposals: Sequence[LegalExceptionProposal]) -> None:
    """Atomically replace reviewable exception proposals."""
    write_jsonl_atomic(path, (proposal.to_dict() for proposal in proposals))


def read_exception_proposals(path: Path) -> tuple[LegalExceptionProposal, ...]:
    """Read typed exception proposals."""
    proposals = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != EXCEPTION_PROPOSAL_SCHEMA_VERSION:
                msg = f"Unsupported exception proposal schema: {payload['schema_version']}"
                raise ValueError(msg)
            proposals.append(
                LegalExceptionProposal(
                    proposal_id=payload["proposal_id"],
                    candidate_id=payload["candidate_id"],
                    general_population_id=payload["general_population_id"],
                    source_document_number=payload["source_document_number"],
                    exception_provision_path=payload["exception_provision_path"],
                    evidence=_evidence(payload["evidence"]),
                    citations=_citations(payload["citations"]),
                    rationale_vi=payload["rationale_vi"],
                    proposed_by_model=payload["proposed_by_model"],
                    prompt_version=payload["prompt_version"],
                    run_id=payload["run_id"],
                )
            )
    return tuple(proposals)


def write_exception_progress(path: Path, progress: Sequence[ExceptionProposalProgress]) -> None:
    """Atomically replace exception proposal checkpoints."""
    write_jsonl_atomic(path, (record.to_dict() for record in progress))


def read_exception_progress(path: Path) -> tuple[ExceptionProposalProgress, ...]:
    """Read exception proposal checkpoints."""
    progress = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != EXCEPTION_PROGRESS_SCHEMA_VERSION:
                msg = f"Unsupported exception progress schema: {payload['schema_version']}"
                raise ValueError(msg)
            progress.append(ExceptionProposalProgress(candidate_id=payload["candidate_id"], run_id=payload["run_id"]))
    return tuple(progress)


def write_exception_reviews(path: Path, reviews: Sequence[ExceptionReview]) -> None:
    """Atomically replace human exception reviews."""
    write_jsonl_atomic(path, (review.to_dict() for review in reviews))


def read_exception_reviews(path: Path) -> tuple[ExceptionReview, ...]:
    """Read human exception reviews."""
    reviews = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != EXCEPTION_REVIEW_SCHEMA_VERSION:
                msg = f"Unsupported exception review schema: {payload['schema_version']}"
                raise ValueError(msg)
            if payload["decision"] not in REVIEW_DECISIONS:
                msg = f"Unsupported exception review decision: {payload['decision']}"
                raise ValueError(msg)
            reviews.append(
                ExceptionReview(
                    proposal_id=payload["proposal_id"],
                    decision=payload["decision"],
                    reviewer=payload["reviewer"],
                    reviewed_on=date.fromisoformat(payload["reviewed_on"]) if payload["reviewed_on"] else None,
                    note=payload["note"],
                )
            )
    return tuple(reviews)


def pending_exception_review_template(proposals: Sequence[LegalExceptionProposal]) -> tuple[ExceptionReview, ...]:
    """Return one pending human-review row for every proposal."""
    return tuple(
        ExceptionReview(proposal_id=proposal.proposal_id, decision="pending", reviewer="", reviewed_on=None, note="")
        for proposal in proposals
    )


def write_legal_exception_records(path: Path, records: Sequence[LegalExceptionRecord]) -> None:
    """Atomically replace reviewed canonical exception records."""
    write_jsonl_atomic(path, (record.to_dict() for record in records))


def read_legal_exception_records(path: Path) -> tuple[LegalExceptionRecord, ...]:
    """Read reviewed canonical exception records."""
    records = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != LEGAL_EXCEPTION_RECORD_SCHEMA_VERSION:
                msg = f"Unsupported legal-exception record schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(
                LegalExceptionRecord(
                    exception_id=payload["exception_id"],
                    general_population_id=payload["general_population_id"],
                    source_document_number=payload["source_document_number"],
                    exception_provision_path=payload["exception_provision_path"],
                    evidence=_evidence(payload["evidence"]),
                    citations=_citations(payload["citations"]),
                    accepted_proposal_ids=tuple(payload["accepted_proposal_ids"]),
                    reviewers=tuple(payload["reviewers"]),
                    reviewed_on=tuple(date.fromisoformat(value) for value in payload["reviewed_on"]),
                )
            )
    return tuple(records)


def _evidence(payload: Sequence[dict[str, str]]) -> tuple[PopulationEvidence, ...]:
    """Translate persisted exact evidence entries."""
    return tuple(PopulationEvidence(chunk_id=item["chunk_id"], exact_text=item["exact_text"]) for item in payload)


def _citations(payload: Sequence[dict[str, object]]) -> tuple[ExceptionCitationResolution, ...]:
    """Translate persisted grouped citation entries."""
    return tuple(
        ExceptionCitationResolution(
            citation_text=str(item["citation_text"]),
            source_chunk_id=str(item["source_chunk_id"]),
            target_population_ids=tuple(item["target_population_ids"]),
        )
        for item in payload
    )
