"""Apply human-reviewed legal changes to canonical rule versions."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from legal_rag.ontology_change.config import ChangeResolutionSettings
from legal_rag.ontology_change.models import ChangeReview, LegalChangeEvent, LegalChangeProposal
from legal_rag.ontology_population.models import (
    OntologyPopulationRecord,
    population_identity,
    with_population_validity,
)


def finalize_legal_changes(
    records: Sequence[OntologyPopulationRecord],
    proposals: Sequence[LegalChangeProposal],
    reviews: Sequence[ChangeReview],
    settings: ChangeResolutionSettings,
) -> tuple[tuple[OntologyPopulationRecord, ...], tuple[LegalChangeEvent, ...]]:
    """Apply accepted change dates and create canonical reviewed change events."""
    records_by_id = {record.population_id: record for record in records}
    proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}
    reviews_by_id = {review.proposal_id: review for review in reviews}
    if len(records_by_id) != len(records):
        msg = "Cross-document canonical population identities must be unique"
        raise ValueError(msg)
    if len(proposals_by_id) != len(proposals) or len(reviews_by_id) != len(reviews):
        msg = "Legal-change proposal and review identifiers must be unique"
        raise ValueError(msg)
    if not set(reviews_by_id) <= set(proposals_by_id):
        msg = "Legal-change reviews must reference known proposals"
        raise ValueError(msg)
    if settings.require_every_proposal_reviewed and set(reviews_by_id) != set(proposals_by_id):
        msg = "Every legal-change proposal requires a human review before finalization"
        raise ValueError(msg)

    accepted = _accepted_proposals(reviews, proposals_by_id)
    _validate_accepted_members(accepted, records_by_id)
    validity = {record.population_id: (record.valid_from, record.valid_to) for record in records}
    for proposal, _ in accepted:
        earlier_from, _ = validity[proposal.earlier_population_id]
        validity[proposal.earlier_population_id] = (earlier_from, proposal.effective_on)
        if proposal.later_population_id is not None:
            _, later_to = validity[proposal.later_population_id]
            validity[proposal.later_population_id] = (proposal.effective_on, later_to)

    canonical = tuple(_apply_validity(record, validity[record.population_id]) for record in records)
    canonical_ids = {
        original.population_id: updated.population_id for original, updated in zip(records, canonical, strict=True)
    }
    if len(set(canonical_ids.values())) != len(canonical_ids):
        msg = "Canonical population identities must remain unique after legal-change finalization"
        raise ValueError(msg)
    events = tuple(
        _legal_change_event(proposal, review, canonical_ids)
        for proposal, review in sorted(accepted, key=lambda item: item[0].proposal_id)
    )
    return canonical, events


def _apply_validity(
    record: OntologyPopulationRecord,
    validity: tuple[date | None, date | None],
) -> OntologyPopulationRecord:
    """Reidentify only rule versions whose reviewed validity changed."""
    valid_from, valid_to = validity
    if (record.valid_from, record.valid_to) == validity:
        return record
    return with_population_validity(record, valid_from=valid_from, valid_to=valid_to)


def _accepted_proposals(
    reviews: Sequence[ChangeReview],
    proposals_by_id: dict[str, LegalChangeProposal],
) -> tuple[tuple[LegalChangeProposal, ChangeReview], ...]:
    """Validate resolved reviews and return accepted proposal-review pairs."""
    if any(review.decision == "pending" for review in reviews):
        msg = "Pending legal-change reviews must be resolved before finalization"
        raise ValueError(msg)
    if any(not review.reviewer or review.reviewed_on is None for review in reviews):
        msg = "Resolved legal-change reviews require reviewer identity and review date"
        raise ValueError(msg)
    return tuple((proposals_by_id[review.proposal_id], review) for review in reviews if review.decision == "accepted")


def _validate_accepted_members(
    accepted: Sequence[tuple[LegalChangeProposal, ChangeReview]],
    records_by_id: dict[str, OntologyPopulationRecord],
) -> None:
    """Reject unknown or multiply changed canonical rule versions."""
    earlier_ids = [proposal.earlier_population_id for proposal, _ in accepted]
    later_ids = [proposal.later_population_id for proposal, _ in accepted if proposal.later_population_id is not None]
    referenced = set(earlier_ids) | set(later_ids)
    if not referenced <= set(records_by_id):
        msg = "Accepted legal changes must reference current canonical population records"
        raise ValueError(msg)
    if len(earlier_ids) != len(set(earlier_ids)):
        msg = "Accepted legal changes must not change one earlier rule more than once"
        raise ValueError(msg)
    if len(later_ids) != len(set(later_ids)):
        msg = "Accepted legal changes must not produce one later rule more than once"
        raise ValueError(msg)
    if any(
        proposal.later_population_id is not None
        and records_by_id[proposal.earlier_population_id].document_number
        == records_by_id[proposal.later_population_id].document_number
        for proposal, _ in accepted
    ):
        msg = "Accepted cross-document legal changes must link rule versions from different documents"
        raise ValueError(msg)


def _legal_change_event(
    proposal: LegalChangeProposal,
    review: ChangeReview,
    canonical_ids: dict[str, str],
) -> LegalChangeEvent:
    """Create one stable event linked to reidentified canonical rule versions."""
    earlier_id = canonical_ids[proposal.earlier_population_id]
    later_id = canonical_ids[proposal.later_population_id] if proposal.later_population_id is not None else None
    content = {
        "change_type": proposal.change_type,
        "effective_on": proposal.effective_on.isoformat(),
        "earlier_population_id": earlier_id,
        "later_population_id": later_id,
        "change_document_number": proposal.change_document_number,
        "change_provision_path": proposal.change_provision_path,
        "evidence": [item.to_dict() for item in proposal.evidence],
        "proposal_id": proposal.proposal_id,
        "reviewer": review.reviewer,
        "reviewed_on": review.reviewed_on.isoformat(),
    }
    return LegalChangeEvent(
        change_id=population_identity(content),
        change_type=proposal.change_type,
        effective_on=proposal.effective_on,
        earlier_population_id=earlier_id,
        later_population_id=later_id,
        change_document_number=proposal.change_document_number,
        change_provision_path=proposal.change_provision_path,
        evidence=proposal.evidence,
        proposal_id=proposal.proposal_id,
        reviewer=review.reviewer,
        reviewed_on=review.reviewed_on,
    )
