"""Apply typed legal-change proposals to canonical rule versions."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from legal_rag.ontology_change.models import LegalChangeEvent, LegalChangeProposal
from legal_rag.ontology_population.models import (
    OntologyPopulationRecord,
    population_identity,
    with_population_validity,
)


def finalize_legal_changes(
    records: Sequence[OntologyPopulationRecord],
    proposals: Sequence[LegalChangeProposal],
) -> tuple[tuple[OntologyPopulationRecord, ...], tuple[LegalChangeEvent, ...]]:
    """Apply proposed change dates and create canonical change events."""
    records_by_id = {record.population_id: record for record in records}
    proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}
    if len(records_by_id) != len(records):
        msg = "Cross-document canonical population identities must be unique"
        raise ValueError(msg)
    if len(proposals_by_id) != len(proposals):
        msg = "Legal-change proposal identifiers must be unique"
        raise ValueError(msg)

    _validate_proposal_members(proposals, records_by_id)
    validity = {record.population_id: (record.valid_from, record.valid_to) for record in records}
    for proposal in proposals:
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
        _legal_change_event(proposal, canonical_ids)
        for proposal in sorted(proposals, key=lambda item: item.proposal_id)
    )
    return canonical, events


def _apply_validity(
    record: OntologyPopulationRecord,
    validity: tuple[date | None, date | None],
) -> OntologyPopulationRecord:
    """Reidentify only rule versions whose validity changed."""
    valid_from, valid_to = validity
    if (record.valid_from, record.valid_to) == validity:
        return record
    return with_population_validity(record, valid_from=valid_from, valid_to=valid_to)


def _validate_proposal_members(
    proposals: Sequence[LegalChangeProposal],
    records_by_id: dict[str, OntologyPopulationRecord],
) -> None:
    """Reject unknown or multiply changed canonical rule versions."""
    earlier_ids = [proposal.earlier_population_id for proposal in proposals]
    later_ids = [proposal.later_population_id for proposal in proposals if proposal.later_population_id is not None]
    referenced = set(earlier_ids) | set(later_ids)
    if not referenced <= set(records_by_id):
        msg = "Legal changes must reference current canonical population records"
        raise ValueError(msg)
    if len(earlier_ids) != len(set(earlier_ids)):
        msg = "Legal changes must not change one earlier rule more than once"
        raise ValueError(msg)
    if len(later_ids) != len(set(later_ids)):
        msg = "Legal changes must not produce one later rule more than once"
        raise ValueError(msg)


def _legal_change_event(
    proposal: LegalChangeProposal,
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
    )
