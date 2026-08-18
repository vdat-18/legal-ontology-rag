"""Apply human-reviewed assembly decisions to canonical population records."""

from __future__ import annotations

from collections.abc import Sequence

from legal_rag.ontology_population.assembly_config import AssemblySettings
from legal_rag.ontology_population.assembly_models import AssemblyProposal, AssemblyReview
from legal_rag.ontology_population.models import OntologyPopulationRecord


def finalize_assemblies(
    records: Sequence[OntologyPopulationRecord],
    proposals: Sequence[AssemblyProposal],
    reviews: Sequence[AssemblyReview],
    settings: AssemblySettings,
) -> tuple[OntologyPopulationRecord, ...]:
    """Replace accepted proposal members and preserve all other records."""
    records_by_id = {record.population_id: record for record in records}
    proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}
    reviews_by_id = {review.proposal_id: review for review in reviews}
    if len(proposals_by_id) != len(proposals) or len(reviews_by_id) != len(reviews):
        msg = "Assembly proposal and review identifiers must be unique"
        raise ValueError(msg)
    if not set(reviews_by_id) <= set(proposals_by_id):
        msg = "Assembly reviews must reference known proposals"
        raise ValueError(msg)
    if settings.require_every_proposal_reviewed and set(reviews_by_id) != set(proposals_by_id):
        msg = "Every assembly proposal requires a human review before finalization"
        raise ValueError(msg)
    accepted = _accepted_proposals(reviews, proposals_by_id)
    accepted_members: set[str] = set()
    for proposal in accepted:
        members = set(proposal.member_population_ids)
        if not members <= set(records_by_id):
            msg = "Accepted assembly members must reference current population records"
            raise ValueError(msg)
        if accepted_members & members:
            msg = "Accepted assembly proposals must not overlap population members"
            raise ValueError(msg)
        accepted_members.update(members)

    canonical = [record for record in records if record.population_id not in accepted_members]
    canonical.extend(proposal.assembled_record for proposal in sorted(accepted, key=lambda item: item.proposal_id))
    canonical_by_id = {record.population_id: record for record in canonical}
    if len(canonical_by_id) != len(canonical):
        msg = "Canonical population identities must remain unique after assembly"
        raise ValueError(msg)
    return tuple(canonical_by_id.values())


def _accepted_proposals(
    reviews: Sequence[AssemblyReview],
    proposals_by_id: dict[str, AssemblyProposal],
) -> tuple[AssemblyProposal, ...]:
    """Validate resolved reviews and return their accepted proposals."""
    if any(review.decision == "pending" for review in reviews):
        msg = "Pending assembly reviews must be resolved before finalization"
        raise ValueError(msg)
    if any(not review.reviewer or review.reviewed_on is None for review in reviews):
        msg = "Resolved assembly reviews require reviewer identity and review date"
        raise ValueError(msg)
    return tuple(proposals_by_id[review.proposal_id] for review in reviews if review.decision == "accepted")
