"""Apply typed assembly proposals to canonical population records."""

from __future__ import annotations

from collections.abc import Sequence

from legal_rag.ontology_population.assembly_models import AssemblyProposal
from legal_rag.ontology_population.models import OntologyPopulationRecord


def finalize_assemblies(
    records: Sequence[OntologyPopulationRecord],
    proposals: Sequence[AssemblyProposal],
) -> tuple[OntologyPopulationRecord, ...]:
    """Replace proposed members and preserve all other records."""
    records_by_id = {record.population_id: record for record in records}
    proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}
    if len(proposals_by_id) != len(proposals):
        msg = "Assembly proposal identifiers must be unique"
        raise ValueError(msg)
    replaced_members: set[str] = set()
    for proposal in proposals:
        members = set(proposal.member_population_ids)
        if not members <= set(records_by_id):
            msg = "Assembly members must reference current population records"
            raise ValueError(msg)
        if replaced_members & members:
            msg = "Assembly proposals must not overlap population members"
            raise ValueError(msg)
        replaced_members.update(members)

    canonical = [record for record in records if record.population_id not in replaced_members]
    canonical.extend(proposal.assembled_record for proposal in sorted(proposals, key=lambda item: item.proposal_id))
    canonical_by_id = {record.population_id: record for record in canonical}
    if len(canonical_by_id) != len(canonical):
        msg = "Canonical population identities must remain unique after assembly"
        raise ValueError(msg)
    return tuple(canonical_by_id.values())
