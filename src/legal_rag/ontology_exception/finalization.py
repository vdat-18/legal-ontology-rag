"""Merge typed exception fragments into canonical records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from legal_rag.ontology_exception.models import (
    ExceptionCitationResolution,
    LegalExceptionProposal,
    LegalExceptionRecord,
)
from legal_rag.ontology_population.models import OntologyPopulationRecord, population_identity


def finalize_legal_exceptions(
    population_records: Sequence[OntologyPopulationRecord],
    proposals: Sequence[LegalExceptionProposal],
) -> tuple[LegalExceptionRecord, ...]:
    """Merge citation fragments by exception identity."""
    records_by_id = {record.population_id: record for record in population_records}
    proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}
    if len(records_by_id) != len(population_records):
        msg = "Canonical population identities must be unique before exception finalization"
        raise ValueError(msg)
    if len(proposals_by_id) != len(proposals):
        msg = "Exception proposal identifiers must be unique"
        raise ValueError(msg)

    _validate_proposals(proposals, records_by_id)
    grouped: dict[tuple[object, ...], list[LegalExceptionProposal]] = defaultdict(list)
    for proposal in proposals:
        grouped[_exception_key(proposal)].append(proposal)
    return tuple(
        _merge_exception_fragments(fragments)
        for _, fragments in sorted(grouped.items(), key=lambda item: repr(item[0]))
    )


def _validate_proposals(
    proposals: Sequence[LegalExceptionProposal],
    records_by_id: dict[str, OntologyPopulationRecord],
) -> None:
    """Require every general and redirect target to be canonical."""
    for proposal in proposals:
        if proposal.general_population_id not in records_by_id:
            msg = "Exceptions must reference a canonical general rule"
            raise ValueError(msg)
        target_ids = {target_id for citation in proposal.citations for target_id in citation.target_population_ids}
        if not target_ids <= set(records_by_id):
            msg = "Exception redirects must reference canonical target rule versions"
            raise ValueError(msg)


def _exception_key(proposal: LegalExceptionProposal) -> tuple[object, ...]:
    """Return the semantic identity shared by mergeable proposal fragments."""
    return (
        proposal.general_population_id,
        proposal.source_document_number,
        proposal.exception_provision_path,
        tuple((item.chunk_id, item.exact_text) for item in proposal.evidence),
    )


def _merge_exception_fragments(
    fragments: Sequence[LegalExceptionProposal],
) -> LegalExceptionRecord:
    """Merge citation fragments while preserving locator grouping."""
    first = fragments[0]
    targets_by_citation: dict[tuple[str, str], set[str]] = defaultdict(set)
    for proposal in fragments:
        for citation in proposal.citations:
            targets_by_citation[(citation.source_chunk_id, citation.citation_text)].update(
                citation.target_population_ids
            )
    citations = tuple(
        ExceptionCitationResolution(
            citation_text=citation_text,
            source_chunk_id=chunk_id,
            target_population_ids=tuple(sorted(targets)),
        )
        for (chunk_id, citation_text), targets in sorted(targets_by_citation.items())
    )
    semantic_content = {
        "general_population_id": first.general_population_id,
        "source_document_number": first.source_document_number,
        "exception_provision_path": first.exception_provision_path,
        "evidence": [item.to_dict() for item in first.evidence],
        "citations": [citation.to_dict() for citation in citations],
    }
    return LegalExceptionRecord(
        exception_id=population_identity(semantic_content),
        general_population_id=first.general_population_id,
        source_document_number=first.source_document_number,
        exception_provision_path=first.exception_provision_path,
        evidence=first.evidence,
        citations=citations,
    )
