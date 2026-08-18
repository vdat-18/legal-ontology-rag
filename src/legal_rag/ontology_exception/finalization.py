"""Merge human-accepted exception fragments into canonical records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from legal_rag.ontology_exception.config import ExceptionResolutionSettings
from legal_rag.ontology_exception.models import (
    ExceptionCitationResolution,
    ExceptionReview,
    LegalExceptionProposal,
    LegalExceptionRecord,
)
from legal_rag.ontology_population.models import OntologyPopulationRecord, population_identity


def finalize_legal_exceptions(
    population_records: Sequence[OntologyPopulationRecord],
    proposals: Sequence[LegalExceptionProposal],
    reviews: Sequence[ExceptionReview],
    settings: ExceptionResolutionSettings,
) -> tuple[LegalExceptionRecord, ...]:
    """Validate reviews and merge accepted citation fragments by exception identity."""
    records_by_id = {record.population_id: record for record in population_records}
    proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}
    reviews_by_id = {review.proposal_id: review for review in reviews}
    if len(records_by_id) != len(population_records):
        msg = "Canonical population identities must be unique before exception finalization"
        raise ValueError(msg)
    if len(proposals_by_id) != len(proposals) or len(reviews_by_id) != len(reviews):
        msg = "Exception proposal and review identifiers must be unique"
        raise ValueError(msg)
    if not set(reviews_by_id) <= set(proposals_by_id):
        msg = "Exception reviews must reference known proposals"
        raise ValueError(msg)
    if settings.require_every_proposal_reviewed and set(reviews_by_id) != set(proposals_by_id):
        msg = "Every exception proposal requires a human review before finalization"
        raise ValueError(msg)

    accepted = _accepted_proposals(reviews, proposals_by_id)
    _validate_accepted_proposals(accepted, records_by_id)
    grouped: dict[tuple[object, ...], list[tuple[LegalExceptionProposal, ExceptionReview]]] = defaultdict(list)
    for proposal, review in accepted:
        grouped[_exception_key(proposal)].append((proposal, review))
    return tuple(
        _merge_exception_fragments(fragments)
        for _, fragments in sorted(grouped.items(), key=lambda item: repr(item[0]))
    )


def _accepted_proposals(
    reviews: Sequence[ExceptionReview],
    proposals_by_id: dict[str, LegalExceptionProposal],
) -> tuple[tuple[LegalExceptionProposal, ExceptionReview], ...]:
    """Validate resolved reviews and return accepted proposal-review pairs."""
    if any(review.decision == "pending" for review in reviews):
        msg = "Pending exception reviews must be resolved before finalization"
        raise ValueError(msg)
    if any(not review.reviewer or review.reviewed_on is None for review in reviews):
        msg = "Resolved exception reviews require reviewer identity and review date"
        raise ValueError(msg)
    return tuple((proposals_by_id[review.proposal_id], review) for review in reviews if review.decision == "accepted")


def _validate_accepted_proposals(
    accepted: Sequence[tuple[LegalExceptionProposal, ExceptionReview]],
    records_by_id: dict[str, OntologyPopulationRecord],
) -> None:
    """Require every accepted general and redirect target to be canonical."""
    for proposal, _ in accepted:
        if proposal.general_population_id not in records_by_id:
            msg = "Accepted exceptions must reference a canonical general rule"
            raise ValueError(msg)
        if records_by_id[proposal.general_population_id].document_number != proposal.source_document_number:
            msg = "Accepted exception source document must match its general rule version"
            raise ValueError(msg)
        target_ids = {target_id for citation in proposal.citations for target_id in citation.target_population_ids}
        if not target_ids <= set(records_by_id):
            msg = "Accepted exception redirects must reference canonical target rule versions"
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
    fragments: Sequence[tuple[LegalExceptionProposal, ExceptionReview]],
) -> LegalExceptionRecord:
    """Merge reviewed citation fragments while preserving locator grouping."""
    first = fragments[0][0]
    targets_by_citation: dict[tuple[str, str], set[str]] = defaultdict(set)
    for proposal, _ in fragments:
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
        accepted_proposal_ids=tuple(sorted(proposal.proposal_id for proposal, _ in fragments)),
        reviewers=tuple(sorted({review.reviewer for _, review in fragments})),
        reviewed_on=tuple(sorted({review.reviewed_on for _, review in fragments if review.reviewed_on is not None})),
    )
