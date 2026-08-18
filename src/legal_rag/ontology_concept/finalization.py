"""Finalize human-reviewed concept identities and taxonomy relations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from legal_rag.ontology_concept.config import ConceptResolutionSettings
from legal_rag.ontology_concept.models import (
    CanonicalConceptRecord,
    CanonicalConceptRelationRecord,
    ConceptOccurrence,
    ConceptRelationProposal,
    ConceptReview,
)
from legal_rag.ontology_population.models import population_identity


def finalize_concept_resolution(
    occurrences: Sequence[ConceptOccurrence],
    proposals: Sequence[ConceptRelationProposal],
    reviews: Sequence[ConceptReview],
    settings: ConceptResolutionSettings,
) -> tuple[tuple[CanonicalConceptRecord, ...], tuple[CanonicalConceptRelationRecord, ...]]:
    """Create canonical identities and relations from attributable accepted reviews."""
    occurrences_by_id = {item.occurrence_id: item for item in occurrences}
    proposals_by_id = {item.proposal_id: item for item in proposals}
    reviews_by_id = {item.proposal_id: item for item in reviews}
    if len(occurrences_by_id) != len(occurrences):
        msg = "Concept occurrence identities must be unique before finalization"
        raise ValueError(msg)
    if len(proposals_by_id) != len(proposals) or len(reviews_by_id) != len(reviews):
        msg = "Concept proposal and review identifiers must be unique"
        raise ValueError(msg)
    if not set(reviews_by_id) <= set(proposals_by_id):
        msg = "Concept reviews must reference known proposals"
        raise ValueError(msg)
    if settings.require_every_proposal_reviewed and set(reviews_by_id) != set(proposals_by_id):
        msg = "Every concept proposal requires a human review before finalization"
        raise ValueError(msg)
    accepted = _accepted_proposals(reviews, proposals_by_id)
    _validate_accepted_proposals(accepted, occurrences_by_id)
    roots = _same_concept_roots(occurrences_by_id, accepted)
    concepts, concept_id_by_occurrence = _canonical_concepts(occurrences_by_id, accepted, roots)
    relations = _canonical_relations(accepted, occurrences_by_id, concept_id_by_occurrence)
    _validate_broader_cycles(relations)
    return concepts, relations


def _accepted_proposals(
    reviews: Sequence[ConceptReview],
    proposals_by_id: Mapping[str, ConceptRelationProposal],
) -> tuple[tuple[ConceptRelationProposal, ConceptReview], ...]:
    """Validate resolved reviews and return accepted proposal-review pairs."""
    if any(item.decision == "pending" for item in reviews):
        msg = "Pending concept reviews must be resolved before finalization"
        raise ValueError(msg)
    if any(not item.reviewer or item.reviewed_on is None for item in reviews):
        msg = "Resolved concept reviews require reviewer identity and review date"
        raise ValueError(msg)
    return tuple((proposals_by_id[item.proposal_id], item) for item in reviews if item.decision == "accepted")


def _validate_accepted_proposals(
    accepted: Sequence[tuple[ConceptRelationProposal, ConceptReview]],
    occurrences_by_id: Mapping[str, ConceptOccurrence],
) -> None:
    """Require accepted relations to reference known same-kind occurrences."""
    seen_pairs = set()
    for proposal, _ in accepted:
        if (
            proposal.left_occurrence_id not in occurrences_by_id
            or proposal.right_occurrence_id not in occurrences_by_id
        ):
            msg = "Accepted concept proposals must reference known occurrences"
            raise ValueError(msg)
        left = occurrences_by_id[proposal.left_occurrence_id]
        right = occurrences_by_id[proposal.right_occurrence_id]
        if left.kind != right.kind:
            msg = "Accepted concept relations require occurrences of the same kind"
            raise ValueError(msg)
        if not proposal.left_evidence or not set(proposal.left_evidence) <= set(left.evidence):
            msg = "Accepted concept left evidence must belong to its occurrence"
            raise ValueError(msg)
        if not proposal.right_evidence or not set(proposal.right_evidence) <= set(right.evidence):
            msg = "Accepted concept right evidence must belong to its occurrence"
            raise ValueError(msg)
        if proposal.relation == "SameConcept":
            if proposal.preferred_label_vi not in {left.label_vi, right.label_vi}:
                msg = "Accepted SameConcept preferred label must belong to its occurrence pair"
                raise ValueError(msg)
        elif proposal.preferred_label_vi:
            msg = "Accepted non-equivalence relations must not select a preferred label"
            raise ValueError(msg)
        pair = tuple(sorted((left.occurrence_id, right.occurrence_id)))
        if pair in seen_pairs:
            msg = "Only one accepted semantic decision is allowed per concept occurrence pair"
            raise ValueError(msg)
        seen_pairs.add(pair)


def _same_concept_roots(
    occurrences_by_id: Mapping[str, ConceptOccurrence],
    accepted: Sequence[tuple[ConceptRelationProposal, ConceptReview]],
) -> dict[str, str]:
    """Return deterministic union roots for accepted SameConcept decisions."""
    parent = {occurrence_id: occurrence_id for occurrence_id in occurrences_by_id}

    def find(occurrence_id: str) -> str:
        while parent[occurrence_id] != occurrence_id:
            parent[occurrence_id] = parent[parent[occurrence_id]]
            occurrence_id = parent[occurrence_id]
        return occurrence_id

    for proposal, _ in accepted:
        if proposal.relation != "SameConcept":
            continue
        left_root = find(proposal.left_occurrence_id)
        right_root = find(proposal.right_occurrence_id)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)
    return {occurrence_id: find(occurrence_id) for occurrence_id in occurrences_by_id}


def _canonical_concepts(
    occurrences_by_id: Mapping[str, ConceptOccurrence],
    accepted: Sequence[tuple[ConceptRelationProposal, ConceptReview]],
    roots: Mapping[str, str],
) -> tuple[tuple[CanonicalConceptRecord, ...], dict[str, str]]:
    """Build one canonical record for every accepted-equivalence component or singleton."""
    members_by_root: dict[str, list[ConceptOccurrence]] = defaultdict(list)
    for occurrence_id, root in roots.items():
        members_by_root[root].append(occurrences_by_id[occurrence_id])
    same_by_root: dict[str, list[tuple[ConceptRelationProposal, ConceptReview]]] = defaultdict(list)
    for proposal, review in accepted:
        if proposal.relation == "SameConcept":
            same_by_root[roots[proposal.left_occurrence_id]].append((proposal, review))
    concepts = []
    concept_id_by_occurrence = {}
    for root, members in sorted(members_by_root.items()):
        fragments = same_by_root[root]
        selected_labels = {proposal.preferred_label_vi for proposal, _ in fragments}
        if len(selected_labels) > 1:
            msg = "Accepted SameConcept component must select one consistent preferred label"
            raise ValueError(msg)
        labels = {item.label_vi for item in members}
        preferred_label = next(iter(selected_labels)) if selected_labels else min(labels)
        if preferred_label not in labels:
            msg = "Canonical preferred label must belong to its concept component"
            raise ValueError(msg)
        occurrence_ids = tuple(sorted(item.occurrence_id for item in members))
        kind = members[0].kind
        concept_id = population_identity(
            {"kind": kind, "preferred_label_vi": preferred_label, "occurrence_ids": occurrence_ids}
        )
        concepts.append(
            CanonicalConceptRecord(
                concept_id=concept_id,
                kind=kind,
                preferred_label_vi=preferred_label,
                alternative_labels_vi=tuple(sorted(labels - {preferred_label})),
                occurrence_ids=occurrence_ids,
                accepted_proposal_ids=tuple(sorted(proposal.proposal_id for proposal, _ in fragments)),
                reviewers=tuple(sorted({review.reviewer for _, review in fragments})),
                reviewed_on=tuple(
                    sorted({review.reviewed_on for _, review in fragments if review.reviewed_on is not None})
                ),
            )
        )
        concept_id_by_occurrence.update(dict.fromkeys(occurrence_ids, concept_id))
    return tuple(sorted(concepts, key=lambda item: item.concept_id)), concept_id_by_occurrence


def _canonical_relations(
    accepted: Sequence[tuple[ConceptRelationProposal, ConceptReview]],
    occurrences_by_id: Mapping[str, ConceptOccurrence],
    concept_id_by_occurrence: Mapping[str, str],
) -> tuple[CanonicalConceptRelationRecord, ...]:
    """Project accepted non-equivalence decisions onto canonical identities."""
    grouped: dict[tuple[str, str, str], list[tuple[ConceptRelationProposal, ConceptReview]]] = defaultdict(list)
    decisions_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    for proposal, review in accepted:
        if proposal.relation == "SameConcept":
            continue
        left = concept_id_by_occurrence[proposal.left_occurrence_id]
        right = concept_id_by_occurrence[proposal.right_occurrence_id]
        if left == right:
            msg = "A canonical concept cannot also have a non-equivalence relation to itself"
            raise ValueError(msg)
        source, target, relation = _canonical_relation_direction(proposal, left, right)
        pair = tuple(sorted((source, target)))
        decisions_by_pair[pair].add(relation)
        grouped[(source, target, relation)].append((proposal, review))
    if any(len(relations) > 1 for relations in decisions_by_pair.values()):
        msg = "Accepted canonical concept relations must not conflict"
        raise ValueError(msg)
    records = []
    for (source, target, relation), fragments in sorted(grouped.items()):
        semantic_content = {"source_concept_id": source, "target_concept_id": target, "relation": relation}
        records.append(
            CanonicalConceptRelationRecord(
                relation_id=population_identity(semantic_content),
                source_concept_id=source,
                target_concept_id=target,
                relation=relation,
                accepted_proposal_ids=tuple(sorted(item.proposal_id for item, _ in fragments)),
                reviewers=tuple(sorted({review.reviewer for _, review in fragments})),
                reviewed_on=tuple(
                    sorted({review.reviewed_on for _, review in fragments if review.reviewed_on is not None})
                ),
            )
        )
    return tuple(records)


def _canonical_relation_direction(
    proposal: ConceptRelationProposal,
    left_concept_id: str,
    right_concept_id: str,
) -> tuple[str, str, str]:
    """Normalize proposal direction to narrower-source/broader-target or ordered symmetric IDs."""
    if proposal.relation == "LeftBroaderThanRight":
        return right_concept_id, left_concept_id, "Broader"
    if proposal.relation == "LeftNarrowerThanRight":
        return left_concept_id, right_concept_id, "Broader"
    source, target = sorted((left_concept_id, right_concept_id))
    return source, target, proposal.relation


def _validate_broader_cycles(relations: Sequence[CanonicalConceptRelationRecord]) -> None:
    """Reject reviewed SKOS broader cycles before RDF materialization."""
    broader_by_source: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        if relation.relation == "Broader":
            broader_by_source[relation.source_concept_id].add(relation.target_concept_id)

    def visit(concept_id: str, path: set[str]) -> None:
        if concept_id in path:
            msg = "Accepted canonical concept hierarchy must not contain a broader cycle"
            raise ValueError(msg)
        next_path = path | {concept_id}
        for broader_id in broader_by_source[concept_id]:
            visit(broader_id, next_path)

    for concept_id in tuple(broader_by_source):
        visit(concept_id, set())
