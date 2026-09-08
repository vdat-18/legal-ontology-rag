"""Finalize typed concept identities and taxonomy relations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise

from legal_rag.ontology_concept.models import (
    CanonicalConceptRecord,
    CanonicalConceptRelationRecord,
    ConceptCandidateInventory,
    ConceptOccurrence,
    ConceptRelationProposal,
)
from legal_rag.ontology_population.models import population_identity


def validate_concept_proposal_candidates(
    inventories: Sequence[ConceptCandidateInventory],
    proposals: Sequence[ConceptRelationProposal],
) -> None:
    """Require persisted proposals to remain inside their source inventory."""
    occurrence_ids_by_candidate = {inventory.candidate_id: set(inventory.occurrence_ids) for inventory in inventories}
    for proposal in proposals:
        allowed_ids = occurrence_ids_by_candidate.get(proposal.candidate_id)
        if (
            allowed_ids is None
            or not {
                proposal.left_occurrence_id,
                proposal.right_occurrence_id,
            }
            <= allowed_ids
        ):
            msg = "Concept proposals must remain inside their source candidate inventory"
            raise ValueError(msg)


def finalize_concept_resolution(
    occurrences: Sequence[ConceptOccurrence],
    proposals: Sequence[ConceptRelationProposal],
) -> tuple[tuple[CanonicalConceptRecord, ...], tuple[CanonicalConceptRelationRecord, ...]]:
    """Create canonical identities and relations from typed proposals."""
    occurrences_by_id = {item.occurrence_id: item for item in occurrences}
    proposals_by_id = {item.proposal_id: item for item in proposals}
    if len(occurrences_by_id) != len(occurrences):
        msg = "Concept occurrence identities must be unique before finalization"
        raise ValueError(msg)
    if len(proposals_by_id) != len(proposals):
        msg = "Concept proposal identifiers must be unique"
        raise ValueError(msg)
    _validate_proposals(proposals, occurrences_by_id)
    roots = _same_concept_roots(occurrences_by_id, proposals)
    concepts, concept_id_by_occurrence = _canonical_concepts(occurrences_by_id, proposals, roots)
    relations = _canonical_relations(proposals, occurrences_by_id, concept_id_by_occurrence)
    return concepts, relations


def _validate_proposals(
    proposals: Sequence[ConceptRelationProposal],
    occurrences_by_id: Mapping[str, ConceptOccurrence],
) -> None:
    """Require relations to reference known same-kind occurrences once."""
    seen_pairs = set()
    for proposal in proposals:
        if (
            proposal.left_occurrence_id not in occurrences_by_id
            or proposal.right_occurrence_id not in occurrences_by_id
        ):
            msg = "Concept proposals must reference known occurrences"
            raise ValueError(msg)
        left = occurrences_by_id[proposal.left_occurrence_id]
        right = occurrences_by_id[proposal.right_occurrence_id]
        if left.kind != right.kind:
            msg = "Concept relations require occurrences of the same kind"
            raise ValueError(msg)
        pair = tuple(sorted((left.occurrence_id, right.occurrence_id)))
        if pair in seen_pairs:
            msg = "Only one semantic decision is allowed per concept occurrence pair"
            raise ValueError(msg)
        seen_pairs.add(pair)


def _same_concept_roots(
    occurrences_by_id: Mapping[str, ConceptOccurrence],
    proposals: Sequence[ConceptRelationProposal],
) -> dict[str, str]:
    """Return deterministic union roots for SameConcept decisions."""
    parent = {occurrence_id: occurrence_id for occurrence_id in occurrences_by_id}

    def find(occurrence_id: str) -> str:
        while parent[occurrence_id] != occurrence_id:
            parent[occurrence_id] = parent[parent[occurrence_id]]
            occurrence_id = parent[occurrence_id]
        return occurrence_id

    occurrences_by_label: dict[tuple[str, str], list[str]] = defaultdict(list)
    for occurrence in occurrences_by_id.values():
        occurrences_by_label[(occurrence.kind, occurrence.label_vi)].append(occurrence.occurrence_id)
    same_pairs = [pair for occurrence_ids in occurrences_by_label.values() for pair in pairwise(occurrence_ids)]
    same_pairs.extend(
        (proposal.left_occurrence_id, proposal.right_occurrence_id)
        for proposal in proposals
        if proposal.relation == "SameConcept"
    )
    for left_occurrence_id, right_occurrence_id in same_pairs:
        left_root = find(left_occurrence_id)
        right_root = find(right_occurrence_id)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)
    roots = {occurrence_id: find(occurrence_id) for occurrence_id in occurrences_by_id}
    for proposal in proposals:
        if (
            proposal.relation != "SameConcept"
            and roots[proposal.left_occurrence_id] == roots[proposal.right_occurrence_id]
        ):
            msg = "Concept relations conflict with a same-concept component"
            raise ValueError(msg)
    return roots


def _canonical_concepts(
    occurrences_by_id: Mapping[str, ConceptOccurrence],
    proposals: Sequence[ConceptRelationProposal],
    roots: Mapping[str, str],
) -> tuple[tuple[CanonicalConceptRecord, ...], dict[str, str]]:
    """Build one canonical record for every equivalence component or singleton."""
    members_by_root: dict[str, list[ConceptOccurrence]] = defaultdict(list)
    for occurrence_id, root in roots.items():
        members_by_root[root].append(occurrences_by_id[occurrence_id])
    same_by_root: dict[str, list[ConceptRelationProposal]] = defaultdict(list)
    for proposal in proposals:
        if proposal.relation == "SameConcept":
            same_by_root[roots[proposal.left_occurrence_id]].append(proposal)
    concepts = []
    concept_id_by_occurrence = {}
    for root, members in sorted(members_by_root.items()):
        fragments = same_by_root[root]
        labels = {item.label_vi for item in members}
        selected_labels = [
            proposal.preferred_label_vi
            for proposal in sorted(fragments, key=lambda item: item.proposal_id)
            if proposal.preferred_label_vi
        ]
        if any(label not in labels for label in selected_labels):
            msg = "Canonical preferred labels must come from source occurrences"
            raise ValueError(msg)
        if len(set(selected_labels)) > 1:
            msg = "One same-concept component must use one preferred label"
            raise ValueError(msg)
        preferred_label = selected_labels[0] if selected_labels else min(labels)
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
            )
        )
        concept_id_by_occurrence.update(dict.fromkeys(occurrence_ids, concept_id))
    return tuple(sorted(concepts, key=lambda item: item.concept_id)), concept_id_by_occurrence


def _canonical_relations(
    proposals: Sequence[ConceptRelationProposal],
    occurrences_by_id: Mapping[str, ConceptOccurrence],
    concept_id_by_occurrence: Mapping[str, str],
) -> tuple[CanonicalConceptRelationRecord, ...]:
    """Project non-equivalence decisions onto canonical identities."""
    grouped: set[tuple[str, str, str]] = set()
    for proposal in proposals:
        if proposal.relation in {"SameConcept", "Distinct"}:
            continue
        left = concept_id_by_occurrence[proposal.left_occurrence_id]
        right = concept_id_by_occurrence[proposal.right_occurrence_id]
        if left == right:
            continue
        source, target, relation = _canonical_relation_direction(proposal, left, right)
        grouped.add((source, target, relation))
    records = []
    for source, target, relation in sorted(grouped):
        semantic_content = {"source_concept_id": source, "target_concept_id": target, "relation": relation}
        records.append(
            CanonicalConceptRelationRecord(
                relation_id=population_identity(semantic_content),
                source_concept_id=source,
                target_concept_id=target,
                relation=relation,
            )
        )
    _validate_relation_graph(records)
    return tuple(records)


def _validate_relation_graph(relations: Sequence[CanonicalConceptRelationRecord]) -> None:
    """Reject contradictory pairs and cycles in the canonical broader graph."""
    relation_by_pair: dict[frozenset[str], str] = {}
    broader_targets: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        pair = frozenset((relation.source_concept_id, relation.target_concept_id))
        previous = relation_by_pair.setdefault(pair, relation.relation)
        if previous != relation.relation:
            msg = "Canonical concepts cannot have contradictory semantic relations"
            raise ValueError(msg)
        if relation.relation == "Broader":
            broader_targets[relation.source_concept_id].add(relation.target_concept_id)

    def visit(concept_id: str, path: set[str]) -> None:
        if concept_id in path:
            msg = "Canonical broader relations must be acyclic"
            raise ValueError(msg)
        next_path = {*path, concept_id}
        for target_id in broader_targets[concept_id]:
            visit(target_id, next_path)

    for concept_id in tuple(broader_targets):
        visit(concept_id, set())


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
