"""Tests for concept finalization and RDF canonicalization."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from rdflib import SKOS

from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_concept.candidates import build_concept_candidates
from legal_rag.ontology_concept.config import ConceptResolutionSettings
from legal_rag.ontology_concept.finalization import (
    finalize_concept_resolution,
    validate_concept_proposal_candidates,
)
from legal_rag.ontology_concept.materializer import canonical_concept_iri, materialize_concept_resolution_graph
from legal_rag.ontology_concept.models import ConceptCandidateSet, ConceptRelationProposal
from legal_rag.ontology_concept.writer import (
    read_canonical_concepts,
    read_canonical_relations,
    write_canonical_concepts,
    write_canonical_relations,
)
from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.materializer import LORNR, LORSP, validate_population_graph
from legal_rag.ontology_validation.paths import OntologyPaths

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _role_proposal(exception_corpus: ChangeCorpus) -> tuple[ConceptCandidateSet, ConceptRelationProposal]:
    """Return candidates and one source-grounded role equivalence proposal."""
    settings = ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml")
    candidates = build_concept_candidates(exception_corpus.documents[0].records, settings)
    roles = [item for item in candidates.occurrences if item.kind == "LegalSubjectRole"]
    left, right = roles
    inventory = next(item for item in candidates.inventories if item.kind == "LegalSubjectRole")
    return candidates, ConceptRelationProposal(
        proposal_id="proposal-role",
        candidate_id=inventory.candidate_id,
        left_occurrence_id=left.occurrence_id,
        right_occurrence_id=right.occurrence_id,
        relation="SameConcept",
        preferred_label_vi=left.label_vi,
        left_evidence=left.evidence,
        right_evidence=right.evidence,
        rationale_vi="Cùng vai trò pháp lý.",
        run_id="run-1",
    )


def test_validated_equivalence_remaps_rules_and_passes_shacl(
    tmp_path: Path,
    exception_corpus: ChangeCorpus,
) -> None:
    """A valid equality creates one canonical role used by both applicability profiles."""
    candidates, proposal = _role_proposal(exception_corpus)
    concepts, relations = finalize_concept_resolution(candidates.occurrences, (proposal,))
    population_settings = PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml")
    graph = materialize_concept_resolution_graph(
        exception_corpus,
        exception_corpus.documents[0].records,
        (),
        (),
        candidates.occurrences,
        concepts,
        relations,
        population_settings.instance_namespace,
    )
    concepts_path = tmp_path / "concepts.jsonl"
    relations_path = tmp_path / "relations.jsonl"
    write_canonical_concepts(concepts_path, concepts)
    write_canonical_relations(relations_path, relations)

    role = next(item for item in concepts if item.kind == "LegalSubjectRole")
    role_iri = canonical_concept_iri(population_settings.instance_namespace, role.concept_id)
    assert len(role.occurrence_ids) == 2
    assert len(set(graph.subjects(LORNR.requiresRole, role_iri))) == 2
    assert (role_iri, SKOS.prefLabel, None) in graph
    assert tuple(graph.objects(role_iri, LORSP.sourcedFromChunk))
    assert relations == ()
    assert validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))
    assert read_canonical_concepts(concepts_path) == concepts
    assert read_canonical_relations(relations_path) == relations


def test_no_proposals_merge_only_identical_controlled_labels(exception_corpus: ChangeCorpus) -> None:
    """Exact controlled labels share identity without inventing cross-label relations."""
    candidates, _ = _role_proposal(exception_corpus)
    concepts, relations = finalize_concept_resolution(candidates.occurrences, ())

    assert len(concepts) == 2
    assert relations == ()


def test_broader_relation_materializes_as_skos_without_merging(exception_corpus: ChangeCorpus) -> None:
    """A valid hierarchy keeps distinct identities and emits inverse SKOS projections."""
    settings = ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml")
    candidates = build_concept_candidates(exception_corpus.documents[0].records, settings)
    vehicles = [item for item in candidates.occurrences if item.kind == "VehicleCategory"]
    left = replace(vehicles[0], label_vi="Phương tiện giao thông")
    right = replace(vehicles[1], label_vi="Xe mô tô")
    occurrences = (*[item for item in candidates.occurrences if item.kind != "VehicleCategory"], left, right)
    inventory = next(item for item in candidates.inventories if item.kind == "VehicleCategory")
    proposal = ConceptRelationProposal(
        proposal_id="proposal-hierarchy",
        candidate_id=inventory.candidate_id,
        left_occurrence_id=left.occurrence_id,
        right_occurrence_id=right.occurrence_id,
        relation="LeftBroaderThanRight",
        preferred_label_vi="",
        left_evidence=left.evidence,
        right_evidence=right.evidence,
        rationale_vi="Khái niệm bên trái rộng hơn theo nguồn.",
        run_id="run-1",
    )
    concepts, relations = finalize_concept_resolution(occurrences, (proposal,))
    population_settings = PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml")
    graph = materialize_concept_resolution_graph(
        exception_corpus,
        exception_corpus.documents[0].records,
        (),
        (),
        occurrences,
        concepts,
        relations,
        population_settings.instance_namespace,
    )
    concept_by_occurrence = {occurrence_id: concept for concept in concepts for occurrence_id in concept.occurrence_ids}
    narrower = canonical_concept_iri(
        population_settings.instance_namespace,
        concept_by_occurrence[right.occurrence_id].concept_id,
    )
    broader = canonical_concept_iri(
        population_settings.instance_namespace,
        concept_by_occurrence[left.occurrence_id].concept_id,
    )

    assert relations[0].relation == "Broader"
    assert (narrower, SKOS.broader, broader) in graph
    assert (broader, SKOS.narrower, narrower) in graph
    assert validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))


def test_duplicate_proposal_ids_are_rejected(exception_corpus: ChangeCorpus) -> None:
    """Proposal identifiers remain unique for deterministic persistence."""
    candidates, proposal = _role_proposal(exception_corpus)

    with pytest.raises(ValueError, match="identifiers must be unique"):
        finalize_concept_resolution(candidates.occurrences, (proposal, proposal))


def test_unknown_occurrences_are_rejected(exception_corpus: ChangeCorpus) -> None:
    """A relation cannot reference a concept occurrence outside current candidates."""
    candidates, proposal = _role_proposal(exception_corpus)

    with pytest.raises(ValueError, match="known occurrences"):
        finalize_concept_resolution(
            candidates.occurrences,
            (replace(proposal, left_occurrence_id="unknown"),),
        )


def test_proposal_must_belong_to_its_candidate_inventory(exception_corpus: ChangeCorpus) -> None:
    """Persisted proposals cannot cross the typed candidate boundary."""
    candidates, proposal = _role_proposal(exception_corpus)

    with pytest.raises(ValueError, match="source candidate inventory"):
        validate_concept_proposal_candidates(
            candidates.inventories,
            (replace(proposal, candidate_id="unknown"),),
        )


def test_transitive_same_concept_conflict_is_rejected(exception_corpus: ChangeCorpus) -> None:
    """A non-equivalence relation cannot survive inside one transitive identity component."""
    candidates, template = _role_proposal(exception_corpus)
    first, second = [item for item in candidates.occurrences if item.kind == "LegalSubjectRole"]
    third = replace(second, occurrence_id="role-third", label_vi="Người lái xe")
    proposals = (
        replace(
            template,
            proposal_id="same-1",
            left_occurrence_id=first.occurrence_id,
            right_occurrence_id=third.occurrence_id,
        ),
        replace(
            template,
            proposal_id="same-2",
            left_occurrence_id=third.occurrence_id,
            right_occurrence_id=second.occurrence_id,
        ),
        replace(
            template,
            proposal_id="related",
            left_occurrence_id=first.occurrence_id,
            right_occurrence_id=second.occurrence_id,
            relation="Related",
            preferred_label_vi="",
        ),
    )

    with pytest.raises(ValueError, match="conflict with a same-concept component"):
        finalize_concept_resolution((*candidates.occurrences, third), proposals)


def test_broader_cycle_is_rejected(exception_corpus: ChangeCorpus) -> None:
    """Canonical broader relations must form an acyclic taxonomy."""
    candidates, template = _role_proposal(exception_corpus)
    source = next(item for item in candidates.occurrences if item.kind == "LegalSubjectRole")
    occurrences = (
        replace(source, occurrence_id="role-a", label_vi="Vai trò A"),
        replace(source, occurrence_id="role-b", label_vi="Vai trò B"),
        replace(source, occurrence_id="role-c", label_vi="Vai trò C"),
    )
    proposals = tuple(
        replace(
            template,
            proposal_id=f"broader-{index}",
            left_occurrence_id=left.occurrence_id,
            right_occurrence_id=right.occurrence_id,
            relation="LeftNarrowerThanRight",
            preferred_label_vi="",
        )
        for index, (left, right) in enumerate(
            ((occurrences[0], occurrences[1]), (occurrences[1], occurrences[2]), (occurrences[2], occurrences[0]))
        )
    )

    with pytest.raises(ValueError, match="must be acyclic"):
        finalize_concept_resolution(occurrences, proposals)


def test_same_component_requires_one_preferred_label(exception_corpus: ChangeCorpus) -> None:
    """A transitive equivalence component cannot choose competing canonical labels."""
    candidates, template = _role_proposal(exception_corpus)
    first, second = [item for item in candidates.occurrences if item.kind == "LegalSubjectRole"]
    third = replace(second, occurrence_id="role-third", label_vi="Người lái xe")
    proposals = (
        replace(
            template,
            proposal_id="same-1",
            left_occurrence_id=first.occurrence_id,
            right_occurrence_id=third.occurrence_id,
            preferred_label_vi=first.label_vi,
        ),
        replace(
            template,
            proposal_id="same-2",
            left_occurrence_id=third.occurrence_id,
            right_occurrence_id=second.occurrence_id,
            preferred_label_vi=third.label_vi,
        ),
    )

    with pytest.raises(ValueError, match="must use one preferred label"):
        finalize_concept_resolution((*candidates.occurrences, third), proposals)
