"""Tests for reviewed concept finalization and RDF canonicalization."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from rdflib import SKOS

from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_concept.candidates import build_concept_candidates
from legal_rag.ontology_concept.config import ConceptResolutionSettings
from legal_rag.ontology_concept.finalization import finalize_concept_resolution
from legal_rag.ontology_concept.materializer import canonical_concept_iri, materialize_concept_resolution_graph
from legal_rag.ontology_concept.models import ConceptRelationProposal, ConceptReview
from legal_rag.ontology_concept.writer import (
    pending_concept_review_template,
    read_canonical_concepts,
    read_canonical_relations,
    write_canonical_concepts,
    write_canonical_relations,
)
from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.materializer import LORNR, validate_population_graph
from legal_rag.ontology_validation.paths import OntologyPaths

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _accepted_role_proposal(exception_corpus: ChangeCorpus) -> tuple[object, ConceptRelationProposal, ConceptReview]:
    """Return candidates and one accepted exact-evidenced role equivalence."""
    settings = ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml")
    candidates = build_concept_candidates(exception_corpus.documents[0].records, settings)
    occurrences = {item.occurrence_id: item for item in candidates.occurrences}
    pair = next(item for item in candidates.pairs if occurrences[item.left_occurrence_id].kind == "LegalSubjectRole")
    left = occurrences[pair.left_occurrence_id]
    right = occurrences[pair.right_occurrence_id]
    proposal = ConceptRelationProposal(
        proposal_id="proposal-role",
        candidate_id=pair.candidate_id,
        left_occurrence_id=left.occurrence_id,
        right_occurrence_id=right.occurrence_id,
        relation="SameConcept",
        preferred_label_vi=left.label_vi,
        left_evidence=left.evidence,
        right_evidence=right.evidence,
        rationale_vi="Cùng vai trò pháp lý.",
        proposed_by_model="test-model",
        prompt_version="0.1.0",
        run_id="run-1",
    )
    review = ConceptReview(
        proposal_id=proposal.proposal_id,
        decision="accepted",
        reviewer="legal-reviewer",
        reviewed_on=date(2026, 8, 18),
        note="Đã đối chiếu hai nguồn.",
    )
    return candidates, proposal, review


def test_accepted_equivalence_remaps_rules_and_passes_shacl(
    tmp_path: Path,
    exception_corpus: ChangeCorpus,
) -> None:
    """Accepted equality creates one canonical role used by both rule applicability profiles."""
    candidates, proposal, review = _accepted_role_proposal(exception_corpus)
    settings = ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml")
    concepts, relations = finalize_concept_resolution(candidates.occurrences, (proposal,), (review,), settings)
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
    assert relations == ()
    assert validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))
    assert read_canonical_concepts(concepts_path) == concepts
    assert read_canonical_relations(relations_path) == relations


@pytest.mark.parametrize(
    ("reviews", "message"),
    [
        ((), "requires a human review"),
        ((ConceptReview("proposal-role", "pending", "", None, ""),), "Pending concept reviews"),
        (
            (ConceptReview("proposal-role", "accepted", "", date(2026, 8, 18), ""),),
            "require reviewer identity",
        ),
        (
            (ConceptReview("unknown", "accepted", "reviewer", date(2026, 8, 18), ""),),
            "must reference known proposals",
        ),
    ],
)
def test_incomplete_or_invalid_review_blocks_finalization(
    reviews: tuple[ConceptReview, ...],
    message: str,
    exception_corpus: ChangeCorpus,
) -> None:
    """No concept merge enters the canonical graph without complete attributable review."""
    candidates, proposal, _ = _accepted_role_proposal(exception_corpus)
    settings = ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml")

    with pytest.raises(ValueError, match=message):
        finalize_concept_resolution(candidates.occurrences, (proposal,), reviews, settings)


def test_rejected_proposal_keeps_every_occurrence_as_a_separate_concept(exception_corpus: ChangeCorpus) -> None:
    """Rejection never merges source-scoped concept occurrences."""
    candidates, proposal, review = _accepted_role_proposal(exception_corpus)
    rejected = ConceptReview(proposal.proposal_id, "rejected", review.reviewer, review.reviewed_on, review.note)
    settings = ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml")
    concepts, relations = finalize_concept_resolution(candidates.occurrences, (proposal,), (rejected,), settings)

    assert len(concepts) == len(candidates.occurrences)
    assert relations == ()


def test_reviewed_broader_relation_materializes_as_skos_without_merging(exception_corpus: ChangeCorpus) -> None:
    """A reviewed hierarchy keeps distinct identities and emits inverse SKOS projections."""
    settings = ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml")
    candidates = build_concept_candidates(exception_corpus.documents[0].records, settings)
    occurrences = {item.occurrence_id: item for item in candidates.occurrences}
    pair = next(item for item in candidates.pairs if occurrences[item.left_occurrence_id].kind == "RegulatedConduct")
    left = occurrences[pair.left_occurrence_id]
    right = occurrences[pair.right_occurrence_id]
    proposal = ConceptRelationProposal(
        proposal_id="proposal-hierarchy",
        candidate_id=pair.candidate_id,
        left_occurrence_id=left.occurrence_id,
        right_occurrence_id=right.occurrence_id,
        relation="LeftBroaderThanRight",
        preferred_label_vi="",
        left_evidence=left.evidence,
        right_evidence=right.evidence,
        rationale_vi="Left là concept rộng hơn theo source đã review.",
        proposed_by_model="test-model",
        prompt_version="0.1.0",
        run_id="run-1",
    )
    review = ConceptReview(
        proposal.proposal_id,
        "accepted",
        "legal-reviewer",
        date(2026, 8, 18),
        "Đã đối chiếu taxonomy.",
    )
    concepts, relations = finalize_concept_resolution(candidates.occurrences, (proposal,), (review,), settings)
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
    concept_by_occurrence = {occurrence_id: concept for concept in concepts for occurrence_id in concept.occurrence_ids}
    narrower = canonical_concept_iri(
        population_settings.instance_namespace,
        concept_by_occurrence[right.occurrence_id].concept_id,
    )
    broader = canonical_concept_iri(
        population_settings.instance_namespace,
        concept_by_occurrence[left.occurrence_id].concept_id,
    )

    assert len(concepts) == len(candidates.occurrences)
    assert relations[0].relation == "Broader"
    assert (narrower, SKOS.broader, broader) in graph
    assert (broader, SKOS.narrower, narrower) in graph
    assert validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))


def test_pending_template_never_preapproves_semantics(exception_corpus: ChangeCorpus) -> None:
    """Generated review rows remain pending and unattributed."""
    _, proposal, _ = _accepted_role_proposal(exception_corpus)
    review = pending_concept_review_template((proposal,))[0]

    assert review.decision == "pending"
    assert review.reviewer == ""
    assert review.reviewed_on is None
