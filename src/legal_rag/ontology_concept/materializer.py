"""Materialize canonical concepts and SKOS taxonomy relations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rdflib import RDF, SKOS, Graph, Literal, Namespace, URIRef

from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_change.models import LegalChangeEvent
from legal_rag.ontology_concept.models import (
    CanonicalConceptRecord,
    CanonicalConceptRelationRecord,
    ConceptOccurrence,
)
from legal_rag.ontology_exception.materializer import materialize_legal_exception_graph
from legal_rag.ontology_exception.models import LegalExceptionRecord
from legal_rag.ontology_population.materializer import (
    LOROC,
    LORSP,
    population_chunk_iri,
    population_scoped_concept_iri,
)
from legal_rag.ontology_population.models import OntologyPopulationRecord

KIND_RDF = {
    "RegulatedConduct": (LOROC.RegulatedConduct, LOROC.conductScheme, "conduct"),
    "LegalSubjectRole": (LOROC.LegalSubjectRole, LOROC.subjectRoleScheme, "role"),
    "VehicleCategory": (LOROC.VehicleCategory, LOROC.vehicleCategoryScheme, "vehicle-category"),
}


def materialize_concept_resolution_graph(
    corpus: ChangeCorpus,
    population_records: Sequence[OntologyPopulationRecord],
    change_events: Sequence[LegalChangeEvent],
    exceptions: Sequence[LegalExceptionRecord],
    occurrences: Sequence[ConceptOccurrence],
    concepts: Sequence[CanonicalConceptRecord],
    relations: Sequence[CanonicalConceptRelationRecord],
    instance_namespace: str,
) -> Graph:
    """Build the canonical graph and replace rule-scoped concepts with canonical identities."""
    graph = materialize_legal_exception_graph(
        corpus,
        population_records,
        change_events,
        exceptions,
        instance_namespace,
    )
    occurrences_by_id = {item.occurrence_id: item for item in occurrences}
    canonical_by_occurrence = {
        occurrence_id: concept for concept in concepts for occurrence_id in concept.occurrence_ids
    }
    for occurrence in occurrences:
        canonical = canonical_by_occurrence[occurrence.occurrence_id]
        old_iri = _occurrence_iri(occurrence, instance_namespace)
        canonical_iri = canonical_concept_iri(instance_namespace, canonical.concept_id)
        _replace_node(graph, old_iri, canonical_iri)
    for concept in concepts:
        _materialize_concept(graph, concept, occurrences_by_id, instance_namespace)
    for relation in relations:
        _materialize_relation(graph, relation, instance_namespace)
    return graph


def canonical_concept_iri(instance_namespace: str, concept_id: str) -> URIRef:
    """Return one canonical concept IRI."""
    return URIRef(f"{Namespace(instance_namespace)}concept/canonical/{concept_id}")


def _occurrence_iri(occurrence: ConceptOccurrence, instance_namespace: str) -> URIRef:
    """Return the scoped RDF node created by population materialization."""
    return population_scoped_concept_iri(
        instance_namespace,
        occurrence.population_id,
        KIND_RDF[occurrence.kind][2],
        occurrence.label_vi,
    )


def _replace_node(graph: Graph, old: URIRef, new: URIRef) -> None:
    """Replace one scoped concept node in both subject and object positions."""
    subject_triples = tuple(graph.triples((old, None, None)))
    object_triples = tuple(graph.triples((None, None, old)))
    for _, predicate, value in subject_triples:
        graph.remove((old, predicate, value))
    for subject, predicate, _ in object_triples:
        graph.remove((subject, predicate, old))
        graph.add((subject, predicate, new))


def _materialize_concept(
    graph: Graph,
    concept: CanonicalConceptRecord,
    occurrences_by_id: Mapping[str, ConceptOccurrence],
    instance_namespace: str,
) -> None:
    """Materialize one canonical controlled concept with aliases and source chunks."""
    concept_iri = canonical_concept_iri(instance_namespace, concept.concept_id)
    concept_class, scheme, _ = KIND_RDF[concept.kind]
    graph.add((concept_iri, RDF.type, concept_class))
    graph.add((concept_iri, SKOS.inScheme, scheme))
    graph.add((concept_iri, SKOS.prefLabel, Literal(concept.preferred_label_vi, lang="vi")))
    for label in concept.alternative_labels_vi:
        graph.add((concept_iri, SKOS.altLabel, Literal(label, lang="vi")))
    evidence = {item for occurrence_id in concept.occurrence_ids for item in occurrences_by_id[occurrence_id].evidence}
    for item in evidence:
        graph.add((concept_iri, LORSP.sourcedFromChunk, population_chunk_iri(instance_namespace, item.chunk_id)))


def _materialize_relation(
    graph: Graph,
    relation: CanonicalConceptRelationRecord,
    instance_namespace: str,
) -> None:
    """Materialize SKOS relations while retaining distinct decisions only in JSON."""
    source = canonical_concept_iri(instance_namespace, relation.source_concept_id)
    target = canonical_concept_iri(instance_namespace, relation.target_concept_id)
    if relation.relation == "Broader":
        graph.add((source, SKOS.broader, target))
        graph.add((target, SKOS.narrower, source))
    elif relation.relation == "Related":
        graph.add((source, SKOS.related, target))
        graph.add((target, SKOS.related, source))
