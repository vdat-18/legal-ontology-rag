"""Materialize canonical multi-document rules and legal changes."""

from __future__ import annotations

from collections.abc import Sequence

from rdflib import RDF, TIME, XSD, Graph, Literal, Namespace, URIRef

from legal_rag.ontology_change.corpus import ChangeCorpus, ChangeCorpusDocument
from legal_rag.ontology_change.models import LegalChangeEvent
from legal_rag.ontology_population.materializer import (
    LORSP,
    LORTC,
    materialize_population_graph,
    population_chunk_iri,
    population_expression_iri,
    population_provision_iri,
    population_rule_iri,
)
from legal_rag.ontology_population.models import OntologyPopulationRecord

CHANGE_TYPE_IRIS = {
    "Amendment": LORTC.amendment,
    "Replacement": LORTC.replacement,
    "RepealOrSupersession": LORTC.repealOrSupersession,
}


def materialize_legal_change_graph(
    corpus: ChangeCorpus,
    records: Sequence[OntologyPopulationRecord],
    events: Sequence[LegalChangeEvent],
    instance_namespace: str,
) -> Graph:
    """Build one canonical multi-document graph with legal changes."""
    graph = Graph()
    event_document_numbers = {event.change_document_number for event in events}
    for document in corpus.documents:
        document_records = tuple(
            record for record in records if record.document_number == document.metadata.document_number
        )
        if document_records or document.metadata.document_number in event_document_numbers:
            graph += materialize_population_graph(
                document.chunks,
                document.metadata,
                document_records,
                instance_namespace,
            )

    documents_by_number = {document.metadata.document_number: document for document in corpus.documents}
    for event in events:
        _materialize_change_event(
            graph,
            documents_by_number[event.change_document_number],
            event,
            instance_namespace,
        )
    return graph


def _materialize_change_event(
    graph: Graph,
    change_document: ChangeCorpusDocument,
    event: LegalChangeEvent,
    instance_namespace: str,
) -> None:
    """Materialize one change with source provenance."""
    data = Namespace(instance_namespace)
    expression = population_expression_iri(instance_namespace, change_document.chunks, change_document.metadata)
    provision = population_provision_iri(instance_namespace, expression, event.change_provision_path)
    change = URIRef(f"{data}change/{event.change_id}")
    effective = URIRef(f"{change}/effective")
    graph.add((expression, LORSP.containsProvision, provision))
    graph.add((provision, RDF.type, LORSP.LegalProvision))
    graph.add((provision, LORSP.structuralPath, Literal(event.change_provision_path)))
    graph.add((change, RDF.type, LORTC.LegalChange))
    graph.add((change, LORTC.changeType, CHANGE_TYPE_IRIS[event.change_type]))
    graph.add((change, LORTC.effectiveAt, effective))
    graph.add((effective, RDF.type, TIME.Instant))
    graph.add(
        (
            effective,
            TIME.inXSDDateTime,
            Literal(f"{event.effective_on.isoformat()}T00:00:00Z", datatype=XSD.dateTime),
        )
    )
    graph.add((change, LORTC.changesEarlierRule, population_rule_iri(instance_namespace, event.earlier_population_id)))
    if event.later_population_id is not None:
        graph.add((change, LORTC.producesLaterRule, population_rule_iri(instance_namespace, event.later_population_id)))
    graph.add((change, LORSP.groundedIn, provision))

    chunks_by_id = {chunk.chunk_id: chunk for chunk in change_document.chunks}
    for evidence in event.evidence:
        chunk = chunks_by_id[evidence.chunk_id]
        chunk_iri = population_chunk_iri(instance_namespace, chunk.chunk_id)
        graph.add((change, LORSP.sourcedFromChunk, chunk_iri))
