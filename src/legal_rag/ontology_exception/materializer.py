"""Materialize legal exceptions and grouped provision citations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rdflib import RDF, Graph, Literal, Namespace, URIRef

from legal_rag.ontology_change.corpus import ChangeCorpus, ChangeCorpusDocument
from legal_rag.ontology_change.materializer import materialize_legal_change_graph
from legal_rag.ontology_change.models import LegalChangeEvent
from legal_rag.ontology_exception.models import ExceptionCitationResolution, LegalExceptionRecord
from legal_rag.ontology_population.materializer import (
    LORNR,
    LORSP,
    population_chunk_iri,
    population_expression_iri,
    population_provision_iri,
    population_rule_iri,
)
from legal_rag.ontology_population.models import OntologyPopulationRecord, population_identity


def materialize_legal_exception_graph(
    corpus: ChangeCorpus,
    population_records: Sequence[OntologyPopulationRecord],
    change_events: Sequence[LegalChangeEvent],
    exceptions: Sequence[LegalExceptionRecord],
    instance_namespace: str,
) -> Graph:
    """Build one canonical graph with rules, changes, exceptions, and citations."""
    graph = materialize_legal_change_graph(corpus, population_records, change_events, instance_namespace)
    documents_by_number = {document.metadata.document_number: document for document in corpus.documents}
    records_by_id = {record.population_id: record for record in population_records}
    for exception in exceptions:
        _materialize_exception(
            graph,
            exception,
            documents_by_number,
            records_by_id,
            instance_namespace,
        )
    return graph


def _materialize_exception(
    graph: Graph,
    exception: LegalExceptionRecord,
    documents_by_number: Mapping[str, ChangeCorpusDocument],
    records_by_id: Mapping[str, OntologyPopulationRecord],
    instance_namespace: str,
) -> None:
    """Materialize one exception and its grouped citation objects."""
    document = documents_by_number[exception.source_document_number]
    expression = population_expression_iri(instance_namespace, document.chunks, document.metadata)
    provision = population_provision_iri(instance_namespace, expression, exception.exception_provision_path)
    exception_iri = URIRef(f"{Namespace(instance_namespace)}exception/{exception.exception_id}")
    general_rule = population_rule_iri(instance_namespace, exception.general_population_id)
    graph.add((expression, LORSP.containsProvision, provision))
    graph.add((provision, RDF.type, LORSP.LegalProvision))
    graph.add((provision, LORSP.structuralPath, Literal(exception.exception_provision_path)))
    graph.add((exception_iri, RDF.type, LORNR.LegalException))
    graph.add((general_rule, LORNR.hasException, exception_iri))
    graph.add((exception_iri, LORNR.exceptionTo, general_rule))
    graph.add((exception_iri, LORSP.groundedIn, provision))

    chunks_by_id = {chunk.chunk_id: chunk for chunk in document.chunks}
    for evidence in exception.evidence:
        chunk = chunks_by_id[evidence.chunk_id]
        chunk_iri = population_chunk_iri(instance_namespace, chunk.chunk_id)
        graph.add((exception_iri, LORSP.sourcedFromChunk, chunk_iri))

    for citation in exception.citations:
        _materialize_citation(
            graph,
            exception_iri,
            exception,
            citation,
            documents_by_number,
            records_by_id,
            instance_namespace,
        )


def _materialize_citation(
    graph: Graph,
    exception_iri: URIRef,
    exception: LegalExceptionRecord,
    citation: ExceptionCitationResolution,
    documents_by_number: Mapping[str, ChangeCorpusDocument],
    records_by_id: Mapping[str, OntologyPopulationRecord],
    instance_namespace: str,
) -> None:
    """Materialize one grouped locator and its provision/rule targets."""
    citation_id = population_identity(
        {
            "exception_id": exception.exception_id,
            "citation_text": citation.citation_text,
            "source_chunk_id": citation.source_chunk_id,
            "target_population_ids": citation.target_population_ids,
        }
    )
    citation_iri = URIRef(f"{Namespace(instance_namespace)}citation/{citation_id}")
    graph.add((citation_iri, RDF.type, LORSP.ProvisionCitation))
    graph.add((citation_iri, LORSP.citationText, Literal(citation.citation_text, lang="vi")))
    graph.add((exception_iri, LORNR.hasProvisionCitation, citation_iri))
    for target_id in citation.target_population_ids:
        target = records_by_id[target_id]
        target_document = documents_by_number[target.document_number]
        target_expression = population_expression_iri(
            instance_namespace,
            target_document.chunks,
            target_document.metadata,
        )
        target_provision = population_provision_iri(
            instance_namespace,
            target_expression,
            target.provision_path,
        )
        graph.add((citation_iri, LORSP.refersToProvision, target_provision))
        graph.add((exception_iri, LORNR.citesExcludedProvision, target_provision))
        graph.add((exception_iri, LORNR.redirectsTo, population_rule_iri(instance_namespace, target_id)))
