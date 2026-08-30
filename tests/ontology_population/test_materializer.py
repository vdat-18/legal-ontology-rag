"""Tests for RDF materialization and SHACL gating."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from rdflib import RDF, TIME, XSD, Literal

from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.materializer import (
    LORLC,
    LORNR,
    LOROC,
    LORSP,
    materialize_population_graph,
    validate_population_graph,
)
from legal_rag.ontology_population.models import OntologyPopulationRecord
from legal_rag.ontology_validation.paths import OntologyPaths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTANCE_NAMESPACE = "https://w3id.org/legal-ontology-rag/data/"


def test_materialized_rule_covers_all_components_and_conforms(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
    population_record: OntologyPopulationRecord,
) -> None:
    """A complete typed record becomes a SHACL-conformant ontology graph."""
    graph = materialize_population_graph((legal_chunk,), document_metadata, (population_record,), INSTANCE_NAMESPACE)

    validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))
    assert len(tuple(graph.subjects(RDF.type, LORNR.PenaltyRule))) == 1
    assert len(tuple(graph.subjects(RDF.type, LORLC.MonetaryPenalty))) == 1
    assert len(tuple(graph.subjects(RDF.type, LORLC.PointDeduction))) == 1
    assert len(tuple(graph.subjects(RDF.type, LORLC.AdditionalSanction))) == 1
    assert len(tuple(graph.subjects(RDF.type, LORLC.RemedialMeasure))) == 1
    assert not tuple(graph.triples((None, LORSP.supportedBy, None)))
    assert len(tuple(graph.triples((None, LORSP.sourcedFromChunk, None)))) >= 8
    sanction = next(graph.subjects(RDF.type, LORLC.AdditionalSanction))
    assert (sanction, LORLC.minimumDurationMonths, Literal(10, datatype=XSD.integer)) in graph
    assert (sanction, LORLC.maximumDurationMonths, Literal(12, datatype=XSD.integer)) in graph
    assert not tuple(graph.objects(predicate=TIME.hasEnd))


def test_missing_reviewed_validity_is_rejected_by_shacl(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
    population_record: OntologyPopulationRecord,
) -> None:
    """Unknown validity is preserved as incomplete data rather than guessed."""
    graph = materialize_population_graph(
        (legal_chunk,),
        document_metadata,
        (replace(population_record, valid_from=None),),
        INSTANCE_NAMESPACE,
    )

    with pytest.raises(ValueError, match="verified beginning instant"):
        validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))


def test_materialization_is_deterministic(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
    population_record: OntologyPopulationRecord,
) -> None:
    """The same reviewed records produce the same RDF triple set."""
    first = materialize_population_graph((legal_chunk,), document_metadata, (population_record,), INSTANCE_NAMESPACE)
    second = materialize_population_graph((legal_chunk,), document_metadata, (population_record,), INSTANCE_NAMESPACE)

    assert set(first) == set(second)


def test_logical_condition_operator_requires_source_chunk_provenance(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
    population_record: OntologyPopulationRecord,
) -> None:
    """Every condition expression remains independently traceable to a source chunk."""
    logical_root = replace(population_record.conditions[0], evidence=())
    graph = materialize_population_graph(
        (legal_chunk,),
        document_metadata,
        (replace(population_record, conditions=(logical_root, *population_record.conditions[1:])),),
        INSTANCE_NAMESPACE,
    )

    with pytest.raises(ValueError, match="source chunk"):
        validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))


def test_single_operand_all_of_is_rejected(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
    population_record: OntologyPopulationRecord,
) -> None:
    """A redundant unary group cannot weaken the reviewed condition-expression model."""
    logical_root = replace(population_record.conditions[0], operand_ids=("required",))
    graph = materialize_population_graph(
        (legal_chunk,),
        document_metadata,
        (replace(population_record, conditions=(logical_root, *population_record.conditions[1:])),),
        INSTANCE_NAMESPACE,
    )

    with pytest.raises(ValueError, match="at least two"):
        validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))


def test_incomplete_monetary_values_are_rejected_by_shacl(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
    population_record: OntologyPopulationRecord,
) -> None:
    """Missing monetary values remain invalid instead of becoming zero."""
    incomplete_money = replace(population_record.consequences[0], maximum_amount=None)
    graph = materialize_population_graph(
        (legal_chunk,),
        document_metadata,
        (replace(population_record, consequences=(incomplete_money, *population_record.consequences[1:])),),
        INSTANCE_NAMESPACE,
    )

    with pytest.raises(ValueError, match="upper bound"):
        validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))


def test_incomplete_additional_sanction_duration_is_rejected_by_shacl(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
    population_record: OntologyPopulationRecord,
) -> None:
    """A persisted additional-sanction duration must keep both source bounds."""
    sanction = replace(population_record.consequences[2], maximum_duration_months=None)
    graph = materialize_population_graph(
        (legal_chunk,),
        document_metadata,
        (replace(population_record, consequences=(*population_record.consequences[:2], sanction)),),
        INSTANCE_NAMESPACE,
    )

    with pytest.raises(ValueError, match="both month bounds"):
        validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))


def test_unresolved_condition_operand_fails_materialization(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
    population_record: OntologyPopulationRecord,
) -> None:
    """A model cannot emit condition references outside its typed rule record."""
    invalid_root = replace(population_record.conditions[0], operand_ids=("required", "missing"))

    with pytest.raises(KeyError, match="missing"):
        materialize_population_graph(
            (legal_chunk,),
            document_metadata,
            (replace(population_record, conditions=(invalid_root, *population_record.conditions[1:])),),
            INSTANCE_NAMESPACE,
        )


def test_candidate_concepts_remain_scoped_to_each_rule(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
    population_record: OntologyPopulationRecord,
) -> None:
    """Equal labels do not silently assert semantic identity across rules."""
    graph = materialize_population_graph(
        (legal_chunk,),
        document_metadata,
        (population_record, replace(population_record, population_id="population-2")),
        INSTANCE_NAMESPACE,
    )

    assert len(tuple(graph.subjects(RDF.type, LOROC.RegulatedConduct))) == 2
