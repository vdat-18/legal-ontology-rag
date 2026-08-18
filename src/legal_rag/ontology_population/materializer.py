"""Materialize typed ontology population records as RDF instances."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

from rdflib import DCTERMS, RDF, RDFS, SKOS, TIME, XSD, Graph, Literal, Namespace, URIRef

from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.models import (
    OntologyPopulationRecord,
    PopulationConsequence,
    PopulationEvidence,
    population_identity,
)
from legal_rag.ontology_validation.loader import load_graph, load_module_policy, merge_graphs
from legal_rag.ontology_validation.paths import OntologyPaths
from legal_rag.ontology_validation.shacl import run_shacl

LORSP = Namespace("https://w3id.org/legal-ontology-rag/ontology/source-provenance#")
LOROC = Namespace("https://w3id.org/legal-ontology-rag/ontology/occurrence-context#")
LORTC = Namespace("https://w3id.org/legal-ontology-rag/ontology/temporal-change#")
LORNR = Namespace("https://w3id.org/legal-ontology-rag/ontology/normative-rule#")
LORLC = Namespace("https://w3id.org/legal-ontology-rag/ontology/legal-consequence#")

CONDITION_CLASSES = {
    "Atomic": LORNR.AtomicCondition,
    "AllOf": LORNR.AllOf,
    "AnyOf": LORNR.AnyOf,
    "Not": LORNR.Not,
}
CONSEQUENCE_CLASSES = {
    "MonetaryPenalty": LORLC.MonetaryPenalty,
    "PointDeduction": LORLC.PointDeduction,
    "AdditionalSanction": LORLC.AdditionalSanction,
    "RemedialMeasure": LORLC.RemedialMeasure,
}


def materialize_population_graph(
    chunks: Sequence[Chunk],
    metadata: DocumentMetadataRecord,
    records: Sequence[OntologyPopulationRecord],
    instance_namespace: str,
) -> Graph:
    """Build a source-grounded RDF instance graph for one legal document."""
    graph = Graph()
    _bind_namespaces(graph, instance_namespace)
    data = Namespace(instance_namespace)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    resource = population_resource_iri(instance_namespace, metadata)
    expression = population_expression_iri(instance_namespace, chunks, metadata)
    graph.add((resource, RDF.type, LORSP.LegalResource))
    graph.add((resource, DCTERMS.identifier, Literal(metadata.document_number)))
    graph.add((resource, LORSP.hasExpression, expression))
    graph.add((expression, RDF.type, LORSP.LegalExpression))

    files = {
        chunk.source_sha256: _iri(
            data, "file", {"source_sha256": chunk.source_sha256, "source_name": chunk.source_name}
        )
        for chunk in chunks
    }
    for chunk in chunks:
        source_file = files[chunk.source_sha256]
        chunk_iri = population_chunk_iri(instance_namespace, chunk.chunk_id)
        graph.add((source_file, RDF.type, LORSP.SourceFile))
        graph.add((source_file, DCTERMS.identifier, Literal(chunk.source_name)))
        graph.add((expression, LORSP.representedBy, source_file))
        graph.add((chunk_iri, RDF.type, LORSP.SourceChunk))
        graph.add((chunk_iri, DCTERMS.identifier, Literal(chunk.chunk_id)))
        graph.add((chunk_iri, LORSP.derivedFromFile, source_file))

    for record in records:
        _materialize_rule(graph, data, expression, chunks_by_id, record)
    return graph


def validate_population_graph(graph: Graph, paths: OntologyPaths) -> str:
    """Validate a populated graph and return the SHACL report text."""
    modules = load_module_policy(paths.project_root, paths.policy)
    ontology_graph = merge_graphs(tuple(module.path for module in modules))
    result = run_shacl(graph, load_graph(paths.shapes), ontology_graph)
    if not result.conforms:
        raise ValueError(result.report_text)
    return result.report_text


def write_population_graph(path: Path, graph: Graph) -> None:
    """Write a validated instance graph as UTF-8 Turtle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph.serialize(format="turtle"), encoding="utf-8")


def _materialize_rule(
    graph: Graph,
    data: Namespace,
    expression: URIRef,
    chunks_by_id: dict[str, Chunk],
    record: OntologyPopulationRecord,
) -> None:
    """Materialize one version-specific penalty rule and its components."""
    rule = population_rule_iri(str(data), record.population_id)
    provision = population_provision_iri(str(data), expression, record.provision_path)
    graph.add((expression, LORSP.containsProvision, provision))
    graph.add((provision, RDF.type, LORSP.LegalProvision))
    graph.add((provision, LORSP.structuralPath, Literal(record.provision_path)))
    graph.add((rule, RDF.type, LORNR.PenaltyRule))
    graph.add((rule, RDFS.label, Literal(record.conduct_label_vi, lang="vi")))
    graph.add((rule, LORSP.groundedIn, provision))

    evidence = _materialize_evidence(graph, data, chunks_by_id, record.evidence)
    for span in evidence:
        graph.add((rule, LORSP.supportedBy, span))

    conduct = _scoped_concept(
        graph, data, record, "conduct", record.conduct_label_vi, LOROC.RegulatedConduct, LOROC.conductScheme
    )
    conduct_specification = URIRef(f"{rule}/conduct")
    graph.add((rule, LORNR.hasConductSpecification, conduct_specification))
    graph.add((conduct_specification, RDF.type, LORNR.ConductSpecification))
    graph.add((conduct_specification, LORNR.usesConductConcept, conduct))
    _add_component_evidence(graph, conduct_specification, provision, evidence)

    applicability, roles = _materialize_applicability(graph, data, rule, provision, evidence, record)
    _materialize_conditions(graph, data, rule, provision, applicability, chunks_by_id, record)
    _materialize_consequences(graph, data, rule, provision, roles, chunks_by_id, record)
    _materialize_validity(graph, rule, record)


def _materialize_applicability(
    graph: Graph,
    data: Namespace,
    rule: URIRef,
    provision: URIRef,
    evidence: Sequence[URIRef],
    record: OntologyPopulationRecord,
) -> tuple[URIRef, dict[str, URIRef]]:
    """Materialize one rule's controlled applicability profile."""
    roles = {
        label: _scoped_concept(graph, data, record, "role", label, LOROC.LegalSubjectRole, LOROC.subjectRoleScheme)
        for label in record.role_labels_vi
    }
    vehicles = {
        label: _scoped_concept(
            graph,
            data,
            record,
            "vehicle-category",
            label,
            LOROC.VehicleCategory,
            LOROC.vehicleCategoryScheme,
        )
        for label in record.vehicle_category_labels_vi
    }
    applicability = URIRef(f"{rule}/applicability")
    graph.add((rule, LORNR.hasApplicability, applicability))
    graph.add((applicability, RDF.type, LORNR.ApplicabilityProfile))
    _add_component_evidence(graph, applicability, provision, evidence)
    if record.subject_kind is not None:
        subject_kind = LOROC.naturalPerson if record.subject_kind == "NaturalPerson" else LOROC.organization
        graph.add((applicability, LORNR.requiresSubjectKind, subject_kind))
    for role in roles.values():
        graph.add((applicability, LORNR.requiresRole, role))
    for vehicle in vehicles.values():
        graph.add((applicability, LORNR.requiresVehicleCategory, vehicle))
    return applicability, roles


def _materialize_conditions(
    graph: Graph,
    data: Namespace,
    rule: URIRef,
    provision: URIRef,
    applicability: URIRef,
    chunks_by_id: dict[str, Chunk],
    record: OntologyPopulationRecord,
) -> None:
    """Materialize a source-evidenced condition-expression tree."""
    conditions = {
        condition.condition_id: _iri(
            data,
            "condition",
            {"population_id": record.population_id, "condition_id": condition.condition_id},
        )
        for condition in record.conditions
    }
    for condition in record.conditions:
        condition_iri = conditions[condition.condition_id]
        graph.add((condition_iri, RDF.type, CONDITION_CLASSES[condition.kind]))
        graph.add((condition_iri, RDFS.label, Literal(condition.text_vi, lang="vi")))
        condition_evidence = _materialize_evidence(graph, data, chunks_by_id, condition.evidence)
        _add_component_evidence(graph, condition_iri, provision, condition_evidence)
        for operand_id in condition.operand_ids:
            graph.add((condition_iri, LORNR.hasOperand, conditions[operand_id]))
    if record.condition_root_id is not None:
        graph.add((applicability, LORNR.hasConditionExpression, conditions[record.condition_root_id]))


def _materialize_consequences(
    graph: Graph,
    data: Namespace,
    rule: URIRef,
    provision: URIRef,
    roles: dict[str, URIRef],
    chunks_by_id: dict[str, Chunk],
    record: OntologyPopulationRecord,
) -> None:
    """Materialize all independently evidenced legal consequences."""
    for consequence in record.consequences:
        consequence_iri = _iri(
            data,
            "consequence",
            {"population_id": record.population_id, "consequence_id": consequence.consequence_id},
        )
        graph.add((consequence_iri, RDF.type, CONSEQUENCE_CLASSES[consequence.kind]))
        graph.add((rule, LORLC.imposesConsequence, consequence_iri))
        consequence_evidence = _materialize_evidence(graph, data, chunks_by_id, consequence.evidence)
        _add_component_evidence(graph, consequence_iri, provision, consequence_evidence)
        _materialize_consequence_values(graph, data, record, consequence_iri, consequence, roles)


def _materialize_validity(graph: Graph, rule: URIRef, record: OntologyPopulationRecord) -> None:
    """Materialize the reviewed validity interval for one rule version."""
    interval = URIRef(f"{rule}/validity")
    graph.add((rule, LORTC.validDuring, interval))
    graph.add((interval, RDF.type, LORTC.ValidityInterval))
    if record.valid_from is not None:
        beginning = URIRef(f"{interval}/beginning")
        graph.add((interval, TIME.hasBeginning, beginning))
        _add_date_instant(graph, beginning, record.valid_from)
    if record.valid_to is not None:
        end = URIRef(f"{interval}/end")
        graph.add((interval, TIME.hasEnd, end))
        _add_date_instant(graph, end, record.valid_to)


def _materialize_consequence_values(
    graph: Graph,
    data: Namespace,
    record: OntologyPopulationRecord,
    consequence_iri: URIRef,
    consequence: PopulationConsequence,
    roles: dict[str, URIRef],
) -> None:
    """Materialize values required by one consequence subtype."""
    if consequence.kind == "MonetaryPenalty":
        amount_range = URIRef(f"{consequence_iri}/amount-range")
        graph.add((consequence_iri, LORLC.hasAmountRange, amount_range))
        graph.add((amount_range, RDF.type, LORLC.MonetaryAmountRange))
        if consequence.minimum_amount is not None:
            graph.add((amount_range, LORLC.minimumAmount, Literal(consequence.minimum_amount, datatype=XSD.decimal)))
        if consequence.maximum_amount is not None:
            graph.add((amount_range, LORLC.maximumAmount, Literal(consequence.maximum_amount, datatype=XSD.decimal)))
        if consequence.currency_code == "VND":
            graph.add((amount_range, LORLC.currency, LORLC.VND))
    elif consequence.kind == "PointDeduction" and consequence.point_value is not None:
        graph.add((consequence_iri, LORLC.pointValue, Literal(consequence.point_value, datatype=XSD.integer)))

    if consequence.target_label_vi is not None:
        target = _iri(
            data,
            "target",
            {"population_id": record.population_id, "label": consequence.target_label_vi},
        )
        graph.add((target, RDF.type, LORLC.ConsequenceTarget))
        graph.add((target, RDFS.label, Literal(consequence.target_label_vi, lang="vi")))
        graph.add((consequence_iri, LORLC.affectsTarget, target))
    if consequence.responsible_role_label_vi is not None:
        role = roles.get(consequence.responsible_role_label_vi)
        if role is None:
            role = _scoped_concept(
                graph,
                data,
                record,
                "role",
                consequence.responsible_role_label_vi,
                LOROC.LegalSubjectRole,
                LOROC.subjectRoleScheme,
            )
        graph.add((consequence_iri, LORLC.mustBePerformedBy, role))


def _materialize_evidence(
    graph: Graph,
    data: Namespace,
    chunks_by_id: dict[str, Chunk],
    evidence: Sequence[PopulationEvidence],
) -> tuple[URIRef, ...]:
    """Materialize exact evidence spans and their chunk containment."""
    spans = []
    for item in evidence:
        chunk = chunks_by_id[item.chunk_id]
        chunk_iri = population_chunk_iri(str(data), chunk.chunk_id)
        span = population_evidence_span_iri(str(data), item)
        graph.add((span, RDF.type, LORSP.EvidenceSpan))
        graph.add((span, LORSP.exactText, Literal(item.exact_text, lang="vi")))
        graph.add((chunk_iri, LORSP.containsSpan, span))
        spans.append(span)
    return tuple(spans)


def _scoped_concept(
    graph: Graph,
    data: Namespace,
    record: OntologyPopulationRecord,
    kind: str,
    label: str,
    concept_class: URIRef,
    scheme: URIRef,
) -> URIRef:
    """Create a rule-scoped concept without asserting cross-rule equivalence."""
    concept = population_scoped_concept_iri(str(data), record.population_id, kind, label)
    graph.add((concept, RDF.type, concept_class))
    graph.add((concept, SKOS.inScheme, scheme))
    graph.add((concept, SKOS.prefLabel, Literal(label, lang="vi")))
    return concept


def population_scoped_concept_iri(
    instance_namespace: str,
    population_id: str,
    kind: str,
    label: str,
) -> URIRef:
    """Return one rule-scoped controlled-concept IRI."""
    return _iri(
        Namespace(instance_namespace),
        f"concept/{kind}",
        {"population_id": population_id, "label": label},
    )


def _add_component_evidence(
    graph: Graph,
    component: URIRef,
    provision: URIRef,
    evidence: Sequence[URIRef],
) -> None:
    """Attach authoritative provision and exact evidence to a material component."""
    graph.add((component, LORSP.groundedIn, provision))
    for span in evidence:
        graph.add((component, LORSP.supportedBy, span))


def _add_date_instant(graph: Graph, instant: URIRef, value: date) -> None:
    """Materialize one date as a midnight UTC OWL-Time instant."""
    graph.add((instant, RDF.type, TIME.Instant))
    graph.add((instant, TIME.inXSDDateTime, Literal(f"{value.isoformat()}T00:00:00Z", datatype=XSD.dateTime)))


def _iri(namespace: Namespace, kind: str, content: dict[str, object]) -> URIRef:
    """Build one deterministic instance IRI from canonical content."""
    return URIRef(f"{namespace}{kind}/{population_identity(content)}")


def population_resource_iri(instance_namespace: str, metadata: DocumentMetadataRecord) -> URIRef:
    """Return the stable legal-resource IRI for one document."""
    return _iri(Namespace(instance_namespace), "resource", {"document_number": metadata.document_number})


def population_expression_iri(
    instance_namespace: str,
    chunks: Sequence[Chunk],
    metadata: DocumentMetadataRecord,
) -> URIRef:
    """Return the stable legal-expression IRI for one document artifact set."""
    return _iri(
        Namespace(instance_namespace),
        "expression",
        {
            "document_number": metadata.document_number,
            "source_hashes": sorted({chunk.source_sha256 for chunk in chunks}),
        },
    )


def population_rule_iri(instance_namespace: str, population_id: str) -> URIRef:
    """Return the stable rule-version IRI for one population record."""
    return URIRef(f"{instance_namespace}rule/{population_id}")


def population_provision_iri(instance_namespace: str, expression: URIRef, provision_path: str) -> URIRef:
    """Return the stable provision IRI within one legal expression."""
    return _iri(
        Namespace(instance_namespace),
        "provision",
        {"expression": str(expression), "structural_path": provision_path},
    )


def population_chunk_iri(instance_namespace: str, chunk_id: str) -> URIRef:
    """Return the stable source-chunk IRI."""
    return _iri(Namespace(instance_namespace), "chunk", {"chunk_id": chunk_id})


def population_evidence_span_iri(instance_namespace: str, evidence: PopulationEvidence) -> URIRef:
    """Return the stable exact-evidence-span IRI."""
    return _iri(Namespace(instance_namespace), "span", evidence.to_dict())


def _bind_namespaces(graph: Graph, instance_namespace: str) -> None:
    """Bind stable prefixes for readable Turtle output."""
    graph.bind("data", Namespace(instance_namespace))
    graph.bind("lorsp", LORSP)
    graph.bind("loroc", LOROC)
    graph.bind("lortc", LORTC)
    graph.bind("lornr", LORNR)
    graph.bind("lorlc", LORLC)
    graph.bind("skos", SKOS)
    graph.bind("time", TIME)
