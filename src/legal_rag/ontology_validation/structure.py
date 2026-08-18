"""Validate ontology structure against the versioned policy."""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

from defusedxml import ElementTree
from rdflib import OWL, RDF, RDFS, SKOS, Graph, URIRef

from legal_rag.ontology_validation.models import ModulePolicy

FORBIDDEN_OWL_PREDICATES = frozenset(
    {
        OWL.cardinality,
        OWL.maxCardinality,
        OWL.maxQualifiedCardinality,
        OWL.minCardinality,
        OWL.minQualifiedCardinality,
        OWL.qualifiedCardinality,
        OWL.equivalentClass,
        OWL.equivalentProperty,
        OWL.disjointWith,
        OWL.propertyDisjointWith,
    }
)
LOCAL_TERM_TYPES = frozenset(
    {
        OWL.Class,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
        SKOS.Concept,
        SKOS.ConceptScheme,
    }
)
SWRL_NAMESPACE = "http://www.w3.org/2003/11/swrl#"


def ontology_imports(graph: Graph, ontology_iri: str) -> frozenset[str]:
    """Return direct imports declared by one ontology."""
    subject = URIRef(ontology_iri)
    return frozenset(str(value) for value in graph.objects(subject, OWL.imports))


def validate_import_dag(modules: tuple[ModulePolicy, ...]) -> bool:
    """Return whether local module imports form an acyclic graph."""
    indegree = {module.module_id: 0 for module in modules}
    dependents: dict[str, list[str]] = {module.module_id: [] for module in modules}
    for module in modules:
        for dependency in module.imports:
            indegree[module.module_id] += 1
            dependents[dependency].append(module.module_id)

    queue = deque(module_id for module_id, count in indegree.items() if count == 0)
    visited = 0
    while queue:
        module_id = queue.popleft()
        visited += 1
        for dependent in dependents[module_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)
    return visited == len(modules)


def catalog_mappings(catalog_path: Path) -> dict[str, Path]:
    """Return canonical IRI-to-file mappings from the XML catalog."""
    root = ElementTree.parse(catalog_path).getroot()
    namespace = {"catalog": "urn:oasis:names:tc:entity:xmlns:xml:catalog"}
    return {
        element.attrib["name"]: catalog_path.parent / element.attrib["uri"]
        for element in root.findall("catalog:uri", namespace)
    }


def forbidden_constructs(graph: Graph) -> tuple[tuple[object, object, object], ...]:
    """Return triples using deferred OWL axioms or SWRL vocabulary."""
    return tuple(
        (subject, predicate, value)
        for subject, predicate, value in graph
        if predicate in FORBIDDEN_OWL_PREDICATES
        or any(str(node).startswith(SWRL_NAMESPACE) for node in (subject, predicate, value))
    )


def local_terms(graph: Graph, namespaces: tuple[str, ...]) -> frozenset[URIRef]:
    """Return declared OWL and SKOS terms in local namespaces."""
    concept_classes = frozenset(graph.subjects(RDFS.subClassOf, SKOS.Concept))
    return frozenset(
        subject
        for subject, term_type in graph.subject_objects(RDF.type)
        if isinstance(subject, URIRef)
        and str(subject).startswith(namespaces)
        and (term_type in LOCAL_TERM_TYPES or term_type in concept_classes)
    )


def undocumented_terms(graph: Graph, namespaces: tuple[str, ...]) -> frozenset[URIRef]:
    """Return local terms missing human-readable labels or descriptions."""
    missing = set()
    concept_classes = frozenset(graph.subjects(RDFS.subClassOf, SKOS.Concept))
    for term in local_terms(graph, namespaces):
        term_types = set(graph.objects(term, RDF.type))
        is_controlled_value = bool(term_types & ({SKOS.Concept, SKOS.ConceptScheme} | concept_classes))
        label_predicate = SKOS.prefLabel if is_controlled_value else RDFS.label
        lacks_description = not is_controlled_value and not any(graph.objects(term, RDFS.comment))
        if not any(graph.objects(term, label_predicate)) or lacks_description:
            missing.add(term)
    return frozenset(missing)


def duplicate_module_ids(modules: tuple[ModulePolicy, ...]) -> frozenset[str]:
    """Return module identifiers declared more than once."""
    counts = Counter(module.module_id for module in modules)
    return frozenset(module_id for module_id, count in counts.items() if count > 1)
