"""Compare versioned ontology projects and reject semantic breaking changes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rdflib import DCTERMS, OWL, RDF, RDFS, SKOS, BNode, Graph, Literal, Namespace, URIRef
from rdflib.compare import to_canonical_graph

from legal_rag.ontology_validation.evolution_config import OntologyEvolutionSettings
from legal_rag.ontology_validation.evolution_models import (
    OntologyEvolutionIssue,
    OntologyEvolutionReport,
    OntologyReplacement,
    OntologySemanticHashChange,
    OntologyTermKindChange,
    OntologyTermSnapshot,
    TermKind,
)
from legal_rag.ontology_validation.loader import load_graph, load_module_policy, load_yaml, merge_graphs
from legal_rag.ontology_validation.paths import OntologyPaths

_SH = Namespace("http://www.w3.org/ns/shacl#")
_TERM_TYPES: tuple[tuple[URIRef, TermKind], ...] = (
    (OWL.Class, "class"),
    (OWL.DatatypeProperty, "datatype_property"),
    (OWL.ObjectProperty, "object_property"),
)


@dataclass(frozen=True, slots=True)
class _ProjectSnapshot:
    """Comparable semantic inventory for one ontology project root."""

    terms: dict[str, OntologyTermSnapshot]
    term_semantic_hashes: dict[str, str]
    node_shape_iris: frozenset[str]
    node_shape_constraint_hashes: dict[str, str]
    mandatory_cq_ids: frozenset[str]
    deprecated_iris: frozenset[str]
    replacements: tuple[OntologyReplacement, ...]


def compare_ontology_projects(
    before_project_root: Path,
    after_project_root: Path,
    settings: OntologyEvolutionSettings,
    checked_at: datetime,
) -> OntologyEvolutionReport:
    """Return a stable semantic diff and fail-closed breaking-change findings."""
    before = _project_snapshot(before_project_root, settings)
    after = _project_snapshot(after_project_root, settings)
    before_iris = set(before.terms)
    after_iris = set(after.terms)
    added_terms = tuple(after.terms[iri] for iri in sorted(after_iris - before_iris))
    removed_terms = tuple(before.terms[iri] for iri in sorted(before_iris - after_iris))
    kind_changes = tuple(
        OntologyTermKindChange(iri, before.terms[iri].kinds, after.terms[iri].kinds)
        for iri in sorted(before_iris & after_iris)
        if before.terms[iri].kinds != after.terms[iri].kinds
    )
    changed_term_semantics = _semantic_hash_changes(before.term_semantic_hashes, after.term_semantic_hashes)
    changed_shape_constraints = _semantic_hash_changes(
        before.node_shape_constraint_hashes,
        after.node_shape_constraint_hashes,
    )
    removed_shapes = tuple(sorted(before.node_shape_iris - after.node_shape_iris))
    removed_cqs = tuple(sorted(before.mandatory_cq_ids - after.mandatory_cq_ids))
    issues = _breaking_issues(
        settings,
        removed_terms,
        kind_changes,
        changed_term_semantics,
        removed_shapes,
        changed_shape_constraints,
        removed_cqs,
        after,
    )
    return OntologyEvolutionReport(
        checked_at=checked_at,
        before_project_root=str(before_project_root.resolve()),
        after_project_root=str(after_project_root.resolve()),
        policy_artifact_id=settings.artifact_id,
        policy_artifact_version=settings.artifact_version,
        overall_status="failed" if issues else "passed",
        added_terms=added_terms,
        removed_terms=removed_terms,
        kind_changes=kind_changes,
        changed_term_semantics=changed_term_semantics,
        added_node_shape_iris=tuple(sorted(after.node_shape_iris - before.node_shape_iris)),
        removed_node_shape_iris=removed_shapes,
        changed_node_shape_constraints=changed_shape_constraints,
        added_mandatory_cq_ids=tuple(sorted(after.mandatory_cq_ids - before.mandatory_cq_ids)),
        removed_mandatory_cq_ids=removed_cqs,
        newly_deprecated_term_iris=tuple(sorted(after.deprecated_iris - before.deprecated_iris)),
        replacements=after.replacements,
        issues=issues,
    )


def _project_snapshot(project_root: Path, settings: OntologyEvolutionSettings) -> _ProjectSnapshot:
    """Load only the formal artifacts that define the compared public contract."""
    paths = OntologyPaths(project_root)
    modules = load_module_policy(project_root, paths.policy)
    graph = merge_graphs(tuple(module.path for module in modules))
    namespaces = tuple(module.namespace for module in modules)
    deprecated_iris = _deprecated_iris(graph, namespaces, settings.deprecation_predicate_iri)
    terms = _term_snapshots(graph, namespaces, deprecated_iris)
    shapes_graph = load_graph(paths.shapes)
    annotation_predicates = _annotation_predicates(graph)
    node_shape_iris = frozenset(
        str(item) for item in shapes_graph.subjects(RDF.type, _SH.NodeShape) if isinstance(item, URIRef)
    )
    competency_questions = load_yaml(paths.competency_questions)
    return _ProjectSnapshot(
        terms=terms,
        term_semantic_hashes={
            iri: _semantic_closure_sha256(graph, URIRef(iri), annotation_predicates) for iri in terms
        },
        node_shape_iris=node_shape_iris,
        node_shape_constraint_hashes={
            iri: _semantic_closure_sha256(shapes_graph, URIRef(iri), annotation_predicates) for iri in node_shape_iris
        },
        mandatory_cq_ids=frozenset(item["id"] for item in competency_questions["competency_questions"]),
        deprecated_iris=deprecated_iris,
        replacements=_replacement_links(graph, namespaces, settings.replacement_predicate_iri),
    )


def _term_snapshots(
    graph: Graph,
    namespaces: tuple[str, ...],
    deprecated_iris: frozenset[str],
) -> dict[str, OntologyTermSnapshot]:
    """Collect local OWL class and property declarations without text-level diff noise."""
    kinds_by_iri: dict[str, set[TermKind]] = {}
    for rdf_type, kind in _TERM_TYPES:
        for subject in graph.subjects(RDF.type, rdf_type):
            if isinstance(subject, URIRef) and _is_local(subject, namespaces):
                kinds_by_iri.setdefault(str(subject), set()).add(kind)
    return {
        iri: OntologyTermSnapshot(
            iri=iri,
            kinds=tuple(sorted(kinds)),
            deprecated=iri in deprecated_iris,
        )
        for iri, kinds in kinds_by_iri.items()
    }


def _deprecated_iris(graph: Graph, namespaces: tuple[str, ...], predicate_iri: str) -> frozenset[str]:
    """Return local IRIs explicitly marked with a true deprecation value."""
    predicate = URIRef(predicate_iri)
    return frozenset(
        str(subject)
        for subject, value in graph.subject_objects(predicate)
        if isinstance(subject, URIRef) and _is_local(subject, namespaces) and _literal_is_true(value)
    )


def _replacement_links(
    graph: Graph,
    namespaces: tuple[str, ...],
    predicate_iri: str,
) -> tuple[OntologyReplacement, ...]:
    """Collect explicit replacement links from local ontology terms."""
    predicate = URIRef(predicate_iri)
    return tuple(
        sorted(
            (
                OntologyReplacement(str(source), str(replacement))
                for source, replacement in graph.subject_objects(predicate)
                if isinstance(source, URIRef) and isinstance(replacement, URIRef) and _is_local(source, namespaces)
            ),
            key=lambda item: (item.source_iri, item.replacement_iri),
        )
    )


def _breaking_issues(
    settings: OntologyEvolutionSettings,
    removed_terms: tuple[OntologyTermSnapshot, ...],
    kind_changes: tuple[OntologyTermKindChange, ...],
    changed_term_semantics: tuple[OntologySemanticHashChange, ...],
    removed_shapes: tuple[str, ...],
    changed_shape_constraints: tuple[OntologySemanticHashChange, ...],
    removed_cqs: tuple[str, ...],
    after: _ProjectSnapshot,
) -> tuple[OntologyEvolutionIssue, ...]:
    """Apply only the breaking checks declared by the versioned policy."""
    issues = []
    if "removed_active_term" in settings.breaking_checks:
        issues.extend(
            OntologyEvolutionIssue(
                "removed_active_term",
                term.iri,
                f"Removed active ontology term declared as {', '.join(term.kinds)}.",
            )
            for term in removed_terms
            if not term.deprecated
        )
    if "changed_term_kind" in settings.breaking_checks:
        issues.extend(
            OntologyEvolutionIssue(
                "changed_term_kind",
                change.iri,
                f"Changed term kind from {', '.join(change.before_kinds)} to {', '.join(change.after_kinds)}.",
            )
            for change in kind_changes
        )
    if "changed_active_term_semantics" in settings.breaking_checks:
        issues.extend(
            OntologyEvolutionIssue(
                "changed_active_term_semantics",
                change.iri,
                "Changed OWL axioms for a retained active ontology term.",
            )
            for change in changed_term_semantics
            if change.iri not in after.deprecated_iris
        )
    if "removed_node_shape" in settings.breaking_checks:
        issues.extend(
            OntologyEvolutionIssue("removed_node_shape", iri, "Removed a named SHACL node shape.")
            for iri in removed_shapes
        )
    if "changed_node_shape_constraints" in settings.breaking_checks:
        issues.extend(
            OntologyEvolutionIssue(
                "changed_node_shape_constraints",
                change.iri,
                "Changed constraints for a retained named SHACL node shape.",
            )
            for change in changed_shape_constraints
        )
    if "removed_mandatory_cq" in settings.breaking_checks:
        issues.extend(
            OntologyEvolutionIssue("removed_mandatory_cq", cq_id, "Removed a competency question.")
            for cq_id in removed_cqs
        )
    if settings.require_retained_declaration:
        issues.extend(
            OntologyEvolutionIssue(
                "removed_deprecated_term",
                term.iri,
                "Removed a previously deprecated ontology term instead of retaining its declaration.",
            )
            for term in removed_terms
            if term.deprecated
        )
        issues.extend(
            OntologyEvolutionIssue(
                "deprecated_term_without_declaration",
                iri,
                "Deprecated local term no longer has a class or property declaration.",
            )
            for iri in sorted(after.deprecated_iris - set(after.terms))
        )
    return tuple(sorted(issues, key=lambda item: (item.check_id, item.subject)))


def _semantic_hash_changes(before: dict[str, str], after: dict[str, str]) -> tuple[OntologySemanticHashChange, ...]:
    """Return stable hash changes for IRIs retained across both snapshots."""
    return tuple(
        OntologySemanticHashChange(iri, before[iri], after[iri])
        for iri in sorted(set(before) & set(after))
        if before[iri] != after[iri]
    )


def _annotation_predicates(graph: Graph) -> frozenset[URIRef]:
    """Collect documentation and traceability predicates excluded from semantic comparisons."""
    declared = {item for item in graph.subjects(RDF.type, OWL.AnnotationProperty) if isinstance(item, URIRef)}
    return frozenset(
        {
            RDF.type,
            RDFS.label,
            RDFS.comment,
            DCTERMS.title,
            DCTERMS.description,
            DCTERMS.created,
            DCTERMS.modified,
            SKOS.prefLabel,
            SKOS.altLabel,
            SKOS.definition,
            *declared,
        }
    )


def _semantic_closure_sha256(graph: Graph, root: URIRef, ignored_predicates: frozenset[URIRef]) -> str:
    """Hash a named resource and the blank-node constraint or axiom closure reachable from it."""
    closure = Graph()
    pending: list[URIRef | BNode] = [root]
    visited: set[URIRef | BNode] = set()
    while pending:
        subject = pending.pop()
        if subject in visited:
            continue
        visited.add(subject)
        for predicate, value in graph.predicate_objects(subject):
            if predicate in ignored_predicates:
                continue
            closure.add((subject, predicate, value))
            if isinstance(value, BNode):
                pending.append(value)
    canonical = to_canonical_graph(closure)
    lines = sorted(" ".join(term.n3() for term in triple) + " ." for triple in canonical)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _is_local(value: URIRef, namespaces: tuple[str, ...]) -> bool:
    """Return whether an IRI belongs to one declared local ontology namespace."""
    return any(str(value).startswith(namespace) for namespace in namespaces)


def _literal_is_true(value: object) -> bool:
    """Recognize the RDF boolean true value without interpreting arbitrary text."""
    return isinstance(value, Literal) and value.toPython() is True
