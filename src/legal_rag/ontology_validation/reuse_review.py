"""Validate external term reuse, boundaries, licenses, and policy consistency."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rdflib import OWL, RDF, RDFS, Graph, Namespace, URIRef

from legal_rag.ontology_validation.loader import load_graph, load_yaml, merge_graphs
from legal_rag.ontology_validation.reuse_config import ExternalReuseSettings
from legal_rag.ontology_validation.reuse_models import ExternalReuseReport, ReuseCheckResult

_LOCAL_NAMESPACE = "https://w3id.org/legal-ontology-rag/"
_SH = Namespace("http://www.w3.org/ns/shacl#")
_DECLARATION_TYPES = (
    OWL.AnnotationProperty,
    OWL.Class,
    OWL.DatatypeProperty,
    OWL.ObjectProperty,
)


def run_external_reuse_review(
    project_root: Path,
    settings: ExternalReuseSettings,
    checked_at: datetime,
) -> ExternalReuseReport:
    """Run every technical external-reuse gate without granting human approval."""
    graph = merge_graphs(settings.formal_paths)
    active_terms = _active_external_terms(graph, settings)
    checks = (
        _policy_check(project_root, settings),
        _term_inventory_check(settings, active_terms),
        _declaration_check(project_root, settings),
        _non_active_namespace_check(settings, active_terms),
        _external_import_check(graph),
        _review_evidence_check(settings),
    )
    overall_status = "passed" if all(check.status == "passed" for check in checks) else "failed"
    return ExternalReuseReport(
        checked_at=checked_at,
        artifact_id=settings.artifact_id,
        artifact_version=settings.artifact_version,
        overall_status=overall_status,
        project_owner_approval=settings.project_owner_approval,
        checks=checks,
        active_external_terms=active_terms,
    )


def collect_external_reuse_issues(project_root: Path, settings: ExternalReuseSettings) -> tuple[str, ...]:
    """Return stable release-gate issues without granting project-owner approval."""
    report = run_external_reuse_review(project_root, settings, datetime.min.replace(tzinfo=UTC))
    return tuple(sorted(issue for check in report.checks for issue in check.issues))


def _policy_check(project_root: Path, settings: ExternalReuseSettings) -> ReuseCheckResult:
    """Require the formalization policy and registry to contain identical decisions."""
    policy_items = load_yaml(project_root / "ontology" / "formalization_policy.yaml")["external_dependency_policy"]
    policy = {item["id"]: item for item in policy_items}
    registry = {item.decision_id: item for item in settings.resources}
    issues = [f"policy-only-decision:{item}" for item in sorted(policy.keys() - registry.keys())]
    issues.extend(f"registry-only-decision:{item}" for item in sorted(registry.keys() - policy.keys()))
    for decision_id in sorted(policy.keys() & registry.keys()):
        resource = registry[decision_id]
        if policy[decision_id]["mode"] != resource.policy_mode:
            issues.append(f"policy-mode-mismatch:{decision_id}")
        if policy[decision_id]["owl_import"] != resource.owl_import:
            issues.append(f"policy-import-mismatch:{decision_id}")
    return _check("policy-registry-consistency", issues)


def _term_inventory_check(settings: ExternalReuseSettings, active_terms: tuple[str, ...]) -> ReuseCheckResult:
    """Require exact term-level coverage for every active external semantic IRI."""
    listed_terms = {
        term.iri for resource in settings.resources if resource.formal_use == "active_terms" for term in resource.terms
    }
    active = set(active_terms)
    issues = [f"unregistered-external-term:{item}" for item in sorted(active - listed_terms)]
    issues.extend(f"listed-term-not-used:{item}" for item in sorted(listed_terms - active))
    return _check("active-term-inventory", issues)


def _declaration_check(project_root: Path, settings: ExternalReuseSettings) -> ReuseCheckResult:
    """Require every registry-marked OWL-closure declaration in the foundational module."""
    required_terms = tuple(
        term for resource in settings.resources for term in resource.terms if term.declaration_in_owl_closure
    )
    if not required_terms:
        return _check("owl-closure-declarations", [])
    graph = load_graph(project_root / "ontology" / "modules" / "source_provenance.ttl")
    issues = [
        f"missing-foundational-declaration:{term.iri}"
        for term in required_terms
        if not any((URIRef(term.iri), RDF.type, declaration_type) in graph for declaration_type in _DECLARATION_TYPES)
    ]
    return _check("owl-closure-declarations", issues)


def _non_active_namespace_check(
    settings: ExternalReuseSettings,
    active_terms: tuple[str, ...],
) -> ReuseCheckResult:
    """Reject formal term use from candidate, reference-only, or no-copy resources."""
    issues = []
    for resource in settings.resources:
        if resource.formal_use == "active_terms" or resource.namespace is None:
            continue
        issues.extend(
            f"non-active-namespace-used:{resource.decision_id}:{term}"
            for term in active_terms
            if term.startswith(resource.namespace)
        )
    return _check("non-active-resource-boundaries", issues)


def _external_import_check(graph: Graph) -> ReuseCheckResult:
    """Reject every external OWL import in the reviewed V0.1 formal graph."""
    issues = [
        f"external-owl-import:{item}"
        for item in sorted(
            str(value) for value in graph.objects(None, OWL.imports) if not str(value).startswith(_LOCAL_NAMESPACE)
        )
    ]
    return _check("external-owl-imports", issues)


def _review_evidence_check(settings: ExternalReuseSettings) -> ReuseCheckResult:
    """Require complete local review evidence for every resource decision."""
    issues = []
    for resource in settings.resources:
        fields = (
            resource.authority,
            resource.source_version,
            resource.source_url,
            resource.license,
            resource.attribution,
            resource.compatibility,
            *resource.boundaries,
        )
        if not all(fields):
            issues.append(f"incomplete-review-evidence:{resource.decision_id}")
        if resource.review_status != "ontology_reviewed":
            issues.append(f"reuse-review-pending:{resource.decision_id}")
        if resource.owl_import:
            issues.append(f"unapproved-import-policy:{resource.decision_id}")
    return _check("review-evidence", issues)


def _active_external_terms(graph: Graph, settings: ExternalReuseSettings) -> tuple[str, ...]:
    """Return external semantic IRIs used as predicates, types, hierarchy targets, or declarations."""
    semantic_iris = {value for value in graph.predicates() if isinstance(value, URIRef)}
    for predicate in (RDF.type, RDFS.subClassOf, RDFS.subPropertyOf, _SH["class"], _SH["path"]):
        semantic_iris.update(value for value in graph.objects(None, predicate) if isinstance(value, URIRef))
    semantic_iris.update(
        subject
        for declaration_type in _DECLARATION_TYPES
        for subject in graph.subjects(RDF.type, declaration_type)
        if isinstance(subject, URIRef)
    )
    return tuple(
        sorted(
            str(value)
            for value in semantic_iris
            if not str(value).startswith(_LOCAL_NAMESPACE)
            and not any(str(value).startswith(namespace) for namespace in settings.foundational_namespaces)
        )
    )


def _check(check_id: str, issues: list[str]) -> ReuseCheckResult:
    """Build one stable reuse-review check result."""
    issues = tuple(sorted(issues))
    return ReuseCheckResult(check_id, "failed" if issues else "passed", issues)
