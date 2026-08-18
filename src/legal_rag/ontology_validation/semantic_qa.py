"""Run OWL profile, DL reasoning, quality, and traceability release gates."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from rdflib import OWL, RDFS, Graph, URIRef

from legal_rag.ontology_validation.loader import (
    load_graph,
    load_module_policy,
    load_query_cases,
    load_validation_cases,
    load_yaml,
    merge_graphs,
)
from legal_rag.ontology_validation.paths import OntologyPaths
from legal_rag.ontology_validation.semantic_config import SemanticQaSettings
from legal_rag.ontology_validation.semantic_models import (
    ProcessResult,
    SemanticCheckResult,
    SemanticQaReport,
)
from legal_rag.ontology_validation.semantic_runner import RobotRunner
from legal_rag.ontology_validation.traceability import (
    REFERENCE_PREDICATES,
    collect_ids,
    referenced_ids,
    shapes_without_traceability,
    subjects_for_cq,
    terms_without_traceability,
)


def run_semantic_qa(
    paths: OntologyPaths,
    settings: SemanticQaSettings,
    runner: RobotRunner,
    output_directory: Path,
    checked_at: datetime,
) -> SemanticQaReport:
    """Run all semantic release gates and return a versioned report."""
    output_directory.mkdir(parents=True, exist_ok=True)
    profile_result = runner.run(_profile_arguments(paths.project_root, settings), output_directory)
    checks = [_process_check("owl-2-dl-profile", profile_result, settings.profile_report_file)]

    unsatisfiable_classes: tuple[str, ...] = ()
    unexpected_equivalences: tuple[tuple[str, str], ...] = ()
    if profile_result.exit_code == 0:
        reason_result = runner.run(_reason_arguments(paths.project_root, settings), output_directory)
        checks.append(_process_check("dl-consistency", reason_result, settings.reasoned_ontology_file))
        if reason_result.exit_code == 0:
            reasoned_graph = load_graph(output_directory / settings.reasoned_ontology_file)
            asserted_graph = _ontology_graph(paths)
            namespaces = tuple(module.namespace for module in load_module_policy(paths.project_root, paths.policy))
            unsatisfiable_classes = _unsatisfiable_classes(reasoned_graph, namespaces)
            unexpected_equivalences = _unexpected_equivalences(reasoned_graph, asserted_graph, namespaces)
            checks.extend(
                (
                    _audit_check(
                        "unsatisfiable-classes",
                        not settings.fail_on_unsatisfiable_classes or not unsatisfiable_classes,
                        f"Found {len(unsatisfiable_classes)} unintended unsatisfiable local classes.",
                    ),
                    _audit_check(
                        "unexpected-equivalences",
                        not settings.fail_on_unexpected_equivalences or not unexpected_equivalences,
                        f"Found {len(unexpected_equivalences)} unintended inferred local class equivalences.",
                    ),
                )
            )
        else:
            checks.extend(
                (
                    _blocked_check("unsatisfiable-classes", "Blocked because DL reasoning failed."),
                    _blocked_check("unexpected-equivalences", "Blocked because DL reasoning failed."),
                )
            )
    else:
        checks.extend(
            (
                _blocked_check("dl-consistency", "Blocked because the import closure is outside OWL 2 DL."),
                _blocked_check("unsatisfiable-classes", "Blocked because the import closure is outside OWL 2 DL."),
                _blocked_check("unexpected-equivalences", "Blocked because the import closure is outside OWL 2 DL."),
            )
        )

    quality_result = runner.run(_quality_arguments(paths.project_root, settings), output_directory)
    checks.append(_process_check("robot-quality", quality_result, settings.quality_report_file))
    traceability_issues = collect_traceability_issues(paths, settings)
    checks.append(
        _audit_check(
            "traceability",
            not traceability_issues,
            f"Found {len(traceability_issues)} unresolved traceability or mandatory-CQ coverage issues.",
        )
    )
    overall_status = "passed" if all(item.status == "passed" for item in checks) else "failed"
    return SemanticQaReport(
        checked_at=checked_at,
        ontology_path=str(settings.ontology_path.relative_to(paths.project_root).as_posix()),
        catalog_path=str(settings.catalog_path.relative_to(paths.project_root).as_posix()),
        input_sha256=semantic_input_sha256(paths, settings),
        tool_image=settings.image,
        profile=settings.profile,
        reasoner=settings.reasoner,
        overall_status=overall_status,
        checks=tuple(checks),
        unsatisfiable_class_iris=unsatisfiable_classes,
        unexpected_equivalences=unexpected_equivalences,
        traceability_issues=traceability_issues,
    )


def semantic_input_sha256(paths: OntologyPaths, settings: SemanticQaSettings) -> str:
    """Hash every versioned file that can affect the semantic-QA result."""
    modules = load_module_policy(paths.project_root, paths.policy)
    query_cases = load_query_cases(paths.project_root, paths.query_manifest)
    validation_cases = load_validation_cases(paths.project_root, paths.query_manifest)
    input_paths = {
        settings.ontology_path,
        settings.catalog_path,
        paths.policy,
        paths.shapes,
        paths.competency_questions,
        paths.requirements,
        paths.glossary,
        paths.modeling_decisions,
        paths.query_manifest,
        paths.semantic_qa_contract,
        *(module.path for module in modules),
        *(case.query_path for case in query_cases),
        *(case.fixture_path for case in query_cases),
        *(case.fixture_path for case in validation_cases),
    }
    digest = hashlib.sha256()
    for path in sorted(input_paths):
        relative_path = path.resolve().relative_to(paths.project_root.resolve()).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def collect_traceability_issues(
    paths: OntologyPaths,
    settings: SemanticQaSettings,
) -> tuple[str, ...]:
    """Return stable release-report issues across all traceability boundaries."""
    modules = load_module_policy(paths.project_root, paths.policy)
    ontology_graph = merge_graphs(tuple(module.path for module in modules))
    shapes_graph = load_graph(paths.shapes)
    combined = ontology_graph + shapes_graph
    issues = []
    if settings.require_resolved_references:
        registries = _traceability_registries(paths)
        for kind, predicate in REFERENCE_PREDICATES.items():
            issues.extend(
                f"unresolved-{kind}:{identifier}"
                for identifier in sorted(referenced_ids(combined, predicate) - registries[kind])
            )
    if settings.require_local_term_traceability:
        namespaces = tuple(module.namespace for module in modules)
        issues.extend(
            f"untraced-term:{term}" for term in sorted(terms_without_traceability(ontology_graph, namespaces))
        )
    if settings.require_named_shape_traceability:
        issues.extend(f"untraced-shape:{shape}" for shape in sorted(shapes_without_traceability(shapes_graph)))
    if settings.require_mandatory_cq_coverage:
        issues.extend(_mandatory_cq_issues(paths, ontology_graph, shapes_graph))
    return tuple(sorted(issues))


def _profile_arguments(project_root: Path, settings: SemanticQaSettings) -> tuple[str, ...]:
    """Build the pinned OWL profile-validation command arguments."""
    return (
        "validate-profile",
        "--input",
        _container_path(project_root, settings.ontology_path),
        "--catalog",
        _container_path(project_root, settings.catalog_path),
        "--profile",
        settings.profile,
        "--output",
        f"/out/{settings.profile_report_file}",
    )


def _reason_arguments(project_root: Path, settings: SemanticQaSettings) -> tuple[str, ...]:
    """Build the pinned DL-reasoning command arguments."""
    return (
        "reason",
        "--input",
        _container_path(project_root, settings.ontology_path),
        "--catalog",
        _container_path(project_root, settings.catalog_path),
        "--reasoner",
        settings.reasoner,
        "--equivalent-classes-allowed",
        settings.equivalent_classes_allowed,
        "--axiom-generators",
        settings.axiom_generators,
        "--output",
        f"/out/{settings.reasoned_ontology_file}",
    )


def _quality_arguments(project_root: Path, settings: SemanticQaSettings) -> tuple[str, ...]:
    """Build the pinned ROBOT quality-report command arguments."""
    return (
        "report",
        "--input",
        _container_path(project_root, settings.ontology_path),
        "--catalog",
        _container_path(project_root, settings.catalog_path),
        "--format",
        settings.quality_format,
        "--fail-on",
        settings.quality_fail_on,
        "--output",
        f"/out/{settings.quality_report_file}",
    )


def _container_path(project_root: Path, path: Path) -> str:
    """Translate one repository path to its read-only container mount."""
    return f"/work/{path.resolve().relative_to(project_root.resolve()).as_posix()}"


def _process_check(check_id: str, result: ProcessResult, artifact: str) -> SemanticCheckResult:
    """Translate one external process result to a bounded check summary."""
    status = "passed" if result.exit_code == 0 else "failed"
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    summary = "Command completed successfully." if status == "passed" else output[-2000:]
    return SemanticCheckResult(
        check_id=check_id,
        status=status,
        summary=summary,
        artifact=artifact,
        exit_code=result.exit_code,
    )


def _audit_check(check_id: str, passed: bool, failed_summary: str) -> SemanticCheckResult:
    """Build one internal graph-audit result."""
    return SemanticCheckResult(
        check_id=check_id,
        status="passed" if passed else "failed",
        summary="No issues found." if passed else failed_summary,
        artifact=None,
        exit_code=None,
    )


def _blocked_check(check_id: str, summary: str) -> SemanticCheckResult:
    """Build one dependency-blocked semantic check result."""
    return SemanticCheckResult(check_id=check_id, status="blocked", summary=summary, artifact=None, exit_code=None)


def _ontology_graph(paths: OntologyPaths) -> Graph:
    """Return the asserted local ontology import closure."""
    modules = load_module_policy(paths.project_root, paths.policy)
    return merge_graphs(tuple(module.path for module in modules))


def _unsatisfiable_classes(reasoned_graph: Graph, namespaces: Sequence[str]) -> tuple[str, ...]:
    """Return named local classes inferred equivalent or subordinate to owl:Nothing."""
    classes = {
        subject
        for subject in reasoned_graph.subjects(OWL.equivalentClass, OWL.Nothing)
        if _is_local_named_class(subject, namespaces)
    }
    classes.update(
        subject
        for subject in reasoned_graph.subjects(RDFS.subClassOf, OWL.Nothing)
        if _is_local_named_class(subject, namespaces)
    )
    return tuple(sorted(str(item) for item in classes))


def _unexpected_equivalences(
    reasoned_graph: Graph,
    asserted_graph: Graph,
    namespaces: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """Return inferred equivalences between distinct local named classes."""
    pairs = set()
    for left, right in reasoned_graph.subject_objects(OWL.equivalentClass):
        if not _is_local_named_class(left, namespaces) or not _is_local_named_class(right, namespaces) or left == right:
            continue
        if (left, OWL.equivalentClass, right) in asserted_graph or (right, OWL.equivalentClass, left) in asserted_graph:
            continue
        pairs.add(tuple(sorted((str(left), str(right)))))
    return tuple(sorted(pairs))


def _is_local_named_class(value: object, namespaces: Sequence[str]) -> bool:
    """Return whether a value is a named resource in one local ontology namespace."""
    return (
        isinstance(value, URIRef) and value != OWL.Nothing and any(str(value).startswith(item) for item in namespaces)
    )


def _traceability_registries(paths: OntologyPaths) -> Mapping[str, frozenset[str]]:
    """Load stable identifiers from every versioned traceability registry."""
    return {
        "requirement": collect_ids(load_yaml(paths.requirements)),
        "cq": collect_ids(load_yaml(paths.competency_questions)),
        "glossary": collect_ids(load_yaml(paths.glossary)),
        "decision": collect_ids(load_yaml(paths.modeling_decisions)),
    }


def _mandatory_cq_issues(paths: OntologyPaths, ontology_graph: Graph, shapes_graph: Graph) -> tuple[str, ...]:
    """Return mandatory CQs missing formal or executable coverage."""
    document = load_yaml(paths.competency_questions)
    mandatory_cqs = {item["id"] for item in document["competency_questions"] if item["priority"] == "must"}
    query_cases = {item.cq_id: item for item in load_query_cases(paths.project_root, paths.query_manifest)}
    issues = []
    for cq_id in sorted(mandatory_cqs):
        if not subjects_for_cq((ontology_graph, shapes_graph), cq_id):
            issues.append(f"cq-without-formal-coverage:{cq_id}")
        case = query_cases.get(cq_id)
        if case is None or not case.query_path.is_file() or not case.fixture_path.is_file():
            issues.append(f"cq-without-executable-coverage:{cq_id}")
    return tuple(issues)
