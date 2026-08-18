"""Tests for semantic-QA orchestration and inferred graph audits."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from legal_rag.ontology_validation.paths import OntologyPaths
from legal_rag.ontology_validation.semantic_config import SemanticQaSettings
from legal_rag.ontology_validation.semantic_qa import (
    collect_traceability_issues,
    run_semantic_qa,
    semantic_input_sha256,
)

from .conftest import FakeRobotRunner

CHECKED_AT = datetime(2026, 8, 18, 10, 0, tzinfo=UTC)


def test_all_checks_pass_with_clean_typed_tool_results(
    tmp_path: Path,
    ontology_paths: OntologyPaths,
    semantic_settings: SemanticQaSettings,
) -> None:
    """A clean profile, reasoner graph, quality report, and traceability audit produce a passing report."""
    runner = FakeRobotRunner(semantic_settings)
    report = run_semantic_qa(ontology_paths, semantic_settings, runner, tmp_path, CHECKED_AT)

    assert report.overall_status == "passed"
    assert all(item.status == "passed" for item in report.checks)
    assert report.unsatisfiable_class_iris == ()
    assert report.unexpected_equivalences == ()
    assert report.traceability_issues == ()
    assert report.input_sha256 == semantic_input_sha256(ontology_paths, semantic_settings)
    assert [call[0] for call in runner.calls] == ["validate-profile", "reason", "report"]


def test_profile_failure_blocks_reasoning_but_still_runs_independent_gates(
    tmp_path: Path,
    ontology_paths: OntologyPaths,
    semantic_settings: SemanticQaSettings,
) -> None:
    """Profile non-conformance blocks dependent reasoning without hiding quality or traceability results."""
    runner = FakeRobotRunner(semantic_settings, profile_exit=1)
    report = run_semantic_qa(ontology_paths, semantic_settings, runner, tmp_path, CHECKED_AT)
    checks = {item.check_id: item for item in report.checks}

    assert report.overall_status == "failed"
    assert checks["owl-2-dl-profile"].status == "failed"
    assert checks["dl-consistency"].status == "blocked"
    assert checks["unsatisfiable-classes"].status == "blocked"
    assert [call[0] for call in runner.calls] == ["validate-profile", "report"]


def test_reasoned_graph_exposes_unsatisfiable_classes_and_unexpected_equivalence(
    tmp_path: Path,
    ontology_paths: OntologyPaths,
    semantic_settings: SemanticQaSettings,
) -> None:
    """Named local classes inferred impossible or equivalent fail explicit semantic audits."""
    reasoned = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix loroc: <https://w3id.org/legal-ontology-rag/ontology/occurrence-context#> .
loroc:Impossible a owl:Class ; owl:equivalentClass owl:Nothing .
loroc:Left a owl:Class ; owl:equivalentClass loroc:Right .
loroc:Right a owl:Class .
"""
    report = run_semantic_qa(
        ontology_paths,
        semantic_settings,
        FakeRobotRunner(semantic_settings, reasoned_turtle=reasoned),
        tmp_path,
        CHECKED_AT,
    )
    checks = {item.check_id: item for item in report.checks}

    assert report.overall_status == "failed"
    assert report.unsatisfiable_class_iris == (
        "https://w3id.org/legal-ontology-rag/ontology/occurrence-context#Impossible",
    )
    assert report.unexpected_equivalences == (
        (
            "https://w3id.org/legal-ontology-rag/ontology/occurrence-context#Left",
            "https://w3id.org/legal-ontology-rag/ontology/occurrence-context#Right",
        ),
    )
    assert checks["unsatisfiable-classes"].status == "failed"
    assert checks["unexpected-equivalences"].status == "failed"


def test_current_formal_artifacts_have_complete_structural_traceability(
    ontology_paths: OntologyPaths,
    semantic_settings: SemanticQaSettings,
) -> None:
    """The report-level audit agrees with the existing traceability regression suite."""
    assert collect_traceability_issues(ontology_paths, semantic_settings) == ()
