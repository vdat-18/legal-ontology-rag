"""Tests for executable competency-question projections."""

from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag.ontology_validation.loader import load_query_cases, load_yaml, merge_graphs
from legal_rag.ontology_validation.models import ModulePolicy, QueryCase
from legal_rag.ontology_validation.paths import OntologyPaths
from legal_rag.ontology_validation.queries import run_query
from legal_rag.ontology_validation.traceability import collect_ids

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = OntologyPaths(PROJECT_ROOT)
QUERY_CASES = load_query_cases(PROJECT_ROOT, PATHS.query_manifest)


def test_manifest_covers_every_mandatory_cq_and_query_file(ontology_paths: OntologyPaths) -> None:
    """Every mandatory CQ has one query and every query file is registered."""
    cases = load_query_cases(ontology_paths.project_root, ontology_paths.query_manifest)
    cq_document = load_yaml(ontology_paths.competency_questions)
    mandatory_cqs = {item["id"] for item in cq_document["competency_questions"]}
    represented_cqs = [case.cq_id for case in cases]
    query_files = set((ontology_paths.ontology_dir / "tests" / "cq_queries").glob("*.rq"))

    assert len(represented_cqs) == len(set(represented_cqs))
    assert set(represented_cqs) == mandatory_cqs
    assert {case.query_path for case in cases} == query_files
    assert not load_yaml(ontology_paths.query_manifest)["deferred_queries"]


def test_deferred_output_fields_are_explicit_projection_boundaries(ontology_paths: OntologyPaths) -> None:
    """Deferred semantic outputs remain explicit and outside structural query columns."""
    cases = load_query_cases(ontology_paths.project_root, ontology_paths.query_manifest)

    for case in cases:
        assert len(case.deferred_output_fields) == len(set(case.deferred_output_fields))
        assert set(case.deferred_output_fields).isdisjoint(case.expected_columns)


@pytest.mark.parametrize("case", QUERY_CASES, ids=lambda case: case.cq_id)
def test_cq_projection_returns_expected_shape_and_rows(
    case: QueryCase,
    module_policy: tuple[ModulePolicy, ...],
) -> None:
    """Each CQ query returns its versioned columns and reviewed fixture cardinality."""
    graph = merge_graphs((case.fixture_path, *(module.path for module in module_policy)))
    result = run_query(graph, case.query_path)

    assert result.columns == case.expected_columns
    if case.expected_rows is not None:
        assert len(result.rows) == case.expected_rows
    if case.expected_minimum_rows is not None:
        assert len(result.rows) >= case.expected_minimum_rows


def test_cq_registry_contains_only_stable_unique_ids(ontology_paths: OntologyPaths) -> None:
    """The CQ registry exposes unique identifiers consumed by QA artifacts."""
    document = load_yaml(ontology_paths.competency_questions)
    identifiers = [item["id"] for item in document["competency_questions"]]

    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) <= collect_ids(document)


def test_cq_registry_uses_minimal_schema(ontology_paths: OntologyPaths) -> None:
    """CQs retain only identity, question, expected outputs, and requirement links."""
    document = load_yaml(ontology_paths.competency_questions)

    assert set(document) == {
        "schema_version",
        "artifact_id",
        "artifact_version",
        "language",
        "competency_questions",
    }
    assert all(
        set(item) == {"id", "question_vi", "expected_answer_fields", "requirement_ids"}
        for item in document["competency_questions"]
    )
