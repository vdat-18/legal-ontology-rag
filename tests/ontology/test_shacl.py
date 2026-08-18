"""Tests for closed-world ontology fixture constraints."""

from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Graph

from legal_rag.ontology_validation.loader import load_graph, load_validation_cases
from legal_rag.ontology_validation.models import ValidationCase
from legal_rag.ontology_validation.paths import OntologyPaths
from legal_rag.ontology_validation.shacl import run_shacl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = OntologyPaths(PROJECT_ROOT)
VALIDATION_CASES = load_validation_cases(PROJECT_ROOT, PATHS.query_manifest)


def validation_case_ids(case: ValidationCase) -> str:
    """Return readable pytest IDs for SHACL fixtures."""
    return case.fixture_path.stem


def test_validation_manifest_covers_positive_and_boundary_fixtures(ontology_paths: OntologyPaths) -> None:
    """Every committed fixture is represented once in the SHACL manifest."""
    cases = load_validation_cases(ontology_paths.project_root, ontology_paths.query_manifest)
    represented = [case.fixture_path.resolve() for case in cases]
    fixtures = {path.resolve() for path in (ontology_paths.ontology_dir / "tests" / "fixtures").glob("*.ttl")}

    assert len(represented) == len(set(represented))
    assert set(represented) == fixtures
    assert {case.expected_conforms for case in cases} == {True, False}


@pytest.mark.parametrize(
    "case",
    VALIDATION_CASES,
    ids=validation_case_ids,
)
def test_shacl_fixture_has_expected_conformance(
    case: ValidationCase,
    ontology_paths: OntologyPaths,
    ontology_graph: Graph,
) -> None:
    """Each positive or boundary graph produces its reviewed conformance result."""
    result = run_shacl(load_graph(case.fixture_path), load_graph(ontology_paths.shapes), ontology_graph)

    assert result.conforms is case.expected_conforms, result.report_text
