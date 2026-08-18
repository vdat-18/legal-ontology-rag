"""Shared fixtures for ontology QA tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Graph

from legal_rag.ontology_validation.loader import load_module_policy, merge_graphs
from legal_rag.ontology_validation.models import ModulePolicy
from legal_rag.ontology_validation.paths import OntologyPaths

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def ontology_paths() -> OntologyPaths:
    """Return repository ontology paths."""
    return OntologyPaths(PROJECT_ROOT)


@pytest.fixture(scope="session")
def module_policy(ontology_paths: OntologyPaths) -> tuple[ModulePolicy, ...]:
    """Return the root and active ontology module policy."""
    return load_module_policy(PROJECT_ROOT, ontology_paths.policy)


@pytest.fixture(scope="session")
def ontology_graph(module_policy: tuple[ModulePolicy, ...]) -> Graph:
    """Return one graph containing the root and active modules."""
    return merge_graphs(tuple(module.path for module in module_policy))
