"""Shared fixtures for ontology_rag tests."""

from pathlib import Path

import pytest


@pytest.fixture
def test_fixture_path() -> Path:
    """Return path to ontology test fixtures."""
    return Path(__file__).parent.parent.parent / "ontology" / "tests" / "fixtures"
