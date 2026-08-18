"""Shared fixtures for ontology namespace-publication tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag.ontology_validation.publication_config import NamespacePublicationSettings


@pytest.fixture
def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def publication_settings(project_root: Path) -> NamespacePublicationSettings:
    """Load the production namespace-publication contract."""
    return NamespacePublicationSettings.from_path(
        project_root / "ontology" / "namespace_publication.yaml",
        project_root,
    )
