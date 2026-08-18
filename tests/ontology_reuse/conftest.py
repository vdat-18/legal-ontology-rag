"""Shared fixtures for external ontology reuse review tests."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from legal_rag.ontology_validation.reuse_config import ExternalReuseSettings


@pytest.fixture
def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def reuse_settings(project_root: Path) -> ExternalReuseSettings:
    """Load the production external-reuse registry."""
    return ExternalReuseSettings.from_path(project_root / "ontology" / "external_reuse_registry.yaml", project_root)


@pytest.fixture
def copied_project(
    tmp_path: Path,
    project_root: Path,
) -> Callable[[], tuple[Path, ExternalReuseSettings]]:
    """Copy versioned ontology artifacts for isolated boundary mutations."""

    def copy() -> tuple[Path, ExternalReuseSettings]:
        target = tmp_path / "project"
        shutil.copytree(project_root / "ontology", target / "ontology")
        settings = ExternalReuseSettings.from_path(
            target / "ontology" / "external_reuse_registry.yaml",
            target,
        )
        return target, settings

    return copy
