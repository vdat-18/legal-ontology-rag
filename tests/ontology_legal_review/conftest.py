"""Shared fixtures for internal ontology legal review."""

from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag.ontology_validation.legal_review_config import LegalReviewSettings


@pytest.fixture
def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def legal_review_settings(project_root: Path) -> LegalReviewSettings:
    """Load the production internal legal-review artifacts."""
    return LegalReviewSettings.from_paths(
        project_root / "ontology" / "legal_review_contract.yaml",
        project_root / "ontology" / "legal_review_decisions.yaml",
        project_root,
    )
