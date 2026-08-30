"""Tests for the versioned external ontology reuse registry."""

from __future__ import annotations

from pathlib import Path

from legal_rag.ontology_validation.loader import load_yaml
from legal_rag.ontology_validation.reuse_config import ExternalReuseSettings


def test_registry_contains_only_formally_used_external_resources(
    project_root: Path,
    reuse_settings: ExternalReuseSettings,
) -> None:
    """The registry excludes research-only resources from machine validation."""
    assert reuse_settings.artifact_version == "0.2.0"
    assert reuse_settings.project_owner_approval == "approved"
    assert len(reuse_settings.formal_paths) == 7
    assert len(reuse_settings.resources) == 4
    assert {resource.decision_id for resource in reuse_settings.resources} == {
        "REUSE-DCTERMS-001",
        "REUSE-PROV-001",
        "REUSE-SKOS-001",
        "REUSE-TIME-001",
    }
    scripts = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'legal-review-ontology-reuse = "legal_rag.ontology_reuse_review_cli:main"' in scripts
    review = load_yaml(project_root / "ontology" / "external_reuse_registry.yaml")["review"]
    assert review == {"project_owner_approval": "approved"}


def test_registry_lists_exactly_twenty_three_active_external_terms(
    reuse_settings: ExternalReuseSettings,
) -> None:
    """Every external semantic IRI has one explicit reviewed term-level record."""
    terms = {
        term.iri
        for resource in reuse_settings.resources
        if resource.formal_use == "active_terms"
        for term in resource.terms
    }
    assert len(terms) == 23
    assert "http://www.w3.org/2004/02/skos/core#broader" in terms
    assert "http://www.w3.org/2006/time#inXSDDateTimeStamp" in terms
