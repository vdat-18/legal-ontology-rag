"""Tests for the versioned external ontology reuse registry."""

from __future__ import annotations

from pathlib import Path

from legal_rag.ontology_validation.reuse_config import ExternalReuseSettings


def test_registry_pins_review_scope_decisions_and_human_approval(
    project_root: Path,
    reuse_settings: ExternalReuseSettings,
) -> None:
    """The registry records exact technical decisions without granting owner acceptance."""
    assert reuse_settings.artifact_version == "0.1.0"
    assert reuse_settings.project_owner_approval == "pending"
    assert len(reuse_settings.formal_paths) == 7
    assert len(reuse_settings.resources) == 10
    assert {resource.decision_id for resource in reuse_settings.resources} == {
        "REUSE-DCTERMS-001",
        "REUSE-PROV-001",
        "REUSE-SKOS-001",
        "REUSE-TIME-001",
        "REUSE-ELI-001",
        "REUSE-TLKB-001",
        "REUSE-LKIF-001",
        "REUSE-AKN-001",
        "REUSE-LRML-001",
        "REUSE-ODP-001",
    }
    scripts = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'legal-review-ontology-reuse = "legal_rag.ontology_reuse_review_cli:main"' in scripts


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
