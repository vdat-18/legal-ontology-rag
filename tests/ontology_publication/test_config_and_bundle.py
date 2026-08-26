"""Tests for the namespace contract and W3ID pull-request bundle."""

from __future__ import annotations

import tomllib
from pathlib import Path

from legal_rag.ontology_validation.publication import expected_w3id_htaccess, expected_w3id_readme
from legal_rag.ontology_validation.publication_config import NamespacePublicationSettings


def test_contract_pins_all_canonical_and_versioned_publication_resources(
    project_root: Path,
    publication_settings: NamespacePublicationSettings,
) -> None:
    """The contract covers the root, five modules, and SHACL graph without an external write."""
    scripts = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]

    assert publication_settings.namespace_root == "https://w3id.org/legal-ontology-rag/"
    assert publication_settings.publication_base_url == "https://vdat-18.github.io/legal-ontology-rag/"
    assert publication_settings.redirect_status == 303
    assert publication_settings.timeout_seconds == 15
    assert publication_settings.max_response_bytes == 2_000_000
    assert len(publication_settings.resources) == 7
    assert scripts["legal-build-ontology-publication"] == "legal_rag.ontology_publication_cli:main"
    assert scripts["legal-verify-ontology-publication"] == "legal_rag.ontology_publication_verification_cli:main"


def test_w3id_bundle_is_contract_derived_and_contains_required_maintainer_context(
    publication_settings: NamespacePublicationSettings,
) -> None:
    """The copy-ready W3ID directory stays synchronized with the publication contract."""
    htaccess = publication_settings.htaccess_path.read_text(encoding="utf-8")
    readme = publication_settings.readme_path.read_text(encoding="utf-8")

    assert htaccess == expected_w3id_htaccess(publication_settings)
    assert readme == expected_w3id_readme(publication_settings)
    assert "RewriteRule ^ontology/?$" in htaccess
    assert "RewriteRule ^shapes/0\\.6\\.0/?$" in htaccess
    assert "[R=303,L,NE]" in htaccess
    assert "GitHub: [@vdat-18]" in readme
