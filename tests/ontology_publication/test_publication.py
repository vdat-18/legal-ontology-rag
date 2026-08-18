"""Tests for static ontology publication generation and readiness gates."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from legal_rag.ontology_validation.publication import build_namespace_publication
from legal_rag.ontology_validation.publication_config import NamespacePublicationSettings

CHECKED_AT = datetime(2026, 8, 18, 14, 0, tzinfo=UTC)


def test_publication_site_is_complete_deterministic_and_byte_exact(
    tmp_path: Path,
    project_root: Path,
    publication_settings: NamespacePublicationSettings,
) -> None:
    """The deployable site has 14 HTML/Turtle endpoints and a deterministic manifest."""
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = build_namespace_publication(project_root, publication_settings, first_output, CHECKED_AT)
    second = build_namespace_publication(project_root, publication_settings, second_output, CHECKED_AT)

    assert first.overall_status == "built"
    assert all(check.status == "passed" for check in first.checks)
    assert len(first.files) == 29
    assert [(item.path, item.sha256) for item in first.files] == [(item.path, item.sha256) for item in second.files]
    manifest = json.loads((first_output / publication_settings.manifest_file).read_text(encoding="utf-8"))
    assert len(manifest["endpoints"]) == 14
    assert manifest["representations"]["turtle"] == "text/turtle; charset=utf-8"

    root_source = publication_settings.resources[0].source_path.read_bytes()
    assert (first_output / "ontology" / "ontology.ttl").read_bytes() == root_source
    assert (first_output / "ontology" / "0.1.0" / "ontology.ttl").read_bytes() == root_source
    assert "CC BY 4.0" in (first_output / "ontology" / "index.html").read_text(encoding="utf-8")


def test_existing_output_and_stale_w3id_bundle_fail_closed(
    tmp_path: Path,
    project_root: Path,
    publication_settings: NamespacePublicationSettings,
) -> None:
    """A stale redirect bundle or non-fresh deployment directory prevents site generation."""
    stale_htaccess = tmp_path / ".htaccess"
    stale_htaccess.write_text("RewriteEngine Off\n", encoding="utf-8")
    stale_settings = replace(publication_settings, htaccess_path=stale_htaccess)
    output_directory = tmp_path / "site"
    output_directory.mkdir()

    report = build_namespace_publication(project_root, stale_settings, output_directory, CHECKED_AT)
    checks = {item.check_id: item for item in report.checks}

    assert report.overall_status == "failed"
    assert checks["w3id-pr-bundle"].issues == ("w3id-htaccess-out-of-date",)
    assert checks["fresh-output-path"].status == "failed"
    assert report.files == ()


def test_catalog_or_ontology_identity_drift_is_reported(
    tmp_path: Path,
    project_root: Path,
    publication_settings: NamespacePublicationSettings,
) -> None:
    """Contract IRIs cannot diverge from the Turtle source or local XML catalog."""
    first = publication_settings.resources[0]
    drifted = replace(first, version_iri="https://w3id.org/legal-ontology-rag/ontology/9.9.9")
    settings = replace(publication_settings, resources=(drifted, *publication_settings.resources[1:]))

    report = build_namespace_publication(project_root, settings, tmp_path / "site", CHECKED_AT)
    checks = {item.check_id: item for item in report.checks}

    assert report.overall_status == "failed"
    assert checks["resource-identity"].status == "failed"
    assert checks["catalog-mappings"].status == "failed"
    assert checks["publication-endpoints"].status == "failed"
