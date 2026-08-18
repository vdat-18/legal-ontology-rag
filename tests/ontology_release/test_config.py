"""Tests for the versioned ontology release contract."""

from __future__ import annotations

from legal_rag.ontology_validation.release_config import OntologyReleaseSettings


def test_production_contract_pins_identity_inventory_and_fail_closed_approvals(
    release_settings: OntologyReleaseSettings,
) -> None:
    """The committed release contract is explicit and remains blocked pending human decisions."""
    assert release_settings.artifact_version == "0.1.0"
    assert release_settings.version_iri.endswith("/0.1.0")
    assert release_settings.archive_file == "legal-ontology-rag-0.1.0.zip"
    assert len(release_settings.source_paths) == 44
    assert len(release_settings.required_reuse_decisions) == 9
    assert all(approval.status == "pending" for approval in release_settings.approvals)
    assert all(approval.evidence is None for approval in release_settings.approvals)
