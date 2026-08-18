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
    assert len(release_settings.source_paths) == 47
    assert len(release_settings.required_reuse_decisions) == 9
    approvals = {approval.approval_id: approval for approval in release_settings.approvals}
    assert approvals["ontology_license"].status == "approved"
    assert approvals["ontology_license"].evidence == "user-confirmed:2026-08-18:CC-BY-4.0"
    assert all(
        approvals[approval_id].status == "pending"
        for approval_id in ("namespace_resolution", "external_reuse_review", "independent_legal_review")
    )
