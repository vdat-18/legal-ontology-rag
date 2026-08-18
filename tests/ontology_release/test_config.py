"""Tests for the versioned ontology release contract."""

from __future__ import annotations

from pathlib import Path

from legal_rag.ontology_validation.release_config import OntologyReleaseSettings


def test_production_contract_pins_approved_internal_freeze_inputs(
    project_root: Path,
    release_settings: OntologyReleaseSettings,
) -> None:
    """The committed contract contains only approved gates required for the internal V0.1 freeze."""
    assert release_settings.artifact_version == "0.1.0"
    assert release_settings.version_iri.endswith("/0.1.0")
    assert release_settings.archive_file == "legal-ontology-rag-0.1.0.zip"
    assert len(release_settings.source_paths) == 50
    assert len(release_settings.required_reuse_decisions) == 10
    assert release_settings.external_reuse_registry_path == project_root / "ontology" / "external_reuse_registry.yaml"
    assert release_settings.legal_review_contract_path == project_root / "ontology" / "legal_review_contract.yaml"
    assert release_settings.legal_review_decisions_path == project_root / "ontology" / "legal_review_decisions.yaml"
    approvals = {approval.approval_id: approval for approval in release_settings.approvals}
    assert approvals["ontology_license"].status == "approved"
    assert approvals["ontology_license"].evidence == "user-confirmed:2026-08-18:CC-BY-4.0"
    assert approvals["external_reuse_review"].status == "approved"
    assert approvals["external_reuse_review"].evidence.startswith("user-confirmed:2026-08-18:")
    assert approvals["project_owner_model_review"].status == "approved"
    assert approvals["project_owner_model_review"].evidence.startswith("user-confirmed:2026-08-18:")
    assert "namespace_resolution" not in approvals
