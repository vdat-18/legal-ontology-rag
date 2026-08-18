"""Typed configuration for ontology release packaging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_validation.loader import load_yaml

RELEASE_CONTRACT_SCHEMA_VERSION = "0.1.0"
RELEASE_REPORT_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class ReleaseApproval:
    """One explicit human release approval."""

    approval_id: str
    status: str
    evidence: str | None


@dataclass(frozen=True, slots=True)
class OntologyReleaseSettings:
    """Immutable release identity, gates, inventory, and output settings."""

    artifact_id: str
    artifact_version: str
    ontology_iri: str
    version_iri: str
    root_ontology_path: Path
    license_path: Path
    approvals: tuple[ReleaseApproval, ...]
    required_reuse_decisions: tuple[str, ...]
    source_paths: tuple[Path, ...]
    semantic_contract_path: Path
    semantic_report_archive_path: str
    archive_root: str
    archive_file: str
    checksum_file: str
    manifest_archive_path: str

    @classmethod
    def from_path(cls, path: Path, project_root: Path) -> OntologyReleaseSettings:
        """Load the versioned ontology release contract."""
        document = load_yaml(path)
        if document["schema_version"] != RELEASE_CONTRACT_SCHEMA_VERSION:
            msg = f"Unsupported ontology release contract: {document['schema_version']}"
            raise ValueError(msg)
        output = document["output"]
        if output["report_schema_version"] != RELEASE_REPORT_SCHEMA_VERSION:
            msg = f"Unsupported ontology release report schema: {output['report_schema_version']}"
            raise ValueError(msg)
        ontology = document["ontology"]
        inputs = document["inputs"]
        return cls(
            artifact_id=document["artifact_id"],
            artifact_version=document["artifact_version"],
            ontology_iri=ontology["ontology_iri"],
            version_iri=ontology["version_iri"],
            root_ontology_path=project_root / ontology["root_file"],
            license_path=project_root / ontology["license_file"],
            approvals=tuple(
                ReleaseApproval(approval_id, approval["status"], approval["evidence"])
                for approval_id, approval in document["approvals"].items()
            ),
            required_reuse_decisions=tuple(document["required_reuse_decisions"]),
            source_paths=tuple(project_root / item for item in inputs["source_files"]),
            semantic_contract_path=project_root / inputs["semantic_contract_file"],
            semantic_report_archive_path=inputs["semantic_report_archive_path"],
            archive_root=output["archive_root"],
            archive_file=output["archive_file"],
            checksum_file=output["checksum_file"],
            manifest_archive_path=output["manifest_archive_path"],
        )
