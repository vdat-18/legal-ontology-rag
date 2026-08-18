"""Typed configuration for ontology namespace publication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_validation.loader import load_yaml

PUBLICATION_CONTRACT_SCHEMA_VERSION = "0.1.0"
PUBLICATION_REPORT_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class PublicationResource:
    """One canonical and versioned ontology publication resource."""

    resource_id: str
    label: str
    ontology_iri: str
    version_iri: str
    source_path: Path
    canonical_path: str
    version_path: str


@dataclass(frozen=True, slots=True)
class NamespacePublicationSettings:
    """Namespace identity, W3ID bundle, and publication-site settings."""

    artifact_id: str
    artifact_version: str
    namespace_root: str
    publication_base_url: str
    license_iri: str
    github_username: str
    project_repository: str
    redirect_status: int
    bundle_directory: Path
    readme_path: Path
    htaccess_path: Path
    timeout_seconds: float
    max_response_bytes: int
    user_agent: str
    manifest_file: str
    resources: tuple[PublicationResource, ...]

    @classmethod
    def from_path(cls, path: Path, project_root: Path) -> NamespacePublicationSettings:
        """Load the versioned namespace-publication contract."""
        document = load_yaml(path)
        if document["schema_version"] != PUBLICATION_CONTRACT_SCHEMA_VERSION:
            msg = f"Unsupported namespace-publication contract: {document['schema_version']}"
            raise ValueError(msg)
        output = document["output"]
        if output["report_schema_version"] != PUBLICATION_REPORT_SCHEMA_VERSION:
            msg = f"Unsupported namespace-publication report schema: {output['report_schema_version']}"
            raise ValueError(msg)
        w3id = document["w3id"]
        bundle_directory = project_root / w3id["bundle_directory"]
        maintainer = document["maintainer"]
        verification = document["verification"]
        return cls(
            artifact_id=document["artifact_id"],
            artifact_version=document["artifact_version"],
            namespace_root=document["namespace_root"],
            publication_base_url=document["publication_base_url"],
            license_iri=document["license_iri"],
            github_username=maintainer["github_username"],
            project_repository=maintainer["project_repository"],
            redirect_status=w3id["redirect_status"],
            bundle_directory=bundle_directory,
            readme_path=bundle_directory / w3id["readme_file"],
            htaccess_path=bundle_directory / w3id["htaccess_file"],
            timeout_seconds=verification["timeout_seconds"],
            max_response_bytes=verification["max_response_bytes"],
            user_agent=verification["user_agent"],
            manifest_file=output["manifest_file"],
            resources=tuple(
                PublicationResource(
                    resource_id=item["id"],
                    label=item["label"],
                    ontology_iri=item["ontology_iri"],
                    version_iri=item["version_iri"],
                    source_path=project_root / item["source_file"],
                    canonical_path=item["canonical_path"],
                    version_path=item["version_path"],
                )
                for item in document["resources"]
            ),
        )
