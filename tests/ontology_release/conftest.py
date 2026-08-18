"""Shared fixtures for ontology release packaging tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from legal_rag.ontology_validation.release_config import OntologyReleaseSettings
from legal_rag.ontology_validation.semantic_models import SEMANTIC_QA_REPORT_SCHEMA_VERSION


@pytest.fixture
def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def release_settings(project_root: Path) -> OntologyReleaseSettings:
    """Load the production release contract."""
    return OntologyReleaseSettings.from_path(project_root / "ontology" / "release_contract.yaml", project_root)


def write_releasable_project(path: Path, source_project_root: Path) -> tuple[OntologyReleaseSettings, Path]:
    """Create one minimal approved project and passing semantic report."""
    ontology_directory = path / "ontology"
    ontology_directory.mkdir(parents=True)
    (ontology_directory / "legal_ontology.ttl").write_text(
        """@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
<https://w3id.org/legal-ontology-rag/ontology> a owl:Ontology ;
    owl:versionIRI <https://w3id.org/legal-ontology-rag/ontology/0.1.0> ;
    dcterms:license <https://example.test/license> .
""",
        encoding="utf-8",
    )
    (path / "LICENSE").write_text("Approved test license.\n", encoding="utf-8")
    (ontology_directory / "formalization_policy.yaml").write_text(
        yaml.safe_dump(
            {
                "external_dependency_policy": [
                    {
                        "id": "REUSE-DCTERMS-001",
                        "mode": "term_iri_reference",
                        "owl_import": False,
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    reuse_registry = {
        "schema_version": "0.1.0",
        "artifact_id": "test-external-reuse-registry",
        "artifact_version": "0.1.0",
        "status": "approved-test-fixture",
        "review": {
            "reviewed_on": "2026-08-18",
            "reviewer_role": "test",
            "project_owner_approval": "approved",
        },
        "scope": {
            "formal_artifacts": ["ontology/legal_ontology.ttl"],
            "foundational_namespaces": [
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "http://www.w3.org/2002/07/owl#",
            ],
        },
        "resources": [
            {
                "id": "REUSE-DCTERMS-001",
                "resource": "DCMI Metadata Terms",
                "authority": "Dublin Core Metadata Initiative",
                "source_version": "test snapshot",
                "source_url": "https://www.dublincore.org/specifications/dublin-core/dcmi-terms/",
                "namespace": "http://purl.org/dc/terms/",
                "license": "CC-BY-4.0",
                "attribution": "Dublin Core Metadata Initiative",
                "decision": "reuse",
                "policy_mode": "term_iri_reference",
                "formal_use": "active_terms",
                "owl_import": False,
                "review_status": "ontology_reviewed",
                "compatibility": "Metadata-only test use.",
                "boundaries": ["Use only the reviewed license predicate."],
                "terms": [
                    {
                        "iri": "http://purl.org/dc/terms/license",
                        "local_use": "Ontology license metadata.",
                        "declaration_in_owl_closure": False,
                    }
                ],
            }
        ],
        "review_conclusion": {},
    }
    (ontology_directory / "external_reuse_registry.yaml").write_text(
        yaml.safe_dump(reuse_registry, sort_keys=False),
        encoding="utf-8",
    )
    shutil.copyfile(
        source_project_root / "ontology" / "semantic_qa_contract.yaml",
        ontology_directory / "semantic_qa_contract.yaml",
    )
    contract = {
        "schema_version": "0.1.0",
        "artifact_id": "test-ontology-release",
        "artifact_version": "0.1.0",
        "ontology": {
            "ontology_iri": "https://w3id.org/legal-ontology-rag/ontology",
            "version_iri": "https://w3id.org/legal-ontology-rag/ontology/0.1.0",
            "root_file": "ontology/legal_ontology.ttl",
            "license_file": "LICENSE",
        },
        "approvals": {
            "namespace_resolution": {"status": "approved", "evidence": "review:NS-001"},
            "ontology_license": {"status": "approved", "evidence": "review:LIC-001"},
            "external_reuse_review": {"status": "approved", "evidence": "review:REUSE-001"},
            "independent_legal_review": {"status": "approved", "evidence": "review:LEGAL-001"},
        },
        "required_reuse_decisions": ["REUSE-DCTERMS-001"],
        "inputs": {
            "semantic_contract_file": "ontology/semantic_qa_contract.yaml",
            "external_reuse_registry_file": "ontology/external_reuse_registry.yaml",
            "semantic_report_archive_path": "qa/semantic_qa_report.json",
            "source_files": [
                "ontology/legal_ontology.ttl",
                "ontology/formalization_policy.yaml",
                "ontology/external_reuse_registry.yaml",
                "ontology/semantic_qa_contract.yaml",
                "ontology/release_contract.yaml",
            ],
        },
        "output": {
            "archive_root": "test-ontology-0.1.0",
            "archive_file": "test-ontology-0.1.0.zip",
            "checksum_file": "test-ontology-0.1.0.zip.sha256",
            "manifest_archive_path": "MANIFEST.json",
            "report_schema_version": "0.1.0",
        },
    }
    contract_path = ontology_directory / "release_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    semantic_report_path = path / "semantic-report.json"
    semantic_report_path.write_text(
        json.dumps(
            {
                "schema_version": SEMANTIC_QA_REPORT_SCHEMA_VERSION,
                "overall_status": "passed",
                "ontology_path": "ontology/legal_ontology.ttl",
                "input_sha256": "semantic-input",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return OntologyReleaseSettings.from_path(contract_path, path), semantic_report_path
