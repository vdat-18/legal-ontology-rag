"""Validate release gates and build a reproducible ontology archive."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from rdflib import DCTERMS, OWL, Graph, URIRef

from legal_rag.ontology_validation.legal_review import collect_legal_review_packet_issues
from legal_rag.ontology_validation.legal_review_config import LegalReviewSettings
from legal_rag.ontology_validation.loader import load_yaml
from legal_rag.ontology_validation.paths import OntologyPaths
from legal_rag.ontology_validation.release_config import OntologyReleaseSettings
from legal_rag.ontology_validation.release_models import (
    OntologyReleaseReport,
    ReleaseFileRecord,
    ReleaseGateResult,
)
from legal_rag.ontology_validation.reuse_config import ExternalReuseSettings
from legal_rag.ontology_validation.reuse_review import collect_external_reuse_issues
from legal_rag.ontology_validation.semantic_config import SemanticQaSettings
from legal_rag.ontology_validation.semantic_models import SEMANTIC_QA_REPORT_SCHEMA_VERSION
from legal_rag.ontology_validation.semantic_qa import semantic_input_sha256

_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ZIP_FILE_MODE = 0o100644 << 16


def build_ontology_release(
    project_root: Path,
    settings: OntologyReleaseSettings,
    semantic_report_path: Path,
    output_directory: Path,
    checked_at: datetime,
) -> OntologyReleaseReport:
    """Evaluate all gates and create the release only when every gate passes."""
    semantic_report = json.loads(semantic_report_path.read_text(encoding="utf-8"))
    files = _source_inventory(project_root, settings, semantic_report_path)
    gates = (
        _approval_gate(settings),
        _semantic_report_gate(project_root, settings, semantic_report),
        _ontology_identity_gate(settings),
        _license_gate(settings),
        _reuse_decisions_gate(project_root, settings),
        _external_reuse_gate(project_root, settings),
        _legal_review_packet_gate(project_root, settings),
        _inventory_gate(settings, semantic_report_path),
        _output_paths_gate(output_directory, settings),
    )
    if any(gate.status == "failed" for gate in gates):
        return OntologyReleaseReport(
            checked_at=checked_at,
            artifact_id=settings.artifact_id,
            artifact_version=settings.artifact_version,
            overall_status="blocked",
            gates=gates,
            files=files,
            archive_path=None,
            archive_sha256=None,
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    archive_path = output_directory / settings.archive_file
    checksum_path = output_directory / settings.checksum_file
    with tempfile.TemporaryDirectory(dir=output_directory) as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_archive = temporary_root / settings.archive_file
        _write_archive(project_root, settings, semantic_report_path, files, temporary_archive)
        archive_sha256 = _sha256(temporary_archive.read_bytes())
        temporary_checksum = temporary_root / settings.checksum_file
        temporary_checksum.write_text(f"{archive_sha256}  {settings.archive_file}\n", encoding="ascii")
        os.link(temporary_archive, archive_path)
        os.link(temporary_checksum, checksum_path)
    return OntologyReleaseReport(
        checked_at=checked_at,
        artifact_id=settings.artifact_id,
        artifact_version=settings.artifact_version,
        overall_status="released",
        gates=gates,
        files=files,
        archive_path=str(archive_path),
        archive_sha256=archive_sha256,
    )


def _approval_gate(settings: OntologyReleaseSettings) -> ReleaseGateResult:
    """Require every named human approval and its evidence reference."""
    pending = tuple(
        approval.approval_id
        for approval in settings.approvals
        if approval.status != "approved" or not approval.evidence
    )
    return _gate("human-approvals", not pending, f"Pending approvals: {', '.join(pending)}")


def _semantic_report_gate(
    project_root: Path,
    settings: OntologyReleaseSettings,
    report: dict[str, object],
) -> ReleaseGateResult:
    """Require a passing semantic report for the contracted ontology version."""
    semantic_settings = SemanticQaSettings.from_path(settings.semantic_contract_path, project_root)
    expected_input_sha256 = semantic_input_sha256(OntologyPaths(project_root), semantic_settings)
    valid = (
        report["schema_version"] == SEMANTIC_QA_REPORT_SCHEMA_VERSION
        and report["overall_status"] == "passed"
        and report["ontology_path"]
        == settings.root_ontology_path.resolve().relative_to(project_root.resolve()).as_posix()
        and report["input_sha256"] == expected_input_sha256
    )
    return _gate("semantic-qa", valid, "Semantic QA is failing, stale, or belongs to another ontology input closure.")


def _ontology_identity_gate(settings: OntologyReleaseSettings) -> ReleaseGateResult:
    """Require the root ontology and version IRIs to match the release contract."""
    graph = Graph().parse(settings.root_ontology_path, format="turtle")
    ontology = URIRef(settings.ontology_iri)
    valid = (ontology, OWL.versionIRI, URIRef(settings.version_iri)) in graph
    return _gate("ontology-identity", valid, "Root ontology or version IRI does not match the release contract.")


def _license_gate(settings: OntologyReleaseSettings) -> ReleaseGateResult:
    """Require a distributable license file and matching ontology metadata."""
    graph = Graph().parse(settings.root_ontology_path, format="turtle")
    has_metadata = any(graph.objects(URIRef(settings.ontology_iri), DCTERMS.license))
    valid = settings.license_path.is_file() and settings.license_path.stat().st_size > 0 and has_metadata
    return _gate("ontology-license", valid, "A license file and dcterms:license assertion are required.")


def _reuse_decisions_gate(project_root: Path, settings: OntologyReleaseSettings) -> ReleaseGateResult:
    """Require every contracted external-reuse decision in the formalization policy."""
    policy = load_yaml(project_root / "ontology" / "formalization_policy.yaml")
    available = {item["id"] for item in policy["external_dependency_policy"]}
    missing = tuple(sorted(set(settings.required_reuse_decisions) - available))
    return _gate("reuse-decision-inventory", not missing, f"Missing reuse decisions: {', '.join(missing)}")


def _external_reuse_gate(project_root: Path, settings: OntologyReleaseSettings) -> ReleaseGateResult:
    """Require the current formal graph to satisfy the versioned external-reuse registry."""
    reuse_settings = ExternalReuseSettings.from_path(settings.external_reuse_registry_path, project_root)
    issues = collect_external_reuse_issues(project_root, reuse_settings)
    return _gate("external-reuse-registry", not issues, f"External reuse issues: {', '.join(issues)}")


def _legal_review_packet_gate(project_root: Path, settings: OntologyReleaseSettings) -> ReleaseGateResult:
    """Require complete internal-review questions, coverage, evidence, and decision records."""
    review_settings = LegalReviewSettings.from_paths(
        settings.legal_review_contract_path,
        settings.legal_review_decisions_path,
        project_root,
    )
    issues = collect_legal_review_packet_issues(project_root, review_settings)
    return _gate("internal-legal-review-packet", not issues, f"Legal review packet issues: {', '.join(issues)}")


def _inventory_gate(
    settings: OntologyReleaseSettings,
    semantic_report_path: Path,
) -> ReleaseGateResult:
    """Require every contracted source and generated QA input."""
    required = (*settings.source_paths, settings.license_path, semantic_report_path)
    missing = tuple(str(path) for path in required if not path.is_file())
    return _gate("release-file-inventory", not missing, f"Missing release files: {', '.join(missing)}")


def _output_paths_gate(output_directory: Path, settings: OntologyReleaseSettings) -> ReleaseGateResult:
    """Protect immutable release outputs from replacement."""
    existing = tuple(
        str(path)
        for path in (output_directory / settings.archive_file, output_directory / settings.checksum_file)
        if path.exists()
    )
    return _gate("immutable-output-paths", not existing, f"Release outputs already exist: {', '.join(existing)}")


def _source_inventory(
    project_root: Path,
    settings: OntologyReleaseSettings,
    semantic_report_path: Path,
) -> tuple[ReleaseFileRecord, ...]:
    """Return deterministic records for every existing release payload file."""
    paths = (*settings.source_paths, settings.license_path)
    records = [
        _file_record(path.resolve().relative_to(project_root.resolve()).as_posix(), path)
        for path in paths
        if path.is_file()
    ]
    if semantic_report_path.is_file():
        records.append(_file_record(settings.semantic_report_archive_path, semantic_report_path))
    return tuple(sorted(records, key=lambda item: item.path))


def _file_record(archive_path: str, source_path: Path) -> ReleaseFileRecord:
    """Hash one release input."""
    content = source_path.read_bytes()
    return ReleaseFileRecord(path=archive_path, sha256=_sha256(content), size_bytes=len(content))


def _write_archive(
    project_root: Path,
    settings: OntologyReleaseSettings,
    semantic_report_path: Path,
    files: tuple[ReleaseFileRecord, ...],
    archive_path: Path,
) -> None:
    """Write the normalized archive and its internal manifest."""
    sources = {
        path.resolve().relative_to(project_root.resolve()).as_posix(): path
        for path in (*settings.source_paths, settings.license_path)
    }
    sources[settings.semantic_report_archive_path] = semantic_report_path
    manifest = {
        "schema_version": "0.1.0",
        "artifact_id": settings.artifact_id,
        "artifact_version": settings.artifact_version,
        "ontology_iri": settings.ontology_iri,
        "version_iri": settings.version_iri,
        "files": [item.to_dict() for item in files],
    }
    entries = {f"{settings.archive_root}/{item.path}": sources[item.path].read_bytes() for item in files}
    manifest_content = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    entries[f"{settings.archive_root}/{settings.manifest_archive_path}"] = manifest_content
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, content in sorted(entries.items()):
            _write_zip_entry(archive, path, content)


def _write_zip_entry(archive: zipfile.ZipFile, path: str, content: bytes) -> None:
    """Write one ZIP member with normalized metadata."""
    info = zipfile.ZipInfo(path, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = _ZIP_FILE_MODE
    archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _gate(gate_id: str, passed: bool, failed_summary: str) -> ReleaseGateResult:
    """Build a stable release gate result."""
    return ReleaseGateResult(
        gate_id=gate_id,
        status="passed" if passed else "failed",
        summary="No issues found." if passed else failed_summary,
    )


def _sha256(content: bytes) -> str:
    """Return the lowercase SHA-256 digest for bytes."""
    return hashlib.sha256(content).hexdigest()
