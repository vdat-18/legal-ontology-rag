"""Tests for ontology release gates and reproducible packaging."""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from legal_rag.ontology_validation.paths import OntologyPaths
from legal_rag.ontology_validation.release_config import OntologyReleaseSettings
from legal_rag.ontology_validation.release_packager import build_ontology_release
from legal_rag.ontology_validation.semantic_config import SemanticQaSettings
from legal_rag.ontology_validation.semantic_models import SEMANTIC_QA_REPORT_SCHEMA_VERSION
from legal_rag.ontology_validation.semantic_qa import semantic_input_sha256

from .conftest import write_releasable_project

CHECKED_AT = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)


def test_production_release_is_blocked_by_explicit_publication_gates(
    tmp_path: Path,
    project_root: Path,
    release_settings: OntologyReleaseSettings,
) -> None:
    """No archive is emitted while approvals and ontology licensing remain unresolved."""
    semantic_settings = SemanticQaSettings.from_path(release_settings.semantic_contract_path, project_root)
    report_path = tmp_path / "semantic-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": SEMANTIC_QA_REPORT_SCHEMA_VERSION,
                "overall_status": "passed",
                "ontology_path": "ontology/legal_ontology.ttl",
                "input_sha256": semantic_input_sha256(OntologyPaths(project_root), semantic_settings),
            }
        ),
        encoding="utf-8",
    )

    report = build_ontology_release(project_root, release_settings, report_path, tmp_path / "release", CHECKED_AT)
    gates = {item.gate_id: item for item in report.gates}

    assert report.overall_status == "blocked"
    assert gates["human-approvals"].status == "failed"
    assert gates["ontology-license"].status == "failed"
    assert gates["release-file-inventory"].status == "failed"
    assert gates["semantic-qa"].status == "passed"
    assert report.archive_path is None
    assert not (tmp_path / "release").exists()


def test_release_archive_is_reproducible_and_self_describing(
    monkeypatch,
    tmp_path: Path,
    project_root: Path,
) -> None:
    """Identical approved inputs produce identical ZIP bytes, checksums, and internal manifests."""
    release_project = tmp_path / "project"
    settings, semantic_report_path = write_releasable_project(release_project, project_root)
    monkeypatch.setattr(
        "legal_rag.ontology_validation.release_packager.semantic_input_sha256",
        lambda _paths, _settings: "semantic-input",
    )

    first = build_ontology_release(
        release_project,
        settings,
        semantic_report_path,
        tmp_path / "first",
        CHECKED_AT,
    )
    second = build_ontology_release(
        release_project,
        settings,
        semantic_report_path,
        tmp_path / "second",
        datetime(2026, 8, 19, 12, 0, tzinfo=UTC),
    )

    assert first.overall_status == "released"
    assert second.overall_status == "released"
    assert first.archive_sha256 == second.archive_sha256
    first_archive = Path(first.archive_path or "")
    second_archive = Path(second.archive_path or "")
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert (tmp_path / "first" / settings.checksum_file).read_text(encoding="ascii") == (
        tmp_path / "second" / settings.checksum_file
    ).read_text(encoding="ascii")

    with zipfile.ZipFile(first_archive) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in archive.infolist())
        manifest = json.loads(archive.read(f"{settings.archive_root}/MANIFEST.json"))
    assert manifest["artifact_version"] == "0.1.0"
    assert manifest["files"] == [item.to_dict() for item in first.files]


def test_stale_semantic_report_blocks_release(
    monkeypatch,
    tmp_path: Path,
    project_root: Path,
) -> None:
    """A passing status cannot authorize changed semantic inputs."""
    release_project = tmp_path / "project"
    settings, semantic_report_path = write_releasable_project(release_project, project_root)
    monkeypatch.setattr(
        "legal_rag.ontology_validation.release_packager.semantic_input_sha256",
        lambda _paths, _settings: "changed-input",
    )

    report = build_ontology_release(
        release_project,
        settings,
        semantic_report_path,
        tmp_path / "release",
        CHECKED_AT,
    )

    assert report.overall_status == "blocked"
    assert {item.gate_id: item.status for item in report.gates}["semantic-qa"] == "failed"
    assert report.archive_path is None


def test_existing_release_outputs_are_never_replaced(
    monkeypatch,
    tmp_path: Path,
    project_root: Path,
) -> None:
    """Immutable output paths fail closed before a prior release can be overwritten."""
    release_project = tmp_path / "project"
    settings, semantic_report_path = write_releasable_project(release_project, project_root)
    monkeypatch.setattr(
        "legal_rag.ontology_validation.release_packager.semantic_input_sha256",
        lambda _paths, _settings: "semantic-input",
    )
    output_directory = tmp_path / "release"
    output_directory.mkdir()
    archive_path = output_directory / settings.archive_file
    archive_path.write_bytes(b"existing-release")

    report = build_ontology_release(
        release_project,
        settings,
        semantic_report_path,
        output_directory,
        CHECKED_AT,
    )

    assert report.overall_status == "blocked"
    assert {item.gate_id: item.status for item in report.gates}["immutable-output-paths"] == "failed"
    assert archive_path.read_bytes() == b"existing-release"
