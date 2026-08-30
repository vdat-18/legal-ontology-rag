"""Tests for structural external ontology reuse gates."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from legal_rag.ontology_validation.reuse_config import ExternalReuseSettings
from legal_rag.ontology_validation.reuse_review import run_external_reuse_review

CHECKED_AT = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)


def _issues(project_root: Path, settings: ExternalReuseSettings) -> set[str]:
    """Return all issue identifiers from one review run."""
    report = run_external_reuse_review(project_root, settings, CHECKED_AT)
    return {issue for check in report.checks for issue in check.issues}


def test_current_external_reuse_review_passes_with_recorded_human_approval(
    project_root: Path,
    reuse_settings: ExternalReuseSettings,
) -> None:
    """The reviewed graph is structurally clean and reports recorded owner acceptance."""
    report = run_external_reuse_review(project_root, reuse_settings, CHECKED_AT)

    assert report.overall_status == "passed"
    assert report.project_owner_approval == "approved"
    assert len(report.active_external_terms) == 23
    assert all(check.status == "passed" for check in report.checks)


def test_unregistered_external_term_is_rejected(
    copied_project: Callable[[], tuple[Path, ExternalReuseSettings]],
) -> None:
    """Formal semantic use cannot silently escape the reviewed term inventory."""
    project_root, settings = copied_project()
    root = project_root / "ontology" / "legal_ontology.ttl"
    root.write_text(
        root.read_text(encoding="utf-8")
        + '\n<https://w3id.org/legal-ontology-rag/ontology> <https://example.test/unregistered> "value" .\n',
        encoding="utf-8",
    )

    assert "unregistered-external-term:https://example.test/unregistered" in _issues(project_root, settings)


def test_missing_foundational_declaration_is_rejected(
    copied_project: Callable[[], tuple[Path, ExternalReuseSettings]],
) -> None:
    """Registry terms marked for the OWL closure must remain explicitly declared."""
    project_root, settings = copied_project()
    source = project_root / "ontology" / "modules" / "source_provenance.ttl"
    source.write_text(
        source.read_text(encoding="utf-8").replace("prov:Agent a owl:Class .\n", ""),
        encoding="utf-8",
    )

    assert "missing-foundational-declaration:http://www.w3.org/ns/prov#Agent" in _issues(project_root, settings)


def test_unregistered_research_namespace_use_is_rejected(
    copied_project: Callable[[], tuple[Path, ExternalReuseSettings]],
) -> None:
    """Research-only namespaces cannot become formal terms without an active registry entry."""
    project_root, settings = copied_project()
    root = project_root / "ontology" / "legal_ontology.ttl"
    root.write_text(
        root.read_text(encoding="utf-8")
        + '\n<https://w3id.org/legal-ontology-rag/ontology> <http://data.europa.eu/eli/ontology#title> "value" .\n',
        encoding="utf-8",
    )

    assert "unregistered-external-term:http://data.europa.eu/eli/ontology#title" in _issues(project_root, settings)


def test_external_owl_import_is_rejected(
    copied_project: Callable[[], tuple[Path, ExternalReuseSettings]],
) -> None:
    """V0.1 cannot acquire an unreviewed external import closure."""
    project_root, settings = copied_project()
    root = project_root / "ontology" / "legal_ontology.ttl"
    root.write_text(
        root.read_text(encoding="utf-8")
        + "\n<https://w3id.org/legal-ontology-rag/ontology> <http://www.w3.org/2002/07/owl#imports> "
        "<https://example.test/external-ontology> .\n",
        encoding="utf-8",
    )

    assert "external-owl-import:https://example.test/external-ontology" in _issues(project_root, settings)
