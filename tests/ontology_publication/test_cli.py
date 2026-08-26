"""End-to-end test for the ontology namespace-publication command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from legal_rag import ontology_publication_cli


def test_cli_builds_site_and_persists_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    project_root: Path,
) -> None:
    """The operator command produces both deployable files and a durable readiness report."""
    site_path = tmp_path / "site"
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-build-ontology-publication",
            str(project_root / "ontology" / "namespace_publication.yaml"),
            str(site_path),
            str(report_path),
        ],
    )

    ontology_publication_cli.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "built"
    assert (site_path / "publication-manifest.json").is_file()
    assert (site_path / "shapes" / "0.6.0" / "ontology.ttl").is_file()
