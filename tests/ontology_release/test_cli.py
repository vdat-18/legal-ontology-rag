"""End-to-end test for the ontology release operator command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from legal_rag import ontology_release_cli
from legal_rag.ontology_validation.paths import OntologyPaths
from legal_rag.ontology_validation.release_config import OntologyReleaseSettings
from legal_rag.ontology_validation.semantic_config import SemanticQaSettings
from legal_rag.ontology_validation.semantic_models import SEMANTIC_QA_REPORT_SCHEMA_VERSION
from legal_rag.ontology_validation.semantic_qa import semantic_input_sha256


def test_cli_persists_blocked_readiness_before_raising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    project_root: Path,
) -> None:
    """Operators receive a durable gate report even when no release can be emitted."""
    contract_path = project_root / "ontology" / "release_contract.yaml"
    settings = OntologyReleaseSettings.from_path(contract_path, project_root)
    semantic_settings = SemanticQaSettings.from_path(settings.semantic_contract_path, project_root)
    semantic_report_path = tmp_path / "semantic-report.json"
    semantic_report_path.write_text(
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
    report_path = tmp_path / "release-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-package-ontology-release",
            str(contract_path),
            str(semantic_report_path),
            str(tmp_path / "release"),
            str(report_path),
        ],
    )

    with pytest.raises(ValueError, match="release is blocked"):
        ontology_release_cli.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "blocked"
    assert report["archive_path"] is None
