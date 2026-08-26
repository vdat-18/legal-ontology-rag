"""End-to-end tests for the ontology evolution diff command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from legal_rag import ontology_evolution_diff_cli
from tests.ontology_evolution.conftest import write_ontology_project


def test_cli_writes_breaking_report_before_failing(monkeypatch, tmp_path: Path) -> None:
    """Operators receive the complete semantic diff even when the gate blocks the candidate ontology."""
    project_root = Path(__file__).resolve().parents[2]
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    report_path = tmp_path / "evolution-report.json"
    write_ontology_project(before_root, terms={"Rule": "class"})
    write_ontology_project(after_root, terms={})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-diff-ontology",
            str(before_root),
            str(after_root),
            str(project_root / "ontology" / "evolution_policy.yaml"),
            str(report_path),
        ],
    )

    with pytest.raises(ValueError, match="contains breaking changes"):
        ontology_evolution_diff_cli.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "failed"
    assert report["issues"][0]["check_id"] == "removed_active_term"
