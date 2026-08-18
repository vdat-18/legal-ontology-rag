"""End-to-end tests for the semantic-QA operator command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from legal_rag import ontology_semantic_qa_cli
from legal_rag.ontology_validation import semantic_runner

from .conftest import FakeRobotRunner


class _CliRunner(FakeRobotRunner):
    """Construct the fake runner from the production CLI arguments."""

    quality_exit = 0

    def __init__(self, _project_root, settings) -> None:
        super().__init__(settings, quality_exit=type(self).quality_exit)


def test_cli_persists_passing_and_failing_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    project_root: Path,
) -> None:
    """The command always writes diagnostics and raises only after a failed report is durable."""
    artifacts = tmp_path / "artifacts"
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(semantic_runner, "DockerRobotRunner", _CliRunner)
    argv = [
        "legal-validate-ontology-semantics",
        str(project_root / "ontology" / "semantic_qa_contract.yaml"),
        str(artifacts),
        str(report_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    _CliRunner.quality_exit = 0
    ontology_semantic_qa_cli.main()
    assert json.loads(report_path.read_text(encoding="utf-8"))["overall_status"] == "passed"

    _CliRunner.quality_exit = 1
    with pytest.raises(ValueError, match="semantic QA failed"):
        ontology_semantic_qa_cli.main()
    assert json.loads(report_path.read_text(encoding="utf-8"))["overall_status"] == "failed"
