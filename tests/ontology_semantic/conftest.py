"""Shared fixtures for ontology semantic-QA tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from legal_rag.ontology_validation.paths import OntologyPaths
from legal_rag.ontology_validation.semantic_config import SemanticQaSettings
from legal_rag.ontology_validation.semantic_models import ProcessResult


@pytest.fixture
def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def ontology_paths(project_root: Path) -> OntologyPaths:
    """Return resolved ontology artifact paths."""
    return OntologyPaths(project_root)


@pytest.fixture
def semantic_settings(project_root: Path) -> SemanticQaSettings:
    """Load the production semantic-QA contract."""
    return SemanticQaSettings.from_path(project_root / "ontology" / "semantic_qa_contract.yaml", project_root)


class FakeRobotRunner:
    """Write deterministic ROBOT artifacts without Docker or Java."""

    def __init__(
        self,
        settings: SemanticQaSettings,
        *,
        profile_exit: int = 0,
        reason_exit: int = 0,
        quality_exit: int = 0,
        reasoned_turtle: str = "@prefix owl: <http://www.w3.org/2002/07/owl#> . <> a owl:Ontology .\n",
    ) -> None:
        """Configure deterministic command exits and reasoned output."""
        self._settings = settings
        self._exit_codes = {
            "validate-profile": profile_exit,
            "reason": reason_exit,
            "report": quality_exit,
        }
        self._reasoned_turtle = reasoned_turtle
        self.calls: list[tuple[str, ...]] = []

    def run(self, arguments: Sequence[str], output_directory: Path) -> ProcessResult:
        """Capture arguments, create configured artifacts, and return one process result."""
        arguments = tuple(arguments)
        self.calls.append(arguments)
        output_directory.mkdir(parents=True, exist_ok=True)
        command = arguments[0]
        exit_code = self._exit_codes[command]
        if command == "validate-profile":
            (output_directory / self._settings.profile_report_file).write_text(
                "OWL 2 DL Profile Report: [Ontology and imports closure in profile]\n",
                encoding="utf-8",
            )
        elif command == "reason" and exit_code == 0:
            (output_directory / self._settings.reasoned_ontology_file).write_text(
                self._reasoned_turtle,
                encoding="utf-8",
            )
        elif command == "report":
            (output_directory / self._settings.quality_report_file).write_text(
                "Level\tRule Name\tSubject\tProperty\tValue\n",
                encoding="utf-8",
            )
        return ProcessResult(
            arguments=arguments,
            exit_code=exit_code,
            stdout="" if exit_code == 0 else f"{command} failed",
            stderr="",
        )
