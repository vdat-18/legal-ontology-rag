"""Run pinned ROBOT commands in a bounded network-isolated container."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from legal_rag.ontology_validation.semantic_config import SemanticQaSettings
from legal_rag.ontology_validation.semantic_models import ProcessResult


class RobotRunner(Protocol):
    """Boundary implemented by the ontology semantic tool runner."""

    def run(self, arguments: Sequence[str], output_directory: Path) -> ProcessResult:
        """Run one ROBOT command and return its bounded process result."""


class DockerRobotRunner:
    """Run ROBOT from a pinned Docker image without network access."""

    def __init__(self, project_root: Path, settings: SemanticQaSettings) -> None:
        """Create a runner for one repository and semantic-QA contract."""
        self._project_root = project_root.resolve()
        self._settings = settings

    def run(self, arguments: Sequence[str], output_directory: Path) -> ProcessResult:
        """Run one safe argument-vector command in an ephemeral container."""
        output_directory.mkdir(parents=True, exist_ok=True)
        command = (
            self._settings.container_engine,
            "run",
            "--rm",
            "--network",
            self._settings.network,
            "--memory",
            self._settings.memory,
            "--cpus",
            self._settings.cpus,
            "--mount",
            f"type=bind,source={self._project_root},target=/work,readonly",
            "--mount",
            f"type=bind,source={output_directory.resolve()},target=/out",
            self._settings.image,
            self._settings.command,
            *arguments,
        )
        completed = subprocess.run(  # noqa: S603 - command is a trusted typed argument vector.
            command,
            cwd=self._project_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self._settings.timeout_seconds,
        )
        return ProcessResult(
            arguments=tuple(arguments),
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
