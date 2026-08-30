"""Tests for integrated one-LLM semantic step orchestration."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from legal_rag import ontology_semantic_step_cli


@pytest.mark.parametrize(
    ("main", "argument_count", "expected_modules"),
    [
        (
            ontology_semantic_step_cli.assemble_main,
            10,
            (
                "legal_rag.ontology_assembly_candidates_cli",
                "legal_rag.ontology_assembly_proposals_cli",
                "legal_rag.ontology_assembly_finalize_cli",
            ),
        ),
        (
            ontology_semantic_step_cli.changes_main,
            9,
            (
                "legal_rag.ontology_change_candidates_cli",
                "legal_rag.ontology_change_proposals_cli",
                "legal_rag.ontology_change_finalize_cli",
            ),
        ),
        (
            ontology_semantic_step_cli.exceptions_main,
            10,
            (
                "legal_rag.ontology_exception_candidates_cli",
                "legal_rag.ontology_exception_proposals_cli",
                "legal_rag.ontology_exception_finalize_cli",
            ),
        ),
        (
            ontology_semantic_step_cli.concepts_main,
            12,
            (
                "legal_rag.ontology_concept_candidates_cli",
                "legal_rag.ontology_concept_proposals_cli",
                "legal_rag.ontology_concept_finalize_cli",
            ),
        ),
    ],
)
def test_integrated_step_runs_one_proposal_component(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    main: Callable[[], None],
    argument_count: int,
    expected_modules: tuple[str, str, str],
) -> None:
    """Each user-facing step wraps preparation, one LLM proposal, and finalization."""
    commands: list[list[str]] = []

    def record_command(command: list[str], *, check: bool) -> None:
        """Capture structured child-process calls without executing them."""
        assert check is True
        commands.append(command)

    monkeypatch.setattr(ontology_semantic_step_cli.subprocess, "run", record_command)
    arguments = [str(tmp_path / f"artifact-{index}") for index in range(argument_count)]
    monkeypatch.setattr(sys, "argv", ["ontology-step", *arguments, "--max-concurrency", "3"])

    main()

    assert [command[2] for command in commands] == list(expected_modules)
    assert sum("_proposals_cli" in command[2] for command in commands) == 1
    assert all(command[:2] == [sys.executable, "-m"] for command in commands)
    proposal_command = next(command for command in commands if "_proposals_cli" in command[2])
    assert proposal_command[-2:] == ["--max-concurrency", "3"]
    assert all("--max-concurrency" not in command for command in commands if command is not proposal_command)
