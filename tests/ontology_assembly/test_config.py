"""Tests for the versioned ontology assembly contract and commands."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from legal_rag.ontology_population.assembly_config import AssemblySettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_assembly_contract_loads_review_and_execution_bounds() -> None:
    """The checked-in contract exposes bounded candidates, LLM calls, and review gates."""
    settings = AssemblySettings.from_path(PROJECT_ROOT / "ontology" / "assembly_contract.yaml")

    assert settings.max_chunk_span == 4
    assert settings.max_records_per_window == 8
    assert settings.batch_size == 1
    assert settings.max_response_bytes == 1_048_576
    assert settings.require_every_proposal_reviewed is True


def test_assembly_contract_rejects_unknown_schema(tmp_path: Path) -> None:
    """An unknown assembly schema cannot silently change pipeline semantics."""
    contract_path = tmp_path / "assembly.yaml"
    contract_path.write_text('schema_version: "9.0.0"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported ontology assembly contract"):
        AssemblySettings.from_path(contract_path)


def test_assembly_console_scripts_are_packaged() -> None:
    """All three operator stages remain available from the installed project."""
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["scripts"]["legal-build-assembly-candidates"] == "legal_rag.ontology_assembly_candidates_cli:main"
    assert project["scripts"]["legal-propose-rule-assemblies"] == "legal_rag.ontology_assembly_proposals_cli:main"
    assert project["scripts"]["legal-finalize-rule-assemblies"] == "legal_rag.ontology_assembly_finalize_cli:main"
