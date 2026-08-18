"""Tests for the versioned ontology population contract."""

from __future__ import annotations

import tomllib
from pathlib import Path

from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_validation.loader import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "ontology" / "population_contract.yaml"


def test_population_contract_loads_typed_execution_limits() -> None:
    """The committed population contract provides bounded execution settings."""
    settings = PopulationSettings.from_path(CONTRACT_PATH)

    assert settings.instance_namespace == "https://w3id.org/legal-ontology-rag/data/"
    assert settings.batch_size == 1
    assert settings.max_preceding_chunks == 1
    assert settings.max_following_chunks == 1
    assert settings.max_output_tokens == 32768


def test_population_contract_excludes_unsupported_semantic_steps() -> None:
    """Source population does not silently create occurrences or cross-document identity."""
    contract = load_yaml(CONTRACT_PATH)
    excluded = {item["id"] for item in contract["population_scope"]["excluded"]}

    assert excluded == {"occurrence_context", "cross_document_legal_change", "concept_equivalence"}


def test_population_console_script_is_published() -> None:
    """The package exposes the ontology population command."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    assert project["scripts"]["legal-populate-ontology"] == "legal_rag.ontology_population_cli:main"
