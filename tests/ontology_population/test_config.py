"""Tests for the versioned ontology population contract."""

from __future__ import annotations

import tomllib
from pathlib import Path

from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_validation.loader import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "ontology" / "population_contract.yaml"
PROMPT_PATH = PROJECT_ROOT / "baml_src" / "ontology_population.baml"


def test_population_contract_loads_typed_execution_limits() -> None:
    """The committed population contract provides bounded execution settings."""
    settings = PopulationSettings.from_path(CONTRACT_PATH)

    assert settings.instance_namespace == "https://w3id.org/legal-ontology-rag/data/"
    assert settings.batch_size == 1
    assert settings.max_preceding_chunks == 2
    assert settings.max_following_chunks == 1
    assert settings.max_article_context_chunks == 32
    assert settings.max_output_tokens == 32768


def test_population_contract_uses_minimal_schema() -> None:
    """The population contract retains only values consumed by runtime code."""
    contract = load_yaml(CONTRACT_PATH)

    assert set(contract) == {
        "schema_version",
        "ontology_version",
        "instance_namespace",
        "reading_window",
        "llm",
    }


def test_population_prompt_preserves_core_semantic_boundaries() -> None:
    """The typed prompt keeps the few semantic boundaries required by the ontology."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    compact_prompt = " ".join(prompt.split())

    assert "Không dùng kiến thức ngoài" in prompt
    assert "Không thêm, bỏ, tách, gộp" in prompt
    assert "condition tree" in prompt
    assert "Không tạo LegalChange" in compact_prompt


def test_population_prompt_preserves_vietnamese_legal_structure() -> None:
    """Vietnamese locators and coordinated conduct remain source-faithful."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    compact_prompt = " ".join(prompt.split())

    assert "closed inventory" in prompt
    assert "Chương > Mục > Điều > Khoản > Điểm" in compact_prompt
    assert "Giữ đúng phủ định" in prompt
    assert "Không dùng kiến thức ngoài" in compact_prompt
    assert "function ExtractOntologyTargetConductBatch" in prompt
    assert "function BindOntologyPopulationBatch" in prompt
    assert len(prompt) < 12000


def test_population_console_script_is_published() -> None:
    """The package exposes the ontology population command."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    assert project["scripts"]["legal-populate-ontology"] == "legal_rag.ontology_population_cli:main"
    assert project["scripts"]["legal-check-ontology-population"] == "legal_rag.ontology_population_preflight_cli:main"
    assert (
        project["scripts"]["legal-evaluate-ontology-population"] == "legal_rag.ontology_population_evaluation_cli:main"
    )
