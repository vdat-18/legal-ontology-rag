"""Tests for the versioned ontology evolution policy."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

from legal_rag.ontology_validation.evolution_config import OntologyEvolutionSettings


def test_production_policy_pins_conservative_evolution_boundaries() -> None:
    """New data remains instance-only and every known breaking change is fail-closed."""
    project_root = Path(__file__).resolve().parents[2]
    settings = OntologyEvolutionSettings.from_path(project_root / "ontology" / "evolution_policy.yaml")
    scripts = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]

    assert settings.new_data_action == "populate_instances"
    assert settings.llm_tbox_mutation == "prohibited"
    assert settings.meaning_change_action == "mint_new_iri"
    assert settings.breaking_checks == {
        "removed_active_term",
        "changed_term_kind",
        "changed_active_term_semantics",
        "removed_node_shape",
        "changed_node_shape_constraints",
        "removed_mandatory_cq",
    }
    assert scripts["legal-diff-ontology"] == "legal_rag.ontology_evolution_diff_cli:main"


def test_policy_rejects_unused_fields(tmp_path: Path) -> None:
    """Versioned policy fields must have explicit lifecycle or gate semantics."""
    project_root = Path(__file__).resolve().parents[2]
    document = yaml.safe_load((project_root / "ontology" / "evolution_policy.yaml").read_text(encoding="utf-8"))
    document["unused"] = True
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown evolution policy fields"):
        OntologyEvolutionSettings.from_path(policy_path)
