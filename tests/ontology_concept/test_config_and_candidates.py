"""Tests for concept-resolution settings and controlled-vocabulary inventories."""

from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_concept.candidates import build_concept_candidates
from legal_rag.ontology_concept.config import ConceptResolutionSettings
from legal_rag.ontology_concept.writer import read_concept_candidates, write_concept_candidates
from legal_rag.ontology_validation.loader import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_concept_contract_and_console_scripts_are_packaged() -> None:
    """The contract and commands expose bounded inventory settings."""
    settings = ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml")
    scripts = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]

    assert settings.included_kinds == ("LegalSubjectRole", "VehicleCategory")
    assert settings.max_occurrences_per_inventory == 10_000
    assert settings.batch_size == 1
    assert set(load_yaml(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml")) == {
        "schema_version",
        "candidates",
        "llm",
    }
    assert scripts["legal-build-concept-candidates"] == "legal_rag.ontology_concept_candidates_cli:main"
    assert scripts["legal-propose-concept-relations"] == "legal_rag.ontology_concept_proposals_cli:main"
    assert scripts["legal-finalize-concept-resolution"] == "legal_rag.ontology_concept_finalize_cli:main"


def test_candidates_are_global_same_kind_deterministic_and_persistent(
    tmp_path: Path,
    exception_corpus: ChangeCorpus,
) -> None:
    """Each reusable kind gets one inventory with one representative per exact label."""
    settings = ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml")
    records = exception_corpus.documents[0].records
    first = build_concept_candidates(records, settings)
    second = build_concept_candidates(tuple(reversed(records)), settings)
    output = tmp_path / "candidates.json"
    write_concept_candidates(output, first)

    assert first == second
    assert len(first.occurrences) == 4
    assert len(first.inventories) == 2
    occurrences_by_id = {item.occurrence_id: item for item in first.occurrences}
    assert all(
        {occurrences_by_id[occurrence_id].kind for occurrence_id in inventory.occurrence_ids} == {inventory.kind}
        for inventory in first.inventories
    )
    assert all(len(inventory.occurrence_ids) == 1 for inventory in first.inventories)
    assert read_concept_candidates(output) == first


def test_candidate_bound_stops_oversized_inventory(exception_corpus: ChangeCorpus) -> None:
    """Candidate construction stops before emitting an oversized inventory."""
    settings = replace(
        ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml"),
        max_occurrences_per_inventory=0,
    )

    with pytest.raises(ValueError, match="exceeds the configured bound"):
        build_concept_candidates(exception_corpus.documents[0].records, settings)


def test_empty_population_has_no_occurrences_or_inventories() -> None:
    """No concept artifact is fabricated without canonical population records."""
    settings = ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml")
    candidates = build_concept_candidates((), settings)

    assert candidates.occurrences == ()
    assert candidates.inventories == ()
