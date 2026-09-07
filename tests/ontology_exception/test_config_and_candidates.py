"""Tests for versioned exception settings and structural candidates."""

from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path

from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_exception.candidates import build_exception_candidates
from legal_rag.ontology_exception.config import ExceptionResolutionSettings
from legal_rag.ontology_exception.writer import read_exception_candidates, write_exception_candidates
from legal_rag.ontology_validation.loader import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_exception_contract_and_console_scripts_are_packaged() -> None:
    """The contract and operator commands expose bounded workflow settings."""
    settings = ExceptionResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml")
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert settings.max_source_chunks_per_window == 6
    assert settings.max_general_rules_per_window == 6
    assert settings.batch_size == 1
    assert set(load_yaml(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml")) == {
        "schema_version",
        "candidates",
        "llm",
    }
    assert project["scripts"]["legal-build-exception-candidates"] == "legal_rag.ontology_exception_candidates_cli:main"
    assert project["scripts"]["legal-propose-legal-exceptions"] == "legal_rag.ontology_exception_proposals_cli:main"
    assert project["scripts"]["legal-finalize-legal-exceptions"] == "legal_rag.ontology_exception_finalize_cli:main"


def test_exception_candidates_are_deterministic_source_local_and_persistent(
    tmp_path: Path,
    exception_corpus: ChangeCorpus,
) -> None:
    """Candidate windows pair each rule only with the source batch that contains it."""
    settings = replace(
        ExceptionResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml"),
        max_source_chunks_per_window=1,
    )
    first = build_exception_candidates(exception_corpus, settings)
    second = build_exception_candidates(exception_corpus, settings)
    output_path = tmp_path / "candidates.json"
    write_exception_candidates(output_path, first)

    assert first == second
    assert len(first.windows) == 2
    memberships = {(window.general_population_ids, window.source_chunk_ids) for window in first.windows}
    assert memberships == {
        (("general-population",), ("general-chunk",)),
        (("target-population",), ("target-chunk",)),
    }
    assert read_exception_candidates(output_path) == first


def test_unrelated_source_batches_do_not_multiply_general_rules(exception_corpus: ChangeCorpus) -> None:
    """Each rule appears once when source and general bounds both hold it."""
    settings = replace(
        ExceptionResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml"),
        max_source_chunks_per_window=1,
        max_general_rules_per_window=1,
    )

    candidates = build_exception_candidates(exception_corpus, settings)

    assert sorted(
        population_id for window in candidates.windows for population_id in window.general_population_ids
    ) == [
        "general-population",
        "target-population",
    ]


def test_empty_corpus_has_no_exception_candidates() -> None:
    """No candidate is emitted without canonical rules and source chunks."""
    settings = ExceptionResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml")

    assert build_exception_candidates(ChangeCorpus(documents=()), settings).windows == ()
