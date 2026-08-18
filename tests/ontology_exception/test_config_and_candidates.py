"""Tests for versioned exception settings and structural candidates."""

from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_exception.candidates import build_exception_candidates
from legal_rag.ontology_exception.config import ExceptionResolutionSettings
from legal_rag.ontology_exception.writer import read_exception_candidates, write_exception_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_exception_contract_and_console_scripts_are_packaged() -> None:
    """The contract and all operator commands expose bounded mandatory-review settings."""
    settings = ExceptionResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml")
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert settings.max_target_rules_per_window == 64
    assert settings.max_total_windows == 10_000
    assert settings.require_every_proposal_reviewed is True
    assert project["scripts"]["legal-build-exception-candidates"] == "legal_rag.ontology_exception_candidates_cli:main"
    assert project["scripts"]["legal-propose-legal-exceptions"] == "legal_rag.ontology_exception_proposals_cli:main"
    assert project["scripts"]["legal-finalize-legal-exceptions"] == "legal_rag.ontology_exception_finalize_cli:main"


def test_exception_candidates_are_deterministic_bounded_and_persistent(
    tmp_path: Path,
    exception_corpus: ChangeCorpus,
) -> None:
    """Candidate windows preserve source/general/target membership without semantic classification."""
    settings = ExceptionResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml")
    first = build_exception_candidates(exception_corpus, settings)
    second = build_exception_candidates(exception_corpus, settings)
    output_path = tmp_path / "candidates.json"
    write_exception_candidates(output_path, first)

    assert first == second
    assert len(first.windows) == 1
    assert first.windows[0].general_population_ids == ("general-population", "target-population")
    assert first.windows[0].target_population_ids == ("general-population", "target-population")
    assert first.windows[0].source_chunk_ids == ("general-chunk", "target-chunk")
    assert read_exception_candidates(output_path) == first


def test_exception_candidates_enforce_total_window_bound(exception_corpus: ChangeCorpus) -> None:
    """Exhaustive structural retrieval stops before creating an unbounded workload."""
    settings = replace(
        ExceptionResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml"),
        max_total_windows=0,
    )

    with pytest.raises(ValueError, match="exceeds the configured bound"):
        build_exception_candidates(exception_corpus, settings)


def test_empty_corpus_has_no_exception_candidates() -> None:
    """No candidate is emitted without canonical rules and source chunks."""
    settings = ExceptionResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml")

    assert build_exception_candidates(ChangeCorpus(documents=()), settings).windows == ()
