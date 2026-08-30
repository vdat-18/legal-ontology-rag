"""Tests for corpus loading and structural legal-change candidates."""

from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.ontology_change.candidates import build_change_candidates
from legal_rag.ontology_change.config import ChangeResolutionSettings
from legal_rag.ontology_change.corpus import ChangeCorpus, ChangeCorpusDocument, read_change_corpus_manifest
from legal_rag.ontology_change.writer import read_change_candidates, write_change_candidates
from legal_rag.ontology_validation.loader import load_yaml

from .conftest import write_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_change_contract_and_console_scripts_are_packaged() -> None:
    """The checked-in contract and operator commands expose bounded workflow settings."""
    settings = ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert settings.max_source_chunks_per_window == 6
    assert settings.max_total_windows == 10_000
    assert set(load_yaml(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")) == {
        "schema_version",
        "candidates",
        "llm",
    }
    assert project["scripts"]["legal-build-change-candidates"] == "legal_rag.ontology_change_candidates_cli:main"
    assert project["scripts"]["legal-propose-legal-changes"] == "legal_rag.ontology_change_proposals_cli:main"
    assert project["scripts"]["legal-finalize-legal-changes"] == "legal_rag.ontology_change_finalize_cli:main"


def test_manifest_and_candidates_are_deterministic_and_round_trip(
    tmp_path: Path,
    change_documents: tuple[ChangeCorpusDocument, ChangeCorpusDocument],
) -> None:
    """Ordered manifests produce stable exhaustive windows without semantic matching."""
    manifest_path = tmp_path / "corpus.json"
    output_path = tmp_path / "candidates.json"
    write_manifest(manifest_path, change_documents)
    corpus = read_change_corpus_manifest(manifest_path)
    settings = ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")
    first = build_change_candidates(corpus, settings)
    second = build_change_candidates(corpus, settings)
    write_change_candidates(output_path, first)

    assert first == second
    assert len(first.windows) == 1
    assert first.windows[0].earlier_population_ids == ("earlier-population",)
    assert first.windows[0].later_population_ids == ("later-population",)
    assert first.windows[0].change_source_chunk_ids == ("later-chunk",)
    assert read_change_candidates(output_path) == first


def test_candidate_generation_uses_only_forward_document_order(
    change_documents: tuple[ChangeCorpusDocument, ChangeCorpusDocument],
) -> None:
    """Only records from an earlier manifest document enter the earlier side of a window."""
    settings = ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")
    corpus = ChangeCorpus(documents=tuple(reversed(change_documents)))
    candidates = build_change_candidates(corpus, settings)

    assert candidates.windows[0].earlier_document_id == "nd-238"
    assert candidates.windows[0].later_document_id == "nd-168"


def test_candidate_generation_enforces_total_window_bound(
    change_corpus: ChangeCorpus,
) -> None:
    """Exhaustive structural retrieval fails before creating an unbounded workload."""
    settings = replace(
        ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml"),
        max_total_windows=0,
    )

    with pytest.raises(ValueError, match="exceeds the configured bound"):
        build_change_candidates(change_corpus, settings)


def test_one_document_corpus_has_no_cross_document_candidates(
    change_documents: tuple[ChangeCorpusDocument, ChangeCorpusDocument],
) -> None:
    """A cross-document phase cannot propose a change with only one document."""
    settings = ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")

    assert build_change_candidates(ChangeCorpus(documents=change_documents[:1]), settings).windows == ()
