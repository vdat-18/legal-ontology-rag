"""Tests for bounded structural assembly candidates."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.assembly_candidates import build_assembly_candidates
from legal_rag.ontology_population.assembly_config import AssemblySettings
from legal_rag.ontology_population.assembly_writer import read_assembly_candidates, write_assembly_candidates
from legal_rag.ontology_population.models import OntologyPopulationRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_candidates_are_bounded_deterministic_and_round_trip(
    tmp_path: Path,
    assembly_chunks: tuple[Chunk, Chunk],
    assembly_records: tuple[OntologyPopulationRecord, ...],
) -> None:
    """Candidate generation stays structural, bounded, stable, and persistent."""
    settings = AssemblySettings.from_path(PROJECT_ROOT / "ontology" / "assembly_contract.yaml")
    first = build_assembly_candidates(assembly_records, assembly_chunks, settings)
    second = build_assembly_candidates(tuple(reversed(assembly_records)), tuple(reversed(assembly_chunks)), settings)
    output_path = tmp_path / "candidates.json"
    write_assembly_candidates(output_path, first)

    assert first == second
    assert len(first.windows) == 1
    assert first.windows[0].population_ids == ("population-money", "population-points")
    assert read_assembly_candidates(output_path) == first


def test_candidates_do_not_cross_span_document_or_chunk_scope(
    assembly_chunks: tuple[Chunk, Chunk],
    assembly_records: tuple[OntologyPopulationRecord, ...],
) -> None:
    """Structural windows never bridge distant chunks, documents, or independent scopes."""
    settings = replace(
        AssemblySettings.from_path(PROJECT_ROOT / "ontology" / "assembly_contract.yaml"),
        max_chunk_span=0,
    )
    assert not build_assembly_candidates(assembly_records, assembly_chunks, settings).windows

    second_chunk = replace(assembly_chunks[1], scope_id="scope-2")
    assert not build_assembly_candidates(
        assembly_records, (assembly_chunks[0], second_chunk), replace(settings, max_chunk_span=4)
    ).windows

    second_record = replace(assembly_records[1], document_number="15/2025/TT-BCA")
    assert not build_assembly_candidates(
        (assembly_records[0], second_record), assembly_chunks, replace(settings, max_chunk_span=4)
    ).windows


def test_candidate_window_honors_record_limit(
    assembly_chunks: tuple[Chunk, Chunk],
    assembly_records: tuple[OntologyPopulationRecord, ...],
) -> None:
    """Candidate windows cannot exceed the configured model context bound."""
    settings = replace(
        AssemblySettings.from_path(PROJECT_ROOT / "ontology" / "assembly_contract.yaml"),
        max_records_per_window=2,
    )
    third = replace(assembly_records[1], population_id="population-third")
    candidates = build_assembly_candidates((*assembly_records, third), assembly_chunks, settings)

    assert all(len(window.population_ids) <= 2 for window in candidates.windows)
