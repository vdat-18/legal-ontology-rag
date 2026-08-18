"""Tests for reviewable ontology population persistence."""

from __future__ import annotations

from pathlib import Path

from legal_rag.ontology_population.models import OntologyPopulationRecord, PopulationProgressRecord
from legal_rag.ontology_population.writer import (
    read_population_progress,
    read_population_records,
    write_population_progress,
    write_population_records,
)


def test_population_records_round_trip(
    tmp_path: Path,
    population_record: OntologyPopulationRecord,
) -> None:
    """Reviewable typed population records round-trip through JSONL."""
    path = tmp_path / "population.jsonl"

    write_population_records(path, (population_record,))

    assert read_population_records(path) == (population_record,)


def test_population_progress_records_empty_extractions(tmp_path: Path) -> None:
    """Progress persists target completion even when the LLM returns no rules."""
    path = tmp_path / "progress.jsonl"
    records = (PopulationProgressRecord(chunk_id="chunk-1"),)

    write_population_progress(path, records)

    assert read_population_progress(path) == records
