"""Tests for reviewable ontology population persistence."""

from __future__ import annotations

import json
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


def test_population_reader_accepts_records_before_duration_fields(
    tmp_path: Path,
    population_record: OntologyPopulationRecord,
) -> None:
    """Optional V0.9 consequence fields preserve existing population JSONL."""
    path = tmp_path / "population.jsonl"
    payload = population_record.to_dict()
    consequences = payload["consequences"]
    assert isinstance(consequences, list)
    for consequence in consequences:
        assert isinstance(consequence, dict)
        consequence.pop("consequence_label_vi")
        consequence.pop("minimum_duration_months")
        consequence.pop("maximum_duration_months")
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    record = read_population_records(path)[0]

    assert all(item.consequence_label_vi is None for item in record.consequences)
    assert all(item.minimum_duration_months is None for item in record.consequences)
    assert all(item.maximum_duration_months is None for item in record.consequences)


def test_population_progress_records_empty_extractions(tmp_path: Path) -> None:
    """Progress persists target completion even when the LLM returns no rules."""
    path = tmp_path / "progress.jsonl"
    records = (PopulationProgressRecord(chunk_id="chunk-1"),)

    write_population_progress(path, records)

    assert read_population_progress(path) == records
