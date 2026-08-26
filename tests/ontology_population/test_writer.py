"""Tests for reviewable ontology population persistence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.ontology_population.models import (
    OntologyPopulationRecord,
    PopulationCondition,
    PopulationEvidence,
    PopulationProgressRecord,
    validate_population_condition_tree,
)
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


def test_population_reader_preserves_legacy_records_for_migration(
    tmp_path: Path,
    population_record: OntologyPopulationRecord,
) -> None:
    """Legacy records remain readable but carry no compatible ontology-run provenance."""
    path = tmp_path / "population.jsonl"
    payload = population_record.to_dict()
    payload["schema_version"] = "0.1.0"
    payload.pop("ontology_version")
    payload.pop("ontology_semantic_sha256")
    payload.pop("population_run_id")
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    record = read_population_records(path)[0]

    assert record.ontology_version is None
    assert record.ontology_semantic_sha256 is None
    assert record.population_run_id is None


@pytest.mark.parametrize(
    ("root_id", "conditions", "message"),
    [
        ("missing", (), "root requires at least one"),
        (
            None,
            (PopulationCondition("atomic", "Atomic", "Điều kiện", (), (PopulationEvidence("chunk", "Nguồn"),)),),
            "require exactly one root",
        ),
        (
            "missing",
            (PopulationCondition("atomic", "Atomic", "Điều kiện", (), (PopulationEvidence("chunk", "Nguồn"),)),),
            "root does not exist",
        ),
        (
            "atomic",
            (
                PopulationCondition(
                    "atomic", "Atomic", "Điều kiện", ("child",), (PopulationEvidence("chunk", "Nguồn"),)
                ),
                PopulationCondition("child", "Atomic", "Con", (), (PopulationEvidence("chunk", "Nguồn"),)),
            ),
            "must not have operands",
        ),
        (
            "not",
            (PopulationCondition("not", "Not", "Không", (), (PopulationEvidence("chunk", "Nguồn"),)),),
            "exactly one operand",
        ),
        (
            "all",
            (
                PopulationCondition("all", "AllOf", "Đồng thời", ("child",), (PopulationEvidence("chunk", "Nguồn"),)),
                PopulationCondition("child", "Atomic", "Con", (), (PopulationEvidence("chunk", "Nguồn"),)),
            ),
            "at least two operands",
        ),
        (
            "any",
            (PopulationCondition("any", "AnyOf", "Một trong", (), (PopulationEvidence("chunk", "Nguồn"),)),),
            "at least two operands",
        ),
        ("atomic", (PopulationCondition("atomic", "Atomic", "Điều kiện", (), ()),), "requires direct evidence"),
    ],
)
def test_condition_tree_rejects_invalid_root_arity_and_evidence(
    root_id: str | None,
    conditions: tuple[PopulationCondition, ...],
    message: str,
) -> None:
    """Condition trees fail before invalid structures can reach persistence or RDF."""
    with pytest.raises(ValueError, match=message):
        validate_population_condition_tree(root_id, conditions)


def test_condition_tree_rejects_duplicate_missing_cyclic_and_unreachable_nodes() -> None:
    """Every condition ID belongs to one connected acyclic expression tree."""
    evidence = (PopulationEvidence("chunk", "Nguồn"),)
    duplicate = (
        PopulationCondition("same", "Atomic", "Một", (), evidence),
        PopulationCondition("same", "Atomic", "Hai", (), evidence),
    )
    missing = (PopulationCondition("root", "Not", "Không", ("missing",), evidence),)
    cyclic = (
        PopulationCondition("root", "Not", "Không A", ("child",), evidence),
        PopulationCondition("child", "Not", "Không B", ("root",), evidence),
    )
    unreachable = (
        PopulationCondition("root", "Atomic", "Gốc", (), evidence),
        PopulationCondition("orphan", "Atomic", "Rời", (), evidence),
    )

    for conditions, message in (
        (duplicate, "must be unique"),
        (missing, "operand does not exist"),
        (cyclic, "contains a cycle"),
        (unreachable, "unreachable from root"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_population_condition_tree("same" if conditions is duplicate else "root", conditions)


def test_population_writer_requires_current_ontology_run_identity(
    tmp_path: Path,
    population_record: OntologyPopulationRecord,
) -> None:
    """New JSONL cannot be written without version, semantic hash, and run identity."""
    path = tmp_path / "population.jsonl"
    fields = (
        ("ontology_version", "ontology version"),
        ("ontology_semantic_sha256", "ontology semantic hash"),
        ("population_run_id", "population run ID"),
    )

    for field, message in fields:
        with pytest.raises(ValueError, match=message):
            write_population_records(path, (replace(population_record, **{field: None}),))


def test_population_reader_rejects_persisted_invalid_condition_tree(
    tmp_path: Path,
    population_record: OntologyPopulationRecord,
) -> None:
    """Hand-edited or stale JSONL cannot bypass the deterministic tree validator."""
    path = tmp_path / "population.jsonl"
    payload = population_record.to_dict()
    payload["conditions"][0]["operand_ids"] = ["required"]
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least two operands"):
        read_population_records(path)


def test_population_progress_records_empty_extractions(tmp_path: Path) -> None:
    """Progress persists target completion even when the LLM returns no rules."""
    path = tmp_path / "progress.jsonl"
    records = (PopulationProgressRecord(chunk_id="chunk-1", run_id="population-run"),)

    write_population_progress(path, records)

    assert read_population_progress(path) == records


def test_population_progress_reader_preserves_legacy_checkpoints_for_migration(tmp_path: Path) -> None:
    """Legacy checkpoints remain readable but carry no compatible execution identity."""
    path = tmp_path / "progress.jsonl"
    path.write_text('{"schema_version":"0.1.0","chunk_id":"chunk-1"}\n', encoding="utf-8")

    assert read_population_progress(path) == (PopulationProgressRecord(chunk_id="chunk-1", run_id=None),)
