"""Persistence for ontology population records and progress."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path

from legal_rag.io_utils import write_jsonl_atomic
from legal_rag.ontology_population.models import (
    ONTOLOGY_POPULATION_PROGRESS_SCHEMA_VERSION,
    ONTOLOGY_POPULATION_SCHEMA_VERSION,
    OntologyPopulationRecord,
    PopulationCondition,
    PopulationConsequence,
    PopulationEvidence,
    PopulationProgressRecord,
)


def write_population_records(path: Path, records: Sequence[OntologyPopulationRecord]) -> None:
    """Atomically replace ontology population JSONL records."""
    write_jsonl_atomic(path, (record.to_dict() for record in records))


def read_population_records(path: Path) -> tuple[OntologyPopulationRecord, ...]:
    """Read ontology population JSONL records."""
    records: list[OntologyPopulationRecord] = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != ONTOLOGY_POPULATION_SCHEMA_VERSION:
                msg = f"Unsupported ontology population schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(population_record_from_dict(payload))
    return tuple(records)


def write_population_progress(path: Path, records: Sequence[PopulationProgressRecord]) -> None:
    """Atomically replace ontology population progress JSONL."""
    write_jsonl_atomic(path, (record.to_dict() for record in records))


def read_population_progress(path: Path) -> tuple[PopulationProgressRecord, ...]:
    """Read ontology population progress JSONL."""
    records: list[PopulationProgressRecord] = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != ONTOLOGY_POPULATION_PROGRESS_SCHEMA_VERSION:
                msg = f"Unsupported ontology population progress schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(PopulationProgressRecord(chunk_id=payload["chunk_id"]))
    return tuple(records)


def population_record_from_dict(payload: dict[str, object]) -> OntologyPopulationRecord:
    """Build one typed population record from trusted persisted data."""
    return OntologyPopulationRecord(
        population_id=payload["population_id"],
        source_chunk_id=payload["source_chunk_id"],
        conduct_label_vi=payload["conduct_label_vi"],
        subject_kind=payload["subject_kind"],
        role_labels_vi=tuple(payload["role_labels_vi"]),
        vehicle_category_labels_vi=tuple(payload["vehicle_category_labels_vi"]),
        condition_root_id=payload["condition_root_id"],
        conditions=tuple(
            PopulationCondition(
                condition_id=item["condition_id"],
                kind=item["kind"],
                text_vi=item["text_vi"],
                operand_ids=tuple(item["operand_ids"]),
                evidence=_parse_evidence(item["evidence"]),
            )
            for item in payload["conditions"]
        ),
        consequences=tuple(
            PopulationConsequence(
                consequence_id=item["consequence_id"],
                kind=item["kind"],
                minimum_amount=Decimal(item["minimum_amount"]) if item["minimum_amount"] is not None else None,
                maximum_amount=Decimal(item["maximum_amount"]) if item["maximum_amount"] is not None else None,
                currency_code=item["currency_code"],
                point_value=item["point_value"],
                target_label_vi=item["target_label_vi"],
                responsible_role_label_vi=item["responsible_role_label_vi"],
                evidence=_parse_evidence(item["evidence"]),
                consequence_label_vi=item.get("consequence_label_vi"),
                minimum_duration_months=item.get("minimum_duration_months"),
                maximum_duration_months=item.get("maximum_duration_months"),
            )
            for item in payload["consequences"]
        ),
        valid_from=date.fromisoformat(payload["valid_from"]) if payload["valid_from"] else None,
        valid_to=date.fromisoformat(payload["valid_to"]) if payload["valid_to"] else None,
        document_number=payload["document_number"],
        provision_path=payload["provision_path"],
        evidence=_parse_evidence(payload["evidence"]),
    )


def _parse_evidence(entries: list[dict[str, str]]) -> tuple[PopulationEvidence, ...]:
    """Parse persisted evidence references."""
    return tuple(PopulationEvidence(chunk_id=item["chunk_id"], exact_text=item["exact_text"]) for item in entries)
