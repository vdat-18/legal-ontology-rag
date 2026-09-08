"""Persistence for target-conduct inventory checkpoints."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from legal_rag.io_utils import write_jsonl_atomic
from legal_rag.ontology_population.conduct_models import (
    TARGET_CONDUCT_INVENTORY_SCHEMA_VERSION,
    TargetConduct,
    TargetConductEvidenceReference,
    TargetConductInventoryRecord,
)
from legal_rag.parser.hashing import sha256_json


def write_target_conduct_inventories(path: Path, records: Sequence[TargetConductInventoryRecord]) -> None:
    """Atomically replace target-conduct inventory JSONL."""
    write_jsonl_atomic(path, (record.to_dict() for record in records))


def read_target_conduct_inventories(path: Path) -> tuple[TargetConductInventoryRecord, ...]:
    """Read typed target-conduct inventory JSONL."""
    records = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != TARGET_CONDUCT_INVENTORY_SCHEMA_VERSION:
                msg = f"Unsupported target-conduct inventory schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(
                TargetConductInventoryRecord(
                    target_chunk_id=payload["target_chunk_id"],
                    conducts=tuple(
                        TargetConduct(
                            conduct_id=item["conduct_id"],
                            conduct_label_vi=item["conduct_label_vi"],
                            provision_path=item["provision_path"],
                            evidence=tuple(
                                TargetConductEvidenceReference(
                                    evidence_id=evidence["evidence_id"],
                                    start_char=evidence["start_char"],
                                    end_char=evidence["end_char"],
                                )
                                for evidence in item["evidence"]
                            ),
                        )
                        for item in payload["conducts"]
                    ),
                    run_id=payload["run_id"],
                )
            )
    return tuple(records)


def target_conduct_inventory_identity(records: Sequence[TargetConductInventoryRecord]) -> str:
    """Fingerprint the exact reviewed inventory consumed by population binding."""
    return sha256_json([record.to_dict() for record in sorted(records, key=lambda item: item.target_chunk_id)])


def validate_target_conduct_inventory(
    records: Sequence[TargetConductInventoryRecord],
    target_chunk_ids: Sequence[str],
    *,
    expected_run_id: str | None,
    require_complete: bool,
    allowed_missing_target_chunk_ids: Sequence[str] = (),
) -> None:
    """Validate only checkpoint identity and target membership."""
    record_ids = [record.target_chunk_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        msg = "Target-conduct inventory contains duplicate target chunk IDs"
        raise ValueError(msg)
    target_ids = set(target_chunk_ids)
    unknown_ids = set(record_ids) - target_ids
    if unknown_ids:
        msg = f"Target-conduct inventory contains an unselected chunk: {sorted(unknown_ids)[0]}"
        raise ValueError(msg)
    if expected_run_id is not None and any(record.run_id != expected_run_id for record in records):
        msg = "Target-conduct inventory does not match the current inputs or LLM configuration"
        raise ValueError(msg)
    allowed_missing_ids = set(allowed_missing_target_chunk_ids)
    if not allowed_missing_ids <= target_ids:
        msg = "Allowed missing conduct targets must belong to the current selection"
        raise ValueError(msg)
    if require_complete and (missing_ids := target_ids - set(record_ids) - allowed_missing_ids):
        msg = f"Target-conduct inventory is incomplete; missing chunk: {sorted(missing_ids)[0]}"
        raise ValueError(msg)
