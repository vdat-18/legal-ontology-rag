"""Parse explicit chunk selections for targeted fact re-extraction."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from legal_rag.benchmark_synthesis.fact_extraction import ExtractionProgressRecord


def load_reextract_chunk_ids(inline_ids: Sequence[str], ids_path: Path | None) -> frozenset[str]:
    """Return validated chunk ids supplied inline or through a JSON string array."""
    chunk_ids = list(inline_ids)
    if ids_path is not None:
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
            msg = f"Re-extract chunk id file must contain a JSON string array: {ids_path}"
            raise ValueError(msg)
        chunk_ids.extend(payload)
    if any(not chunk_id.strip() for chunk_id in chunk_ids):
        msg = "Re-extract chunk ids must be non-empty strings"
        raise ValueError(msg)
    return frozenset(chunk_ids)


def validate_selective_extraction(
    accepted_chunk_ids: frozenset[str],
    progress: Sequence[ExtractionProgressRecord],
    requested_ids: frozenset[str],
) -> None:
    """Require accepted selections and prior progress for every unselected target."""
    unknown_ids = requested_ids - accepted_chunk_ids
    if unknown_ids:
        msg = f"Unknown or excluded chunk ids requested for re-extraction: {sorted(unknown_ids)}"
        raise ValueError(msg)
    existing_ids = {record.chunk_id for record in progress}
    if accepted_chunk_ids - requested_ids - existing_ids:
        msg = "Selective fact re-extraction requires prior progress for every unselected accepted chunk"
        raise ValueError(msg)
