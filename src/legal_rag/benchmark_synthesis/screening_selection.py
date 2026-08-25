"""Parse explicit chunk selections for targeted screening refreshes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.chunker.models import Chunk


def load_rescreen_chunk_ids(inline_ids: Sequence[str], ids_path: Path | None) -> frozenset[str]:
    """Return validated chunk ids supplied inline or through a JSON string array."""
    chunk_ids = list(inline_ids)
    if ids_path is not None:
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
            msg = f"Re-screen chunk id file must contain a JSON string array: {ids_path}"
            raise ValueError(msg)
        chunk_ids.extend(payload)
    if any(not chunk_id.strip() for chunk_id in chunk_ids):
        msg = "Re-screen chunk ids must be non-empty strings"
        raise ValueError(msg)
    return frozenset(chunk_ids)


def validate_selective_screening(
    chunks: Sequence[Chunk], records: Sequence[ChunkScreeningRecord], requested_ids: frozenset[str]
) -> None:
    """Require a known selection and prior verdicts for every unselected chunk."""
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    unknown_ids = requested_ids - chunk_ids
    if unknown_ids:
        msg = f"Unknown chunk ids requested for re-screening: {sorted(unknown_ids)}"
        raise ValueError(msg)
    existing_ids = {record.chunk_id for record in records}
    if chunk_ids - requested_ids - existing_ids:
        msg = "Selective re-screening requires prior verdicts for every unselected chunk"
        raise ValueError(msg)
