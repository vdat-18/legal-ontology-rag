"""Parse and validate explicit candidate-generation rerun selections."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


def load_regenerate_task_ids(inline_ids: Sequence[str], ids_path: Path | None) -> frozenset[str]:
    """Return task IDs supplied inline or through a JSON string array."""
    task_ids = list(inline_ids)
    if ids_path is not None:
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
            msg = f"Regenerate task id file must contain a JSON string array: {ids_path}"
            raise ValueError(msg)
        task_ids.extend(payload)
    if any(not task_id.strip() for task_id in task_ids):
        msg = "Regenerate task IDs must be non-empty strings"
        raise ValueError(msg)
    return frozenset(task_ids)


def validate_selective_generation(
    task_ids: frozenset[str],
    completed_ids: frozenset[str],
    requested_ids: frozenset[str],
) -> None:
    """Require known selections and prior progress for every unselected task."""
    unknown_ids = requested_ids - task_ids
    if unknown_ids:
        msg = f"Unknown candidate-generation tasks requested for regeneration: {sorted(unknown_ids)}"
        raise ValueError(msg)
    if task_ids - requested_ids - completed_ids:
        msg = "Selective candidate generation requires prior progress for every unselected task"
        raise ValueError(msg)
