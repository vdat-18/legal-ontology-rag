"""Parse explicit candidate selections for targeted multi-hop-3 confirmation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


def load_reconfirm_candidate_ids(
    inline_ids: Sequence[str],
    ids_path: Path | None,
) -> frozenset[str]:
    """Return candidate IDs supplied inline or through a JSON string array."""
    candidate_ids = list(inline_ids)
    if ids_path is not None:
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
            msg = f"Reconfirm candidate id file must contain a JSON string array: {ids_path}"
            raise ValueError(msg)
        candidate_ids.extend(payload)
    if any(not candidate_id.strip() for candidate_id in candidate_ids):
        msg = "Reconfirm candidate IDs must be non-empty strings"
        raise ValueError(msg)
    return frozenset(candidate_ids)


def validate_selective_confirmation(
    candidate_ids: frozenset[str],
    completed_ids: frozenset[str],
    requested_ids: frozenset[str],
) -> None:
    """Require known selections and prior progress for every unselected candidate."""
    unknown_ids = requested_ids - candidate_ids
    if unknown_ids:
        msg = f"Unknown multi-hop-3 candidates requested for reconfirmation: {sorted(unknown_ids)}"
        raise ValueError(msg)
    if candidate_ids - requested_ids - completed_ids:
        msg = "Selective multi-hop-3 confirmation requires prior progress for every unselected candidate"
        raise ValueError(msg)
