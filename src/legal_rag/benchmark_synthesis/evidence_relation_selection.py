"""Parse explicit fact-pair selections for targeted relation confirmation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


def load_reconfirm_pairs(
    inline_pair_ids: Sequence[str],
    pair_ids_path: Path | None,
) -> frozenset[tuple[str, str]]:
    """Return canonical fact-id pairs supplied inline or through a JSON string array."""
    pair_ids = list(inline_pair_ids)
    if pair_ids_path is not None:
        payload = json.loads(pair_ids_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
            msg = f"Reconfirm pair id file must contain a JSON string array: {pair_ids_path}"
            raise ValueError(msg)
        pair_ids.extend(payload)
    return frozenset(_parse_pair_id(pair_id) for pair_id in pair_ids)


def _parse_pair_id(pair_id: str) -> tuple[str, str]:
    """Parse one `fact_id_a::fact_id_b` selection into canonical order."""
    parts = pair_id.split("::")
    if len(parts) != 2 or any(not part.strip() for part in parts) or parts[0] == parts[1]:
        msg = f"Invalid evidence relation pair id: {pair_id!r}"
        raise ValueError(msg)
    return tuple(sorted((parts[0], parts[1])))


def validate_selective_confirmation(
    candidate_pairs: frozenset[tuple[str, str]],
    checked_pairs: frozenset[tuple[str, str]],
    requested_pairs: frozenset[tuple[str, str]],
) -> None:
    """Require known selections and prior progress for every unselected candidate pair."""
    unknown_pairs = requested_pairs - candidate_pairs
    if unknown_pairs:
        msg = f"Unknown candidate pairs requested for reconfirmation: {sorted(unknown_pairs)}"
        raise ValueError(msg)
    if candidate_pairs - requested_pairs - checked_pairs:
        msg = "Selective relation confirmation requires prior progress for every unselected candidate pair"
        raise ValueError(msg)
