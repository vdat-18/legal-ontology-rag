"""Parse and validate explicit scenario replacement selections."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


def load_replace_scenario_ids(inline_ids: Sequence[str], ids_path: Path | None) -> frozenset[str]:
    """Return scenario IDs supplied inline or through a JSON string array."""
    scenario_ids = list(inline_ids)
    if ids_path is not None:
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
            msg = f"Replace scenario id file must contain a JSON string array: {ids_path}"
            raise ValueError(msg)
        scenario_ids.extend(payload)
    if any(not scenario_id.strip() for scenario_id in scenario_ids):
        msg = "Replace scenario IDs must be non-empty strings"
        raise ValueError(msg)
    return frozenset(scenario_ids)
