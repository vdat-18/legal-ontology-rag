"""Parse explicit bundle selections for targeted scope reconfirmation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


def load_reconfirm_bundle_ids(inline_ids: Sequence[str], ids_path: Path | None) -> frozenset[str]:
    """Return bundle IDs supplied inline or through a JSON string array."""
    bundle_ids = list(inline_ids)
    if ids_path is not None:
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
            msg = f"Reconfirm bundle id file must contain a JSON string array: {ids_path}"
            raise ValueError(msg)
        bundle_ids.extend(payload)
    if any(not bundle_id.strip() for bundle_id in bundle_ids):
        msg = "Reconfirm bundle IDs must be non-empty strings"
        raise ValueError(msg)
    return frozenset(bundle_ids)


def validate_selective_scope_confirmation(
    bundle_ids: frozenset[str],
    completed_ids: frozenset[str],
    requested_ids: frozenset[str],
) -> None:
    """Require known selections and prior decisions for every unselected bundle."""
    unknown_ids = requested_ids - bundle_ids
    if unknown_ids:
        msg = f"Unknown bundle IDs requested for scope reconfirmation: {sorted(unknown_ids)}"
        raise ValueError(msg)
    if bundle_ids - requested_ids - completed_ids:
        msg = "Selective bundle-scope confirmation requires prior decisions for every unselected bundle"
        raise ValueError(msg)
