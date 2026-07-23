"""Canonical hashing helpers for persisted parser records."""

from __future__ import annotations

import hashlib
import json


def sha256_json(value: object) -> str:
    """Hash a value using canonical UTF-8 JSON serialization."""
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
