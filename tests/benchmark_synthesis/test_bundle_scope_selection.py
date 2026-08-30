"""Tests for selective bundle-scope confirmation input."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.bundle_scope_selection import (
    load_reconfirm_bundle_ids,
    validate_selective_scope_confirmation,
)


def test_load_reconfirm_bundle_ids_combines_inline_and_file(tmp_path: Path) -> None:
    """Inline and JSON selections form one deduplicated set."""
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps(["bundle-b", "bundle-c"]), encoding="utf-8")

    result = load_reconfirm_bundle_ids(["bundle-a", "bundle-b"], ids_path)

    assert result == frozenset({"bundle-a", "bundle-b", "bundle-c"})


def test_validate_selective_scope_confirmation_requires_complete_prior_state() -> None:
    """Selective replay cannot silently skip an unchecked unselected bundle."""
    with pytest.raises(ValueError, match="prior decisions"):
        validate_selective_scope_confirmation(
            frozenset({"bundle-a", "bundle-b"}),
            frozenset(),
            frozenset({"bundle-a"}),
        )
