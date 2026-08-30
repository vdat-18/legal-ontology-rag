"""Tests for selective candidate-generation task selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.candidate_selection import (
    load_regenerate_task_ids,
    validate_selective_generation,
)


def test_load_regenerate_task_ids_combines_inline_and_file(tmp_path: Path) -> None:
    """Inline and file selections share one deduplicated task set."""
    ids_path = tmp_path / "task_ids.json"
    ids_path.write_text(json.dumps(["scenario-b:0", "scenario-a:0"]), encoding="utf-8")

    task_ids = load_regenerate_task_ids(["scenario-a:0"], ids_path)

    assert task_ids == frozenset({"scenario-a:0", "scenario-b:0"})


def test_validate_selective_generation_requires_complete_unselected_progress() -> None:
    """A selective rerun cannot silently omit an unrelated task."""
    with pytest.raises(ValueError, match="prior progress"):
        validate_selective_generation(
            frozenset({"scenario-a:0", "scenario-b:0"}),
            frozenset(),
            frozenset({"scenario-a:0"}),
        )
