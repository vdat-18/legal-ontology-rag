"""Tests for explicit scenario replacement selections."""

from __future__ import annotations

import json
from pathlib import Path

from legal_rag.benchmark_synthesis.scenario_replacement import load_replace_scenario_ids


def test_load_replace_scenario_ids_combines_inline_and_file(tmp_path: Path) -> None:
    """Inline and file selections share one deduplicated scenario set."""
    ids_path = tmp_path / "scenario_ids.json"
    ids_path.write_text(json.dumps(["scenario-b", "scenario-a"]), encoding="utf-8")

    scenario_ids = load_replace_scenario_ids(["scenario-a"], ids_path)

    assert scenario_ids == frozenset({"scenario-a", "scenario-b"})
