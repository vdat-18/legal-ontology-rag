"""Tests for explicit concept-inventory rerun selections."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.ontology_concept.models import ConceptCandidateInventory, ConceptProposalProgress
from legal_rag.ontology_concept.selection import (
    load_concept_rerun_selection,
    select_concept_rerun_inventories,
)


def _inventories() -> tuple[ConceptCandidateInventory, ...]:
    return (
        ConceptCandidateInventory("role-inventory", "LegalSubjectRole", ("role-1",)),
        ConceptCandidateInventory("vehicle-inventory", "VehicleCategory", ("vehicle-1",)),
    )


def test_load_and_select_reviewed_concept_inventory(tmp_path: Path) -> None:
    """A complete prior run permits replaying only the reviewed inventory."""
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps({"schema_version": "0.1.0", "candidate_ids": ["vehicle-inventory"]}),
        encoding="utf-8",
    )
    inventories = _inventories()
    progress = tuple(ConceptProposalProgress(item.candidate_id, "run") for item in inventories)

    selection = load_concept_rerun_selection(path)

    assert select_concept_rerun_inventories(inventories, progress, selection) == (inventories[1],)


def test_selective_rerun_requires_complete_prior_progress(tmp_path: Path) -> None:
    """Selective replay cannot hide an inventory that has never completed."""
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps({"schema_version": "0.1.0", "candidate_ids": ["vehicle-inventory"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prior progress for every inventory"):
        select_concept_rerun_inventories(_inventories(), (), load_concept_rerun_selection(path))
