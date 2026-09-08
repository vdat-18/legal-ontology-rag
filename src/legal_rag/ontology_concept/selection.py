"""Parse explicit candidate selections for concept-inventory reruns."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_concept.models import ConceptCandidateInventory, ConceptProposalProgress

CONCEPT_RERUN_SELECTION_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class ConceptRerunSelection:
    """Reviewed inventory IDs selected for one concept replay."""

    candidate_ids: tuple[str, ...]


def load_concept_rerun_selection(path: Path) -> ConceptRerunSelection:
    """Load one versioned concept-inventory rerun selection."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != CONCEPT_RERUN_SELECTION_SCHEMA_VERSION:
        msg = f"Unsupported concept rerun selection schema: {payload['schema_version']}"
        raise ValueError(msg)
    candidate_ids = payload["candidate_ids"]
    if not isinstance(candidate_ids, list) or not all(isinstance(value, str) for value in candidate_ids):
        msg = "Concept rerun selection candidate_ids must be a JSON string array"
        raise ValueError(msg)
    return ConceptRerunSelection(candidate_ids=tuple(candidate_ids))


def select_concept_rerun_inventories(
    inventories: Sequence[ConceptCandidateInventory],
    progress: Sequence[ConceptProposalProgress],
    selection: ConceptRerunSelection,
) -> tuple[ConceptCandidateInventory, ...]:
    """Require a complete prior run and return explicitly selected inventories."""
    if not selection.candidate_ids:
        msg = "Concept rerun selection must contain at least one candidate ID"
        raise ValueError(msg)
    if len(set(selection.candidate_ids)) != len(selection.candidate_ids):
        msg = "Concept rerun candidate IDs must be unique"
        raise ValueError(msg)
    inventories_by_id = {inventory.candidate_id: inventory for inventory in inventories}
    unknown_ids = set(selection.candidate_ids) - set(inventories_by_id)
    if unknown_ids:
        msg = f"Concept rerun selection contains an unknown candidate: {sorted(unknown_ids)[0]}"
        raise ValueError(msg)
    progress_ids = [item.candidate_id for item in progress]
    if len(set(progress_ids)) != len(progress_ids) or set(progress_ids) != set(inventories_by_id):
        msg = "Selective concept rerun requires prior progress for every inventory"
        raise ValueError(msg)
    selected_ids = set(selection.candidate_ids)
    return tuple(inventory for inventory in inventories if inventory.candidate_id in selected_ids)
