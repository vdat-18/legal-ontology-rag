"""Parse explicit candidate selections for legal-change reruns."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_change.models import ChangeCandidateWindow, ChangeProposalProgress

CHANGE_RERUN_SELECTION_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class ChangeRerunSelection:
    """Reviewed candidate IDs selected for one legal-change replay."""

    candidate_ids: tuple[str, ...]


def load_change_rerun_selection(path: Path) -> ChangeRerunSelection:
    """Load one versioned legal-change rerun selection."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != CHANGE_RERUN_SELECTION_SCHEMA_VERSION:
        msg = f"Unsupported legal-change rerun selection schema: {payload['schema_version']}"
        raise ValueError(msg)
    candidate_ids = payload["candidate_ids"]
    if not isinstance(candidate_ids, list) or not all(isinstance(value, str) for value in candidate_ids):
        msg = "Legal-change rerun selection candidate_ids must be a JSON string array"
        raise ValueError(msg)
    return ChangeRerunSelection(candidate_ids=tuple(candidate_ids))


def select_change_rerun_candidates(
    candidates: Sequence[ChangeCandidateWindow],
    progress: Sequence[ChangeProposalProgress],
    selection: ChangeRerunSelection,
) -> tuple[ChangeCandidateWindow, ...]:
    """Require a complete prior run and return explicitly selected candidates."""
    if not selection.candidate_ids:
        msg = "Legal-change rerun selection must contain at least one candidate ID"
        raise ValueError(msg)
    if len(set(selection.candidate_ids)) != len(selection.candidate_ids):
        msg = "Legal-change rerun candidate IDs must be unique"
        raise ValueError(msg)

    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    unknown_ids = set(selection.candidate_ids) - set(candidates_by_id)
    if unknown_ids:
        msg = f"Legal-change rerun selection contains an unknown candidate: {sorted(unknown_ids)[0]}"
        raise ValueError(msg)

    progress_ids = [item.candidate_id for item in progress]
    if len(set(progress_ids)) != len(progress_ids) or set(progress_ids) != set(candidates_by_id):
        msg = "Selective legal-change rerun requires prior progress for every candidate"
        raise ValueError(msg)
    selected_ids = set(selection.candidate_ids)
    return tuple(candidate for candidate in candidates if candidate.candidate_id in selected_ids)
