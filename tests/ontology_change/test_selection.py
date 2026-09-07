"""Tests for explicit legal-change candidate rerun selections."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.ontology_change.models import ChangeCandidateWindow, ChangeProposalProgress
from legal_rag.ontology_change.selection import (
    ChangeRerunSelection,
    load_change_rerun_selection,
    select_change_rerun_candidates,
)


def _candidate(candidate_id: str) -> ChangeCandidateWindow:
    """Build one compact legal-change candidate."""
    return ChangeCandidateWindow(
        candidate_id=candidate_id,
        earlier_document_id="earlier",
        later_document_id="later",
        earlier_population_ids=(f"earlier-{candidate_id}",),
        later_population_ids=(f"later-{candidate_id}",),
        change_source_chunk_ids=(f"chunk-{candidate_id}",),
    )


def test_change_rerun_selection_loads_versioned_candidate_ids(tmp_path: Path) -> None:
    """The selective-rerun boundary accepts a typed candidate string array."""
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps({"schema_version": "0.1.0", "candidate_ids": ["candidate-1"]}),
        encoding="utf-8",
    )

    assert load_change_rerun_selection(selection_path).candidate_ids == ("candidate-1",)


@pytest.mark.parametrize(
    ("selection", "progress_ids", "message"),
    [
        (ChangeRerunSelection(("candidate-1", "candidate-1")), ("candidate-1",), "must be unique"),
        (ChangeRerunSelection(("unknown",)), ("candidate-1",), "unknown candidate"),
        (ChangeRerunSelection(("candidate-1",)), (), "prior progress for every candidate"),
    ],
)
def test_change_rerun_selection_rejects_unsafe_replay_state(
    selection: ChangeRerunSelection,
    progress_ids: tuple[str, ...],
    message: str,
) -> None:
    """Selective replay cannot discard unknown or unfinished decisions."""
    candidates = (_candidate("candidate-1"),)
    progress = tuple(ChangeProposalProgress(candidate_id=item, run_id="old-run") for item in progress_ids)

    with pytest.raises(ValueError, match=message):
        select_change_rerun_candidates(candidates, progress, selection)


def test_change_rerun_selection_returns_only_reviewed_candidates() -> None:
    """A complete prior run permits replaying only the reviewed subset."""
    candidates = (_candidate("candidate-1"), _candidate("candidate-2"))
    progress = tuple(
        ChangeProposalProgress(candidate_id=candidate.candidate_id, run_id="old-run") for candidate in candidates
    )

    selected = select_change_rerun_candidates(
        candidates,
        progress,
        ChangeRerunSelection(("candidate-2",)),
    )

    assert selected == (candidates[1],)
