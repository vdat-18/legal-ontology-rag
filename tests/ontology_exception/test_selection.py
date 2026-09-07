"""Tests for explicit legal-exception candidate rerun selections."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.ontology_exception.models import ExceptionCandidateWindow, ExceptionProposalProgress
from legal_rag.ontology_exception.selection import (
    ExceptionRerunSelection,
    load_exception_rerun_selection,
    select_exception_rerun_candidates,
)


def _candidate(candidate_id: str) -> ExceptionCandidateWindow:
    """Build one compact exception candidate."""
    return ExceptionCandidateWindow(
        candidate_id=candidate_id,
        source_document_id="document",
        general_population_ids=(f"population-{candidate_id}",),
        source_chunk_ids=(f"chunk-{candidate_id}",),
    )


def test_exception_rerun_selection_loads_versioned_candidate_ids(tmp_path: Path) -> None:
    """The selective-rerun boundary accepts a typed candidate string array."""
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps({"schema_version": "0.1.0", "candidate_ids": ["candidate-1"]}),
        encoding="utf-8",
    )

    assert load_exception_rerun_selection(selection_path).candidate_ids == ("candidate-1",)


@pytest.mark.parametrize(
    ("selection", "progress_ids", "message"),
    [
        (ExceptionRerunSelection(("candidate-1", "candidate-1")), ("candidate-1",), "must be unique"),
        (ExceptionRerunSelection(("unknown",)), ("candidate-1",), "unknown candidate"),
        (ExceptionRerunSelection(("candidate-1",)), (), "prior progress for every candidate"),
    ],
)
def test_exception_rerun_selection_rejects_unsafe_replay_state(
    selection: ExceptionRerunSelection,
    progress_ids: tuple[str, ...],
    message: str,
) -> None:
    """Selective replay cannot discard unknown or unfinished decisions."""
    candidates = (_candidate("candidate-1"),)
    progress = tuple(ExceptionProposalProgress(candidate_id=item, run_id="old-run") for item in progress_ids)

    with pytest.raises(ValueError, match=message):
        select_exception_rerun_candidates(candidates, progress, selection)


def test_exception_rerun_selection_returns_only_reviewed_candidates() -> None:
    """A complete prior run permits replaying only the reviewed subset."""
    candidates = (_candidate("candidate-1"), _candidate("candidate-2"))
    progress = tuple(
        ExceptionProposalProgress(candidate_id=candidate.candidate_id, run_id="old-run") for candidate in candidates
    )

    selected = select_exception_rerun_candidates(
        candidates,
        progress,
        ExceptionRerunSelection(("candidate-2",)),
    )

    assert selected == (candidates[1],)
