"""Tests for shared ontology proposal resume safeguards."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from legal_rag.ontology_proposal_resume import (
    load_compatible_proposal_resume_state,
    replace_replayed_proposals,
)


@dataclass(frozen=True)
class _Proposal:
    proposal_id: str
    candidate_id: str
    run_id: str


@dataclass(frozen=True)
class _Progress:
    candidate_id: str
    run_id: str


def test_load_compatible_proposal_resume_state_indexes_valid_artifacts() -> None:
    """Compatible proposals and progress are returned as indexed resume state."""
    proposal = _Proposal("proposal-1", "candidate-1", "run-1")
    progress = _Progress("candidate-1", "run-1")

    proposals_by_id, processed_ids = load_compatible_proposal_resume_state(
        (proposal,),
        (progress,),
        {"candidate-1"},
        "run-1",
        workflow_label="test",
        candidate_scope="windows",
    )

    assert proposals_by_id == {"proposal-1": proposal}
    assert processed_ids == {"candidate-1"}


@pytest.mark.parametrize(
    ("proposals", "progress", "match"),
    [
        ((_Proposal("proposal-1", "candidate-1", "old-run"),), (), "different candidate, prompt, or model run"),
        ((), (_Progress("candidate-1", "run-1"), _Progress("candidate-1", "run-1")), "must be unique"),
        ((_Proposal("proposal-1", "unknown", "run-1"),), (), "unknown candidate windows"),
    ],
)
def test_load_compatible_proposal_resume_state_rejects_invalid_artifacts(
    proposals: tuple[_Proposal, ...],
    progress: tuple[_Progress, ...],
    match: str,
) -> None:
    """Persisted state must belong to one current, unambiguous candidate run."""
    with pytest.raises(ValueError, match=match):
        load_compatible_proposal_resume_state(
            proposals,
            progress,
            {"candidate-1"},
            "run-1",
            workflow_label="test",
            candidate_scope="windows",
        )


def test_replace_replayed_proposals_removes_all_stale_candidate_output() -> None:
    """A replay replaces its complete prior proposal set, including with zero results."""
    proposals_by_id = {
        "stale-1": _Proposal("stale-1", "candidate-1", "run-1"),
        "stale-2": _Proposal("stale-2", "candidate-1", "run-1"),
        "kept": _Proposal("kept", "candidate-2", "run-1"),
    }

    replace_replayed_proposals(proposals_by_id, (("candidate-1", ()),))

    assert proposals_by_id == {"kept": _Proposal("kept", "candidate-2", "run-1")}
