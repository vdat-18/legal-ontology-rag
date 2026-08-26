"""Shared validation and replay handling for resumable ontology proposals."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar


class ProposalArtifact(Protocol):
    """Minimum proposal fields required for safe resume."""

    proposal_id: str
    candidate_id: str
    run_id: str


class ProgressArtifact(Protocol):
    """Minimum progress fields required for safe resume."""

    candidate_id: str
    run_id: str


ProposalT = TypeVar("ProposalT", bound=ProposalArtifact)


def load_compatible_proposal_resume_state(
    existing_proposals: Sequence[ProposalT],
    existing_progress: Sequence[ProgressArtifact],
    candidate_ids: set[str],
    run_id: str,
    *,
    workflow_label: str,
    candidate_scope: str,
) -> tuple[dict[str, ProposalT], set[str]]:
    """Validate persisted artifacts and return their indexed resume state."""
    if any(item.run_id != run_id for item in (*existing_proposals, *existing_progress)):
        msg = f"Existing {workflow_label} outputs belong to a different candidate, prompt, or model run"
        raise ValueError(msg)
    processed_ids = {item.candidate_id for item in existing_progress}
    if len(processed_ids) != len(existing_progress):
        msg = f"Existing {workflow_label} progress candidate identifiers must be unique"
        raise ValueError(msg)
    if not processed_ids <= candidate_ids:
        msg = f"Existing {workflow_label} progress references unknown candidate {candidate_scope}"
        raise ValueError(msg)
    if any(item.candidate_id not in candidate_ids for item in existing_proposals):
        msg = f"Existing {workflow_label} proposals reference unknown candidate {candidate_scope}"
        raise ValueError(msg)
    proposals_by_id = {item.proposal_id: item for item in existing_proposals}
    if len(proposals_by_id) != len(existing_proposals):
        msg = f"Existing {workflow_label} proposal identifiers must be unique"
        raise ValueError(msg)
    return proposals_by_id, processed_ids


def replace_replayed_proposals(
    proposals_by_id: dict[str, ProposalT],
    batch_results: Sequence[tuple[str, Sequence[ProposalT]]],
) -> None:
    """Replace every prior proposal owned by candidates replayed in one batch."""
    replayed_candidate_ids = {candidate_id for candidate_id, _ in batch_results}
    stale_proposal_ids = tuple(
        proposal_id
        for proposal_id, proposal in proposals_by_id.items()
        if proposal.candidate_id in replayed_candidate_ids
    )
    for proposal_id in stale_proposal_ids:
        del proposals_by_id[proposal_id]
    for _, proposals in batch_results:
        proposals_by_id.update({proposal.proposal_id: proposal for proposal in proposals})
