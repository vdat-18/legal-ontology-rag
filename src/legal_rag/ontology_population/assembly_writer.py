"""Persistence for assembly candidates, proposals, and progress."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from legal_rag.io_utils import write_json_atomic, write_jsonl_atomic
from legal_rag.ontology_population.assembly_models import (
    ASSEMBLY_CANDIDATE_SCHEMA_VERSION,
    ASSEMBLY_PROPOSAL_PROGRESS_SCHEMA_VERSION,
    ASSEMBLY_PROPOSAL_SCHEMA_VERSION,
    ASSEMBLY_RESOLUTION_PROGRESS_SCHEMA_VERSION,
    LEGACY_ASSEMBLY_PROPOSAL_SCHEMA_VERSION,
    AssemblyCandidateSet,
    AssemblyCandidateWindow,
    AssemblyProposal,
    AssemblyProposalProgress,
    AssemblyResolutionProgress,
)
from legal_rag.ontology_population.writer import population_record_from_dict


def write_assembly_candidates(path: Path, candidates: AssemblyCandidateSet) -> None:
    """Atomically replace one candidate-set JSON artifact."""
    write_json_atomic(path, candidates.to_dict())


def read_assembly_candidates(path: Path) -> AssemblyCandidateSet:
    """Read one fingerprinted assembly candidate set."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != ASSEMBLY_CANDIDATE_SCHEMA_VERSION:
        msg = f"Unsupported assembly candidate schema: {payload['schema_version']}"
        raise ValueError(msg)
    return AssemblyCandidateSet(
        run_id=payload["run_id"],
        windows=tuple(
            AssemblyCandidateWindow(
                candidate_id=item["candidate_id"],
                population_ids=tuple(item["population_ids"]),
                source_chunk_ids=tuple(item["source_chunk_ids"]),
                provision_paths=tuple(item["provision_paths"]),
            )
            for item in payload["windows"]
        ),
    )


def write_assembly_proposals(path: Path, proposals: Sequence[AssemblyProposal]) -> None:
    """Atomically replace assembly proposals."""
    write_jsonl_atomic(path, (proposal.to_dict() for proposal in proposals))


def read_assembly_proposals(path: Path) -> tuple[AssemblyProposal, ...]:
    """Read typed assembly proposals."""
    proposals = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] not in {
                LEGACY_ASSEMBLY_PROPOSAL_SCHEMA_VERSION,
                ASSEMBLY_PROPOSAL_SCHEMA_VERSION,
            }:
                msg = f"Unsupported assembly proposal schema: {payload['schema_version']}"
                raise ValueError(msg)
            proposals.append(
                AssemblyProposal(
                    proposal_id=payload["proposal_id"],
                    candidate_id=payload["candidate_id"],
                    member_population_ids=tuple(payload["member_population_ids"]),
                    assembled_record=population_record_from_dict(payload["assembled_record"]),
                    rationale_vi=payload["rationale_vi"],
                    run_id=payload["run_id"],
                )
            )
    return tuple(proposals)


def write_assembly_progress(path: Path, progress: Sequence[AssemblyProposalProgress]) -> None:
    """Atomically replace proposal checkpoints."""
    write_jsonl_atomic(path, (record.to_dict() for record in progress))


def read_assembly_progress(path: Path) -> tuple[AssemblyProposalProgress, ...]:
    """Read proposal checkpoints."""
    progress = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != ASSEMBLY_PROPOSAL_PROGRESS_SCHEMA_VERSION:
                msg = f"Unsupported assembly progress schema: {payload['schema_version']}"
                raise ValueError(msg)
            progress.append(AssemblyProposalProgress(candidate_id=payload["candidate_id"], run_id=payload["run_id"]))
    return tuple(progress)


def write_assembly_resolution_progress(path: Path, progress: Sequence[AssemblyResolutionProgress]) -> None:
    """Atomically replace conflict-resolution checkpoints."""
    write_jsonl_atomic(path, (record.to_dict() for record in progress))


def read_assembly_resolution_progress(path: Path) -> tuple[AssemblyResolutionProgress, ...]:
    """Read conflict-resolution checkpoints."""
    progress = []
    with path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != ASSEMBLY_RESOLUTION_PROGRESS_SCHEMA_VERSION:
                msg = f"Unsupported assembly resolution progress schema: {payload['schema_version']}"
                raise ValueError(msg)
            progress.append(AssemblyResolutionProgress(component_id=payload["component_id"], run_id=payload["run_id"]))
    return tuple(progress)
