"""Validated schemas for synthetic legal evaluation data."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ApprovalStatus(StrEnum):
    """Human review state for a generated candidate."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReferenceContext(BaseModel):
    """One source chunk that supports a generated answer."""

    model_config = ConfigDict(frozen=True)

    context_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_description: str = Field(min_length=1)
    effective_date: date


class GroundTruthCandidate(BaseModel):
    """A machine-generated sample awaiting human legal review."""

    model_config = ConfigDict(frozen=True)

    candidate_id: UUID
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    reference_contexts: tuple[ReferenceContext, ...] = Field(min_length=1)
    synthesizer_name: str = Field(min_length=1)
    persona_name: str | None = None
    query_style: str | None = None
    query_length: str | None = None
    as_of_date: date | None = None
    approval_status: ApprovalStatus = ApprovalStatus.PENDING


class CorpusArtifact(BaseModel):
    """One immutable input artifact recorded in a corpus snapshot."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    sha256: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)


class CorpusSnapshot(BaseModel):
    """Deterministic provenance for a selected legal corpus."""

    model_config = ConfigDict(frozen=True)

    snapshot_id: str = Field(min_length=1)
    archive: CorpusArtifact
    selection: CorpusArtifact
    selected_documents: tuple[str, ...] = Field(min_length=1)
