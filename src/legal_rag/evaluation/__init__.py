"""Synthetic dataset generation and evaluation support."""

from __future__ import annotations

from legal_rag.evaluation.schemas import ApprovalStatus, GroundTruthCandidate, ReferenceContext
from legal_rag.evaluation.synthesis import (
    candidates_from_testset,
    prepare_ragas_documents,
    synthesize_single_hop_candidates,
    write_candidates_jsonl,
)

__all__ = [
    "ApprovalStatus",
    "GroundTruthCandidate",
    "ReferenceContext",
    "candidates_from_testset",
    "prepare_ragas_documents",
    "synthesize_single_hop_candidates",
    "write_candidates_jsonl",
]
