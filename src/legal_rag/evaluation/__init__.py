"""Synthetic dataset generation and evaluation support."""

from __future__ import annotations

from legal_rag.evaluation.groundtruth import build_ragas_generation_components, generate_groundtruth_candidates
from legal_rag.evaluation.schemas import (
    ApprovalStatus,
    CorpusArtifact,
    CorpusSnapshot,
    GroundTruthCandidate,
    ReferenceContext,
)
from legal_rag.evaluation.snapshot import build_corpus_snapshot, write_corpus_snapshot
from legal_rag.evaluation.synthesis import (
    candidates_from_testset,
    prepare_ragas_documents,
    synthesize_single_hop_candidates,
    write_candidates_jsonl,
)

__all__ = [
    "ApprovalStatus",
    "CorpusArtifact",
    "CorpusSnapshot",
    "GroundTruthCandidate",
    "ReferenceContext",
    "build_corpus_snapshot",
    "build_ragas_generation_components",
    "candidates_from_testset",
    "generate_groundtruth_candidates",
    "prepare_ragas_documents",
    "synthesize_single_hop_candidates",
    "write_candidates_jsonl",
    "write_corpus_snapshot",
]
