"""Export synthesized candidates to the evaluation ground-truth schema."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from legal_rag.benchmark_synthesis.candidate_generation import CandidateDraft

GROUND_TRUTH_EXPORT_SCHEMA_VERSION = "2.0.0"


def export_ground_truth(candidates: Sequence[CandidateDraft]) -> tuple[dict[str, Any], ...]:
    """Map single-turn candidates to evaluation-ready ground-truth records."""
    return tuple(_export_candidate(candidate) for candidate in candidates)


def _export_candidate(candidate: CandidateDraft) -> dict[str, Any]:
    """Return one evaluation record while preserving synthesis metadata."""
    return {
        "schema_version": GROUND_TRUTH_EXPORT_SCHEMA_VERSION,
        "question_id": candidate.sample_id,
        "question": candidate.query_vi,
        "expected_answer": candidate.reference_answer_vi,
        "expected_chunk_ids": list(candidate.gold_chunk_ids),
        "scenario_id": candidate.scenario_id,
        "interaction_type": "single_turn",
        "bundle_kind": candidate.bundle_kind,
        "bridge": candidate.bridge,
        "reasoning_vi": candidate.reasoning_vi,
        "required_claims": [claim.to_dict() for claim in candidate.required_claims],
        "evolutions": list(candidate.evolutions),
        "query_type": candidate.query_type,
        "query_form": candidate.query_form,
        "persona": candidate.persona,
        "query_style": candidate.query_style,
        "query_length": candidate.query_length,
        "answerability": candidate.answerability,
        "reasoning_hops": candidate.reasoning_hops,
        "as_of_date": candidate.as_of_date.isoformat() if candidate.as_of_date else None,
        "source_fact_ids": list(candidate.source_fact_ids),
        "generator_model": candidate.generator_model,
    }
