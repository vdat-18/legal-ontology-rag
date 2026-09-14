"""Export synthesized candidates to the evaluation ground-truth schema."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from legal_rag.benchmark_synthesis.candidate_generation import CandidateDraft
from legal_rag.benchmark_synthesis.multi_turn_candidate_generation import MultiTurnCandidateDraft

GROUND_TRUTH_EXPORT_SCHEMA_VERSION = "5.0.0"
MULTI_TURN_GROUND_TRUTH_EXPORT_SCHEMA_VERSION = "4.0.0"


def export_ground_truth(candidates: Sequence[CandidateDraft]) -> tuple[dict[str, Any], ...]:
    """Map single-turn candidates to evaluation-ready ground-truth records."""
    return tuple(_export_candidate(candidate) for candidate in candidates)


def export_multi_turn_ground_truth(
    conversations: Sequence[MultiTurnCandidateDraft],
) -> tuple[dict[str, Any], ...]:
    """Map multi-turn candidates to evaluation-ready ground-truth records."""
    return tuple(_export_conversation(conversation) for conversation in conversations)


def _export_candidate(candidate: CandidateDraft) -> dict[str, Any]:
    """Return one evaluation record while preserving synthesis metadata."""
    return {
        "schema_version": GROUND_TRUTH_EXPORT_SCHEMA_VERSION,
        "question_id": candidate.sample_id,
        "question": candidate.query,
        "expected_answer": candidate.reference_answer,
        "expected_chunk_ids": list(candidate.gold_chunk_ids),
        "scenario_id": candidate.scenario_id,
        "interaction_type": "single_turn",
        "bundle_kind": candidate.bundle_kind,
        "bridge": candidate.bridge,
        "reasoning": candidate.reasoning,
        "required_claims": [claim.to_dict() for claim in candidate.required_claims],
        "query_form": candidate.query_form,
        "persona": candidate.persona,
        "query_style": candidate.query_style,
        "query_length": candidate.query_length,
        "answerability": candidate.answerability,
        "reasoning_hops": candidate.reasoning_hops,
        "source_fact_ids": list(candidate.source_fact_ids),
        "generator_model": candidate.generator_model,
    }


def _export_conversation(conversation: MultiTurnCandidateDraft) -> dict[str, Any]:
    """Return one evaluation record targeting only the final conversation turn."""
    target = conversation.turns[-1]
    return {
        "schema_version": MULTI_TURN_GROUND_TRUTH_EXPORT_SCHEMA_VERSION,
        "conversation_id": conversation.conversation_id,
        "question_id": f"{conversation.conversation_id}:{target.turn_number}",
        "interaction_type": "multi_turn",
        "persona": conversation.persona,
        "conversation_length": conversation.conversation_length,
        "target_turn_number": target.turn_number,
        "history": [
            {"question": turn.query, "answer": turn.reference_answer}
            for turn in conversation.turns[:-1]
        ],
        "question": target.query,
        "standalone_question": target.standalone_query,
        "expected_answer": target.reference_answer,
        "expected_chunk_ids": list(target.gold_chunk_ids),
        "dialogue_act": target.dialogue_act,
        "context_dependency": target.context_dependency,
        "answerability": target.answerability,
        "missing_information": target.missing_information,
        "evaluation_mode": "retrieval_required",
        "reasoning": target.reasoning,
        "required_claims": [claim.to_dict() for claim in target.required_claims],
        "source_fact_ids": list(target.source_fact_ids),
        "generator_model": conversation.generator_model,
        "prompt_version": conversation.prompt_version,
    }
