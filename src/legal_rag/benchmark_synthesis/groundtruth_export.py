"""Export synthesized candidates to the evaluation ground-truth schema."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from legal_rag.benchmark_synthesis.candidate_generation import CandidateDraft
from legal_rag.benchmark_synthesis.multi_turn_candidate_generation import MultiTurnCandidateDraft

GROUND_TRUTH_EXPORT_SCHEMA_VERSION = "5.0.0"
MULTI_TURN_GROUND_TRUTH_EXPORT_SCHEMA_VERSION = "1.0.0"


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
    """Return one evaluation record keeping turn-level evidence for each question."""
    return {
        "schema_version": MULTI_TURN_GROUND_TRUTH_EXPORT_SCHEMA_VERSION,
        "conversation_id": conversation.conversation_id,
        "interaction_type": "multi_turn",
        "persona": conversation.persona,
        "conversation_length": conversation.conversation_length,
        "expected_chunk_ids": list(conversation.gold_chunk_ids),
        "turns": [_export_turn(conversation, turn) for turn in conversation.turns],
        "generator_model": conversation.generator_model,
        "prompt_version": conversation.prompt_version,
    }


def _export_turn(conversation: MultiTurnCandidateDraft, turn: Any) -> dict[str, Any]:
    """Return one turn-level evaluation question inside a conversation record."""
    return {
        "turn_number": turn.turn_number,
        "question_id": f"{conversation.conversation_id}:{turn.turn_number}",
        "question": turn.query,
        "expected_answer": turn.reference_answer,
        "expected_chunk_ids": list(turn.gold_chunk_ids),
        "dialogue_act": turn.dialogue_act,
        "context_dependency": turn.context_dependency,
        "answerability": turn.answerability,
        "required_claims": [claim.to_dict() for claim in turn.required_claims],
        "source_fact_ids": list(turn.source_fact_ids),
    }
