"""Export synthesized candidates to the evaluation ground-truth schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from legal_rag.benchmark_synthesis.candidate_generation import CandidateDraft
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.benchmark_synthesis.multi_turn_candidate_generation import MultiTurnCandidateDraft
from legal_rag.temporal import extract_explicit_dates

GROUND_TRUTH_EXPORT_SCHEMA_VERSION = "6.0.0"
MULTI_TURN_GROUND_TRUTH_EXPORT_SCHEMA_VERSION = "6.0.0"


def export_ground_truth(
    candidates: Sequence[CandidateDraft],
    facts_by_id: Mapping[str, LegalFact] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Map single-turn candidates to evaluation-ready ground-truth records."""
    return tuple(_export_candidate(candidate, facts_by_id) for candidate in candidates)


def export_multi_turn_ground_truth(
    conversations: Sequence[MultiTurnCandidateDraft],
    facts_by_id: Mapping[str, LegalFact] | None = None,
    bundle_kind_by_scenario_id: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Map multi-turn candidates to evaluation-ready ground-truth records."""
    if bundle_kind_by_scenario_id is None:
        msg = "Multi-turn export requires bundle kinds keyed by anchor scenario_id"
        raise ValueError(msg)
    return tuple(
        _export_conversation(conversation, facts_by_id, bundle_kind_by_scenario_id) for conversation in conversations
    )


def _export_candidate(
    candidate: CandidateDraft,
    facts_by_id: Mapping[str, LegalFact] | None,
) -> dict[str, Any]:
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
        "query_dates": _query_dates(
            candidate.query,
            candidate.source_fact_ids,
            facts_by_id,
            include_source_dates=candidate.bundle_kind == "temporal",
        ),
        "generator_model": candidate.generator_model,
    }


def _export_conversation(
    conversation: MultiTurnCandidateDraft,
    facts_by_id: Mapping[str, LegalFact] | None,
    bundle_kind_by_scenario_id: Mapping[str, str],
) -> dict[str, Any]:
    """Return one evaluation record targeting only the final conversation turn."""
    target = conversation.turns[-1]
    try:
        bundle_kind = bundle_kind_by_scenario_id[target.anchor_scenario_id]
    except KeyError as error:
        msg = f"Missing bundle kind for target scenario: {target.anchor_scenario_id}"
        raise ValueError(msg) from error
    target_fact_ids = tuple(dict.fromkeys(claim.fact_id for claim in target.required_claims))
    if bundle_kind == "temporal" and facts_by_id is None:
        msg = "Temporal multi-turn export requires legal facts"
        raise ValueError(msg)
    return {
        "schema_version": MULTI_TURN_GROUND_TRUTH_EXPORT_SCHEMA_VERSION,
        "conversation_id": conversation.conversation_id,
        "question_id": f"{conversation.conversation_id}:{target.turn_number}",
        "interaction_type": "multi_turn",
        "bundle_kind": bundle_kind,
        "persona": conversation.persona,
        "conversation_length": conversation.conversation_length,
        "target_turn_number": target.turn_number,
        "history": [{"question": turn.query, "answer": turn.reference_answer} for turn in conversation.turns[:-1]],
        "question": target.query,
        "standalone_question": target.standalone_query or None,
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
        "query_dates": _query_dates(
            target.standalone_query or target.query,
            target_fact_ids,
            facts_by_id,
            include_source_dates=bundle_kind == "temporal",
        ),
        "generator_model": conversation.generator_model,
        "prompt_version": conversation.prompt_version,
    }


def _query_dates(
    question: str,
    source_fact_ids: Sequence[str],
    facts_by_id: Mapping[str, LegalFact] | None,
    *,
    include_source_dates: bool,
) -> list[str]:
    """Return explicit targets plus authoritative version starts for temporal comparisons."""
    dates = set(extract_explicit_dates(question))
    if include_source_dates:
        dates.update(_source_dates(source_fact_ids, facts_by_id))
    return [item.isoformat() for item in sorted(dates)]


def _source_dates(
    source_fact_ids: Sequence[str],
    facts_by_id: Mapping[str, LegalFact] | None,
) -> tuple[date, ...]:
    """Return unique valid-from dates for exactly the authoritative target facts."""
    if facts_by_id is None:
        return ()
    missing = [fact_id for fact_id in source_fact_ids if fact_id not in facts_by_id]
    if missing:
        msg = f"Legal facts are missing source_fact_ids: {', '.join(missing)}"
        raise ValueError(msg)
    return tuple(
        sorted(
            {fact.valid_from for fact_id in source_fact_ids if (fact := facts_by_id[fact_id]).valid_from is not None}
        )
    )
