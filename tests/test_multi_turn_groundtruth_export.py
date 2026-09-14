"""Tests for multi-turn ground-truth export."""

from __future__ import annotations

from legal_rag.benchmark_synthesis.candidate_generation import ClaimEvidence, RequiredClaim
from legal_rag.benchmark_synthesis.groundtruth_export import export_multi_turn_ground_truth
from legal_rag.benchmark_synthesis.multi_turn_candidate_generation import (
    ConversationTurnDraft,
    MultiTurnCandidateDraft,
)
from legal_rag.evaluation.metrics import expand_evaluation_row


def _turn(
    turn_number: int,
    dialogue_act: str,
    *,
    answerability: str = "answerable",
    missing_information: str | None = None,
) -> ConversationTurnDraft:
    """Build one generated turn anchored on one chunk."""
    return ConversationTurnDraft(
        turn_number=turn_number,
        dialogue_act=dialogue_act,
        context_dependency="standalone" if turn_number == 1 else "ellipsis",
        answerability=answerability,
        anchor_scenario_id=f"scenario-{turn_number}",
        query=f"Câu hỏi lượt {turn_number}",
        standalone_query=f"Câu hỏi đầy đủ lượt {turn_number}",
        reference_answer=f"Đáp án lượt {turn_number}",
        reasoning="Theo căn cứ của lượt.",
        missing_information=missing_information,
        required_claims=(
            RequiredClaim(
                claim=f"Ý bắt buộc lượt {turn_number}",
                fact_id=f"fact-{turn_number}",
                evidence=(ClaimEvidence(chunk_id=f"chunk-{turn_number}"),),
            ),
        ),
        gold_chunk_ids=(f"chunk-{turn_number}",),
        source_fact_ids=(f"fact-{turn_number}",),
    )


def test_export_multi_turn_ground_truth_targets_only_final_turn() -> None:
    """Earlier turns become history and only the final turn is evaluated."""
    conversation = MultiTurnCandidateDraft(
        conversation_id="conv-1",
        persona="tai_xe_ca_nhan",
        conversation_length=2,
        turns=(
            _turn(1, "inquiry"),
            _turn(
                2,
                "continuity",
                answerability="partially_answerable",
                missing_information="mức xử lý bổ sung",
            ),
        ),
        gold_chunk_ids=("chunk-1", "chunk-2"),
        generator_model="provider/test-model",
    )

    (record,) = export_multi_turn_ground_truth((conversation,))

    assert record["interaction_type"] == "multi_turn"
    assert record["conversation_id"] == "conv-1"
    assert record["question_id"] == "conv-1:2"
    assert record["target_turn_number"] == 2
    assert record["history"] == [
        {"question": "Câu hỏi lượt 1", "answer": "Đáp án lượt 1"}
    ]
    assert record["question"] == "Câu hỏi lượt 2"
    assert record["standalone_question"] == "Câu hỏi đầy đủ lượt 2"
    assert record["evaluation_mode"] == "retrieval_required"
    assert record["expected_answer"] == "Đáp án lượt 2"
    assert record["expected_chunk_ids"] == ["chunk-2"]
    assert record["dialogue_act"] == "continuity"
    assert record["context_dependency"] == "ellipsis"
    assert record["missing_information"] == "mức xử lý bổ sung"
    assert record["required_claims"][0]["fact_id"] == "fact-2"
    assert expand_evaluation_row(record) == (record,)
