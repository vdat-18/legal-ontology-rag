"""Tests for exporting synthesized candidates as evaluation ground truth."""

from __future__ import annotations

from datetime import date

from legal_rag.benchmark_synthesis.candidate_generation import (
    CandidateDraft,
    ClaimEvidence,
    RequiredClaim,
)
from legal_rag.benchmark_synthesis.groundtruth_export import export_ground_truth
from legal_rag.evaluation.metrics import EvaluationSample


def _candidate() -> CandidateDraft:
    """Build one representative synthesized candidate."""
    return CandidateDraft(
        sample_id="sample-1",
        scenario_id="scenario-1",
        query_vi="Xe máy vượt đèn đỏ bị phạt bao nhiêu?",
        reference_answer_vi="Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.",
        reasoning_vi="Áp dụng mức phạt tại Điều 7.",
        required_claims=(
            RequiredClaim(
                claim="Mức phạt tiền",
                fact_id="fact-1",
                requested_chunk_ids=("chunk-1",),
                evidence=(ClaimEvidence(chunk_id="chunk-1"),),
            ),
        ),
        evolutions=("concretizing",),
        gold_chunk_ids=("chunk-1",),
        query_type="penalty_for_conduct",
        query_form="specific",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        answerability="answerable",
        reasoning_hops=1,
        as_of_date=date(2025, 1, 1),
        source_fact_ids=("fact-1",),
        generator_model="provider/model",
        bridge="Cùng hành vi vượt đèn đỏ",
    )


def test_export_ground_truth_maps_evaluation_fields_and_preserves_metadata() -> None:
    """Exported records are directly readable by the evaluation schema."""
    candidate = _candidate()

    record = export_ground_truth((candidate,))[0]
    sample = EvaluationSample.from_mapping(record)

    assert sample == EvaluationSample(
        question=candidate.query_vi,
        expected_answer=candidate.reference_answer_vi,
        expected_chunk_ids=candidate.gold_chunk_ids,
    )
    assert record["question_id"] == candidate.sample_id
    assert record["answerability"] == candidate.answerability
    assert record["required_claims"] == [candidate.required_claims[0].to_dict()]
    assert record["as_of_date"] == "2025-01-01"
    assert record["bridge"] == "Cùng hành vi vượt đèn đỏ"
