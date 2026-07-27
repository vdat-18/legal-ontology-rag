"""Tests for atomic fact-ledger pipeline persistence."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from legal_rag.benchmark_synthesis.binding_gate import BindingCheckResult, BindingViolation
from legal_rag.benchmark_synthesis.candidate_generation import (
    CandidateDraft,
    ClaimEvidence,
    FactAnswerFields,
    RequiredClaim,
)
from legal_rag.benchmark_synthesis.candidate_record import CandidateRecord
from legal_rag.benchmark_synthesis.evidence_graph import EvidenceRelation
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario
from legal_rag.benchmark_synthesis.verification import VerificationResult
from legal_rag.benchmark_synthesis.writer import (
    read_candidates,
    read_evidence_relations,
    read_legal_facts,
    read_scenarios,
    write_candidates,
    write_evidence_relations,
    write_legal_facts,
    write_scenarios,
)


def _fact() -> LegalFact:
    """Build one legal fact covering every nullable and tuple field."""
    return LegalFact(
        fact_id="fact-1",
        fact_content_sha256="content-1",
        conduct="Vượt đèn đỏ",
        subject="Người điều khiển ô tô",
        conditions=("Ban đêm",),
        monetary_penalty="4.000.000 đồng",
        license_points_deducted=None,
        additional_penalties=(),
        remedial_measures=(),
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6",
        evidence=(
            EvidenceReference(
                chunk_id="chunk-1",
                quoted_span="Phạt tiền từ 4.000.000 đồng.",
                unit_ids=("unit-1",),
                source_refs=("word/document.xml#/body/1",),
            ),
        ),
    )


def test_legal_facts_round_trip_through_jsonl(tmp_path: Path) -> None:
    """A written legal fact reads back with identical field values."""
    output_path = tmp_path / "legal_facts.jsonl"
    fact = _fact()

    write_legal_facts(output_path, [fact])
    facts = read_legal_facts(output_path)

    assert facts == (fact,)


def test_evidence_relations_round_trip_through_jsonl(tmp_path: Path) -> None:
    """A written evidence relation reads back with identical field values."""
    output_path = tmp_path / "evidence_relations.jsonl"
    relation = EvidenceRelation(
        relation_id="relation-1",
        relation_type="amended_by",
        from_fact_id="fact-1",
        to_fact_id="fact-2",
        bridge=("Điều 6 khoản 9 điểm b",),
        confirmed_by_model="provider/test-model",
    )

    write_evidence_relations(output_path, [relation])
    relations = read_evidence_relations(output_path)

    assert relations == (relation,)


def test_scenarios_round_trip_through_jsonl(tmp_path: Path) -> None:
    """A written scenario reads back with identical field values, including its date."""
    output_path = tmp_path / "scenarios.jsonl"
    scenario = Scenario(
        scenario_id="scenario-1",
        query_type="penalty_for_conduct",
        query_form="specific",
        answerability="answerable",
        persona="tai_xe_ca_nhan",
        query_style="no_diacritics",
        query_length="short",
        reasoning_hops=2,
        as_of_date=date(2025, 2, 10),
        fact_ids=("fact-1", "fact-2"),
        bridge=("Điều 6",),
        gold_chunk_ids=("chunk-1", "chunk-2"),
    )

    write_scenarios(output_path, [scenario])
    scenarios = read_scenarios(output_path)

    assert scenarios == (scenario,)


def test_scenarios_round_trip_preserves_null_as_of_date(tmp_path: Path) -> None:
    """A scenario without a temporal anchor keeps as_of_date as None through the round trip."""
    output_path = tmp_path / "scenarios.jsonl"
    scenario = Scenario(
        scenario_id="scenario-1",
        query_type="penalty_for_conduct",
        query_form="specific",
        answerability="answerable",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        reasoning_hops=1,
        as_of_date=None,
        fact_ids=("fact-1",),
        bridge=(),
        gold_chunk_ids=("chunk-1",),
    )

    write_scenarios(output_path, [scenario])
    scenarios = read_scenarios(output_path)

    assert scenarios[0].as_of_date is None


def _candidate_draft() -> CandidateDraft:
    """Build one candidate draft covering every nullable and tuple field."""
    return CandidateDraft(
        sample_id="sample-1",
        scenario_id="scenario-1",
        query_vi="oto vuot den do bi phat bao nhieu",
        reference_answer_vi="Phạt tiền 4.000.000 đồng.",
        reasoning_vi="Theo Điều 6.",
        required_claims=(
            RequiredClaim(
                claim="Mức phạt tiền áp dụng",
                fact_id="fact-1",
                evidence=(ClaimEvidence(chunk_id="chunk-1", quoted_span="Phạt tiền 4.000.000 đồng."),),
            ),
        ),
        answer_fields=(
            FactAnswerFields(
                fact_id="fact-1",
                monetary_penalty="4.000.000 đồng",
                license_points_deducted=None,
                additional_penalties=(),
                remedial_measures=(),
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
        as_of_date=date(2025, 2, 10),
        source_fact_ids=("fact-1",),
        source_fact_content_sha256=("content-fact-1",),
        generator_model="provider/generator-model",
    )


def test_candidates_round_trip_with_full_verification(tmp_path: Path) -> None:
    """A candidate that passed both gates reads back identically, including verification."""
    output_path = tmp_path / "single_turn.jsonl"
    record = CandidateRecord(
        draft=_candidate_draft(),
        binding=BindingCheckResult(passed=True, grounding_score=1.0, violations=()),
        verification=VerificationResult(
            passed=True,
            is_standalone=True,
            is_specific=True,
            is_question_grounded=True,
            is_answer_grounded=True,
            is_domain_relevant=True,
            is_temporal_correct=True,
            necessary_reasoning_hops=1,
            observed_query_type="penalty_for_conduct",
            observed_answerability="answerable",
            reasoning="Đầy đủ căn cứ.",
            verifier_model="provider/verifier-model",
        ),
    )

    write_candidates(output_path, [record])
    records = read_candidates(output_path)

    assert records == (record,)
    assert records[0].review_status == "automatically_verified"


def test_candidates_round_trip_rejected_by_binding_gate_has_no_verification(tmp_path: Path) -> None:
    """A candidate rejected at gate 1 never reaches verification; None survives the round trip."""
    output_path = tmp_path / "single_turn.jsonl"
    record = CandidateRecord(
        draft=_candidate_draft(),
        binding=BindingCheckResult(
            passed=False,
            grounding_score=0.4,
            violations=(BindingViolation(claim="Mức phạt tiền áp dụng", reason="span below threshold"),),
        ),
        verification=None,
    )

    write_candidates(output_path, [record])
    records = read_candidates(output_path)

    assert records == (record,)
    assert records[0].verification is None
    assert records[0].review_status == "rejected_by_binding_gate"
