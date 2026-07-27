"""Tests for the deterministic evidence-binding gate (gate 1)."""

from __future__ import annotations

from legal_rag.benchmark_synthesis.binding_gate import check_binding
from legal_rag.benchmark_synthesis.candidate_generation import CandidateDraft, ClaimEvidence, RequiredClaim
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.chunker.models import Chunk

_THRESHOLD = 0.85


def _chunk(chunk_id: str, text: str) -> Chunk:
    """Build one minimal chunk for binding-gate tests."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=0,
        kind="paragraph_group",
        source_name="168_2024_ND-CP.docx",
        source_sha256="source-hash",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=("unit-0",),
        source_refs=("word/document.xml#/body/0",),
        sequence_start=0,
        sequence_end=0,
        text=text,
        context_text="",
        embedding_text=text,
        text_token_count=5,
        embedding_token_count=5,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


def _fact(fact_id: str = "fact-1", *, evidence_chunk_id: str = "chunk-1") -> LegalFact:
    """Build one minimal legal fact for binding-gate tests."""
    return LegalFact(
        fact_id=fact_id,
        fact_content_sha256=f"content-{fact_id}",
        conduct="Vượt đèn đỏ",
        subject="Người điều khiển ô tô",
        conditions=(),
        monetary_penalty="4.000.000 đồng",
        license_points_deducted=None,
        additional_penalties=(),
        remedial_measures=(),
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6",
        evidence=(EvidenceReference(chunk_id=evidence_chunk_id, quoted_span=""),),
    )


def _draft(claims: tuple[RequiredClaim, ...]) -> CandidateDraft:
    """Build one minimal candidate draft with the given required claims."""
    return CandidateDraft(
        sample_id="sample-1",
        scenario_id="scenario-1",
        query_vi="oto vuot den do bi phat bao nhieu",
        reference_answer_vi="Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.",
        reasoning_vi="Theo Điều 6, hành vi vượt đèn đỏ bị phạt tiền.",
        required_claims=claims,
        answer_fields=(),
        evolutions=(),
        gold_chunk_ids=("chunk-1",),
        query_type="penalty_for_conduct",
        query_form="specific",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        answerability="answerable",
        reasoning_hops=1,
        as_of_date=None,
        source_fact_ids=("fact-1",),
        source_fact_content_sha256=("content-fact-1",),
        generator_model="provider/test-model",
    )


def test_check_binding_passes_with_exact_matching_span() -> None:
    """An exact quoted span passes with a grounding_score of 1.0."""
    chunk = _chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    fact = _fact()
    claim = RequiredClaim(
        claim="Mức phạt tiền áp dụng",
        fact_id="fact-1",
        evidence=(ClaimEvidence(chunk_id="chunk-1", quoted_span="Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),),
    )

    result = check_binding(_draft((claim,)), {"fact-1": fact}, {"chunk-1": chunk}, span_match_threshold=_THRESHOLD)

    assert result.passed is True
    assert result.grounding_score == 1.0
    assert result.violations == ()


def test_check_binding_rejects_span_below_threshold() -> None:
    """A quoted span far from the chunk text fails with a recorded violation."""
    chunk = _chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    fact = _fact()
    claim = RequiredClaim(
        claim="Mức phạt tiền áp dụng",
        fact_id="fact-1",
        evidence=(ClaimEvidence(chunk_id="chunk-1", quoted_span="Hoàn toàn không liên quan tới đoạn văn bản."),),
    )

    result = check_binding(_draft((claim,)), {"fact-1": fact}, {"chunk-1": chunk}, span_match_threshold=_THRESHOLD)

    assert result.passed is False
    assert any("below threshold" in violation.reason for violation in result.violations)


def test_check_binding_rejects_chunk_outside_fact_evidence_set() -> None:
    """A chunk_id not belonging to the fact's own evidence set is rejected."""
    chunk = _chunk("chunk-2", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    fact = _fact(evidence_chunk_id="chunk-1")
    claim = RequiredClaim(
        claim="Mức phạt tiền áp dụng",
        fact_id="fact-1",
        evidence=(ClaimEvidence(chunk_id="chunk-2", quoted_span="Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),),
    )

    result = check_binding(_draft((claim,)), {"fact-1": fact}, {"chunk-2": chunk}, span_match_threshold=_THRESHOLD)

    assert result.passed is False
    assert any("outside fact" in violation.reason for violation in result.violations)


def test_check_binding_rejects_unknown_fact_id() -> None:
    """A claim citing a fact_id absent from facts_by_id is rejected, not a crash."""
    claim = RequiredClaim(claim="Mức phạt tiền áp dụng", fact_id="fact-missing", evidence=())

    result = check_binding(_draft((claim,)), {}, {}, span_match_threshold=_THRESHOLD)

    assert result.passed is False
    assert any("not found" in violation.reason for violation in result.violations)


def test_check_binding_rejects_claim_with_no_evidence() -> None:
    """A claim with an empty evidence list is rejected."""
    fact = _fact()
    claim = RequiredClaim(claim="Mức phạt tiền áp dụng", fact_id="fact-1", evidence=())

    result = check_binding(_draft((claim,)), {"fact-1": fact}, {}, span_match_threshold=_THRESHOLD)

    assert result.passed is False
    assert any("no evidence" in violation.reason for violation in result.violations)


def test_check_binding_rejects_missing_chunk() -> None:
    """A referenced chunk_id absent from chunks_by_id is rejected, not a crash."""
    fact = _fact()
    claim = RequiredClaim(
        claim="Mức phạt tiền áp dụng",
        fact_id="fact-1",
        evidence=(ClaimEvidence(chunk_id="chunk-1", quoted_span="bất kỳ"),),
    )

    result = check_binding(_draft((claim,)), {"fact-1": fact}, {}, span_match_threshold=_THRESHOLD)

    assert result.passed is False
    assert any("not found" in violation.reason for violation in result.violations)


def test_check_binding_grounding_score_averages_across_all_checked_spans() -> None:
    """grounding_score is the mean partial ratio across every evaluated span."""
    chunk = _chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    fact = LegalFact(
        fact_id="fact-1",
        fact_content_sha256="content-fact-1",
        conduct="Vượt đèn đỏ",
        subject="Người điều khiển ô tô",
        conditions=(),
        monetary_penalty="4.000.000 đồng",
        license_points_deducted="4 điểm",
        additional_penalties=(),
        remedial_measures=(),
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6",
        evidence=(
            EvidenceReference(chunk_id="chunk-1", quoted_span=""),
            EvidenceReference(chunk_id="chunk-2", quoted_span=""),
        ),
    )
    chunk2 = _chunk("chunk-2", "Trừ 4 điểm giấy phép lái xe.")
    claim = RequiredClaim(
        claim="Mức phạt và điểm trừ",
        fact_id="fact-1",
        evidence=(
            ClaimEvidence(chunk_id="chunk-1", quoted_span="Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),
            ClaimEvidence(chunk_id="chunk-2", quoted_span="Trừ 4 điểm giấy phép lái xe."),
        ),
    )

    result = check_binding(
        _draft((claim,)),
        {"fact-1": fact},
        {"chunk-1": chunk, "chunk-2": chunk2},
        span_match_threshold=_THRESHOLD,
    )

    assert result.passed is True
    assert result.grounding_score == 1.0
