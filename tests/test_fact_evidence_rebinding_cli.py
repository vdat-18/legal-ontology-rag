"""Tests for persisted legal-fact evidence rebinding."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from legal_rag import fact_evidence_rebinding_cli
from legal_rag.benchmark_synthesis.evidence_binding import bind_evidence_references
from legal_rag.benchmark_synthesis.models import EvidenceReference, build_legal_fact
from legal_rag.benchmark_synthesis.writer import read_legal_facts, write_legal_facts
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks


def _chunk(text: str) -> Chunk:
    """Build one source chunk for rebinding tests."""
    return Chunk(
        chunk_id="chunk-1",
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=0,
        kind="paragraph_group",
        source_name="document.docx",
        source_sha256="source-hash",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=("unit-1",),
        source_refs=("word/document.xml#/body/1",),
        sequence_start=0,
        sequence_end=0,
        text=text,
        context_text="",
        embedding_text=text,
        text_token_count=10,
        embedding_token_count=10,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


def test_rebinding_cli_splits_non_contiguous_quote_and_recomputes_content_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The migration preserves fact identity while replacing one joined quote with exact spans."""
    facts_path = tmp_path / "legal_facts.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    fact = build_legal_fact(
        statement="Đơn vị kinh doanh vận tải vi phạm điều kiện giấy phép bị thu hồi giấy phép.",
        conduct="Vi phạm điều kiện giấy phép",
        subject="Đơn vị kinh doanh vận tải",
        conditions=(),
        monetary_penalty=None,
        license_points_deducted=None,
        additional_penalties=("Thu hồi giấy phép",),
        remedial_measures=(),
        valid_from=None,
        valid_to=None,
        document_number="158/2024/NĐ-CP",
        provision_path="Điều 31 khoản 1 điểm b",
        evidence=(EvidenceReference(chunk_id="chunk-1", quoted_span="Quy định chung.\n\nĐiểm b áp dụng."),),
        source_chunk_id="chunk-1",
    )
    write_legal_facts(facts_path, [fact])
    write_chunks(chunks_path, [_chunk("Quy định chung.\n\nĐiểm a áp dụng.\n\nĐiểm b áp dụng.")])
    monkeypatch.setattr(
        sys,
        "argv",
        ["legal-rebind-fact-evidence", str(facts_path), str(chunks_path)],
    )

    fact_evidence_rebinding_cli.main()

    rebound = read_legal_facts(facts_path)[0]
    assert rebound.fact_id == fact.fact_id
    assert rebound.fact_content_sha256 != fact.fact_content_sha256
    assert tuple(reference.quoted_span for reference in rebound.evidence) == ("Quy định chung.", "Điểm b áp dụng.")


def test_evidence_binding_rejects_reversed_paragraph_order() -> None:
    """Paragraphs must occur in the source in the same order as the model citation."""
    evidence = (EvidenceReference(chunk_id="chunk-1", quoted_span="Đoạn sau.\n\nĐoạn trước."),)

    with pytest.raises(ValueError, match="not exact source text"):
        bind_evidence_references(evidence, {"chunk-1": "Đoạn trước.\n\nĐoạn sau."})


def test_evidence_binding_uses_distinct_ordered_repeated_paragraphs() -> None:
    """Repeated cited paragraphs bind to successive source occurrences."""
    evidence = (EvidenceReference(chunk_id="chunk-1", quoted_span="Lặp lại.\n\nLặp lại."),)

    bound = bind_evidence_references(evidence, {"chunk-1": "Lặp lại.\n\nNội dung giữa.\n\nLặp lại."})

    assert bound == (
        EvidenceReference(chunk_id="chunk-1", quoted_span="Lặp lại."),
        EvidenceReference(chunk_id="chunk-1", quoted_span="Lặp lại."),
    )


def test_evidence_binding_preserves_exact_multiple_references() -> None:
    """Already exact multi-reference evidence remains unchanged."""
    evidence = (
        EvidenceReference(chunk_id="chunk-1", quoted_span="Đoạn một."),
        EvidenceReference(chunk_id="chunk-1", quoted_span="Đoạn hai."),
    )

    assert bind_evidence_references(evidence, {"chunk-1": "Đoạn một.\n\nĐoạn hai."}) == evidence
