"""Tests for context builder."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest
from rdflib import URIRef

from legal_rag.naive_rag.models import RetrievedChunk
from legal_rag.ontology_rag.context.context_builder import ContextBuilder
from legal_rag.ontology_rag.models import PenaltyRule


def test_context_builder_creates_structured_context() -> None:
    """ContextBuilder creates structured JSON from rules."""
    rules = [
        PenaltyRule(
            uri=URIRef("rule_001"),
            conduct_label_vi="Không đội mũ bảo hiểm",
            vehicle_categories=("Xe máy",),
            subject_kind="NaturalPerson",
            roles=("Người lái xe",),
            monetary_penalty_min=Decimal("400000"),
            monetary_penalty_max=Decimal("600000"),
            currency="VND",
            point_value=2,
            valid_from=date(2021, 1, 1),
            valid_to=None,
            source_chunk_ids=("chunk_001",),
            provision_path="Điều 5",
            document_number="NĐ 17/2020",
        )
    ]
    chunks = [
        RetrievedChunk(chunk_id="chunk_002", text="fallback text", source_name="doc1", kind="paragraph", score=0.5)
    ]

    builder = ContextBuilder()
    context = builder.build(rules, chunks)

    assert "valid_penalty_rules" in context
    assert "fallback_chunks" in context
    assert len(context["valid_penalty_rules"]) == 1
    assert len(context["fallback_chunks"]) == 1


def test_context_builder_formats_consequences() -> None:
    """ContextBuilder formats monetary penalties correctly."""
    rules = [
        PenaltyRule(
            uri=URIRef("rule_001"),
            conduct_label_vi="Test",
            vehicle_categories=(),
            subject_kind=None,
            roles=(),
            monetary_penalty_min=Decimal("100000"),
            monetary_penalty_max=Decimal("200000"),
            currency="VND",
            point_value=None,
            valid_from=date(2020, 1, 1),
            valid_to=None,
            source_chunk_ids=(),
            provision_path="test",
            document_number="test",
        )
    ]

    builder = ContextBuilder()
    context = builder.build(rules, [])

    rule = context["valid_penalty_rules"][0]
    assert len(rule["consequences"]) == 1
    consequence = rule["consequences"][0]
    assert consequence["type"] == "MonetaryPenalty"
    assert "min_amount" in consequence
    assert "max_amount" in consequence


def test_answer_context_keeps_unknown_validity_and_exact_source_text() -> None:
    """Generation receives structured uncertainty beside the source wording it may quote."""
    rule = PenaltyRule(
        uri=URIRef("https://example.test/rule/1"),
        conduct_label_vi="Không đội mũ bảo hiểm",
        vehicle_categories=("Xe mô tô",),
        subject_kind="NaturalPerson",
        roles=("Người điều khiển",),
        monetary_penalty_min=Decimal("400000"),
        monetary_penalty_max=Decimal("600000"),
        currency="VND",
        point_value=2,
        valid_from=date(2025, 1, 1),
        valid_to=None,
        source_chunk_ids=("chunk_001",),
        provision_path="Điều 7 khoản 2",
        document_number="168/2024/NĐ-CP",
    )
    source = RetrievedChunk(
        chunk_id="chunk_001",
        text="Phạt tiền từ 400.000 đồng đến 600.000 đồng.",
        source_name="168_2024_ND-CP.docx",
        kind="paragraph_group",
        score=0.9,
    )

    [context] = ContextBuilder().build_answer_contexts(
        [rule],
        [source],
        {str(rule.uri): "indeterminate"},
    )

    assert '"valid_to_exclusive":null' in context.text
    assert '"status_for_query":"indeterminate"' in context.text
    assert "Phạt tiền từ 400.000 đồng đến 600.000 đồng." in context.text
    assert context.evidence_chunk_ids == ("chunk_001",)
    assert context.source_name == "Điều 7 khoản 2 — 168/2024/NĐ-CP"


def test_answer_context_budget_preserves_one_exact_source_per_rule() -> None:
    """Budgeting omits oversized extras while preserving rule evidence and small fallback."""
    base_rule = PenaltyRule(
        uri=URIRef("https://example.test/rule/1"),
        conduct_label_vi="Không đội mũ bảo hiểm",
        vehicle_categories=("Xe mô tô",),
        subject_kind="NaturalPerson",
        roles=("Người điều khiển",),
        monetary_penalty_min=Decimal("400000"),
        monetary_penalty_max=Decimal("600000"),
        currency="VND",
        point_value=None,
        valid_from=date(2025, 1, 1),
        valid_to=date(2027, 1, 1),
        source_chunk_ids=("rule-1-main", "rule-1-extra"),
        provision_path="Điều 7 khoản 2",
        document_number="168/2024/NĐ-CP",
    )
    second_rule = replace(
        base_rule,
        uri=URIRef("https://example.test/rule/2"),
        conduct_label_vi="Không chấp hành tín hiệu đèn",
        source_chunk_ids=("rule-2-main",),
    )
    chunks = [
        RetrievedChunk("rule-1-main", "Nguồn chính rule 1", "doc-1", "paragraph", 1.0),
        RetrievedChunk("rule-1-extra", "X" * 6_000, "doc-1", "paragraph", 0.9),
        RetrievedChunk("rule-2-main", "Nguồn chính rule 2", "doc-2", "paragraph", 1.0),
        RetrievedChunk("background", "Ngữ cảnh độc lập", "doc-3", "paragraph", 0.8),
    ]
    builder = ContextBuilder()

    contexts = builder.build_answer_contexts(
        [base_rule, second_rule],
        chunks,
        max_characters=5_000,
    )

    assert contexts == builder.build_answer_contexts(
        [base_rule, second_rule],
        chunks,
        max_characters=5_000,
    )
    assert "Nguồn chính rule 1" in contexts[0].text
    assert "Nguồn chính rule 2" in contexts[1].text
    assert contexts[0].evidence_chunk_ids == ("rule-1-main",)
    assert contexts[1].evidence_chunk_ids == ("rule-2-main",)
    assert contexts[-1].chunk_id == "background"
    assert "X" * 100 not in "".join(context.text for context in contexts)
    assert sum(len(context.chunk_id) + len(context.source_name) + len(context.text) for context in contexts) <= 5_000

    with pytest.raises(ValueError, match="minimum rule evidence"):
        builder.build_answer_contexts([base_rule, second_rule], chunks, max_characters=10)
