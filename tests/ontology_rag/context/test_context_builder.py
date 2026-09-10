"""Tests for context builder."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

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
    assert consequence["type"] == "monetary_penalty"
    assert "min_amount" in consequence
    assert "max_amount" in consequence
