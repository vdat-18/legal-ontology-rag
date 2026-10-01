"""Tests for exporting synthesized candidates as evaluation ground truth."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.candidate_generation import (
    CandidateDraft,
    ClaimEvidence,
    RequiredClaim,
)
from legal_rag.benchmark_synthesis.groundtruth_export import GROUND_TRUTH_EXPORT_SCHEMA_VERSION, export_ground_truth
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.writer import read_legal_facts
from legal_rag.evaluation.metrics import EvaluationSample


def _candidate() -> CandidateDraft:
    """Build one representative synthesized candidate."""
    return CandidateDraft(
        sample_id="sample-1",
        scenario_id="scenario-1",
        query="Xe máy vượt đèn đỏ bị phạt bao nhiêu?",
        reference_answer="Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.",
        reasoning="Áp dụng mức phạt tại Điều 7.",
        required_claims=(
            RequiredClaim(
                claim="Mức phạt tiền",
                fact_id="fact-1",
                evidence=(ClaimEvidence(chunk_id="chunk-1"),),
            ),
        ),
        gold_chunk_ids=("chunk-1",),
        query_form="specific",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        answerability="answerable",
        reasoning_hops=1,
        source_fact_ids=("fact-1",),
        generator_model="provider/model",
        bridge="Cùng hành vi vượt đèn đỏ",
    )


def test_export_ground_truth_maps_evaluation_fields_and_preserves_metadata() -> None:
    """Exported records are directly readable by the evaluation schema."""
    candidate = _candidate()

    record = export_ground_truth((candidate,))[0]
    sample = EvaluationSample.from_mapping(record)

    assert sample.question == candidate.query
    assert sample.expected_answer == candidate.reference_answer
    assert sample.expected_chunk_ids == candidate.gold_chunk_ids
    assert sample.bundle_kind == candidate.bundle_kind
    assert sample.answerability == candidate.answerability
    assert sample.reasoning_hops == candidate.reasoning_hops
    assert sample.query_style == candidate.query_style
    assert sample.expected_claims[0].claim == candidate.required_claims[0].claim
    assert record["question_id"] == candidate.sample_id
    assert record["answerability"] == candidate.answerability
    assert record["required_claims"] == [candidate.required_claims[0].to_dict()]
    assert "query_type" not in record
    assert "as_of_date" not in record
    assert record["bridge"] == "Cùng hành vi vượt đèn đỏ"
    assert record["query_dates"] == []


def test_export_ground_truth_persists_every_temporal_source_version_date() -> None:
    """Temporal comparisons carry authoritative version starts for plural retrieval and judging."""
    candidate = replace(
        _candidate(),
        bundle_kind="temporal",
        query="So sánh quy định cũ và mới",
        source_fact_ids=("old", "new"),
    )
    facts = {
        fact.fact_id: fact
        for fact in (
            LegalFact("old", "old rule", date(2025, 1, 1), None, "old", (EvidenceReference("chunk-1"),)),
            LegalFact("new", "new rule", date(2026, 3, 1), None, "new", (EvidenceReference("chunk-2"),)),
        )
    }

    [record] = export_ground_truth((candidate,), facts)

    assert record["query_dates"] == ["2025-01-01", "2026-03-01"]


def test_export_ground_truth_rejects_missing_temporal_source_fact() -> None:
    """A temporal target cannot silently omit an authoritative source version."""
    candidate = replace(_candidate(), bundle_kind="temporal", source_fact_ids=("missing",))

    with pytest.raises(ValueError, match="missing source_fact_ids"):
        export_ground_truth((candidate,), {})


def test_default_ground_truth_persists_temporal_source_version_dates() -> None:
    """The committed benchmark, not only future exports, carries plural temporal targets."""
    repository_root = Path(__file__).resolve().parents[2]
    ground_truth_path = repository_root / "data/benchmark_synthesis/groundtruth.jsonl"
    facts_path = repository_root / "data/benchmark_synthesis/legal_facts.jsonl"
    rows = [json.loads(line) for line in ground_truth_path.read_text(encoding="utf-8").splitlines()]
    facts_by_id = {fact.fact_id: fact for fact in read_legal_facts(facts_path)}
    temporal_rows = [row for row in rows if row.get("bundle_kind") == "temporal"]

    assert temporal_rows
    assert all(row["schema_version"] == GROUND_TRUTH_EXPORT_SCHEMA_VERSION for row in rows)
    for row in temporal_rows:
        persisted = set(row["query_dates"])
        source_dates = {
            fact.valid_from.isoformat()
            for fact_id in row["source_fact_ids"]
            if (fact := facts_by_id[fact_id]).valid_from is not None
        }
        assert source_dates <= persisted
    assert any(len(row["query_dates"]) > 1 for row in temporal_rows)
