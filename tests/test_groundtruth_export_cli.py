"""Tests for the evaluation ground-truth export command."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

from legal_rag import groundtruth_export_cli
from legal_rag.benchmark_synthesis.candidate_generation import CandidateDraft
from legal_rag.benchmark_synthesis.groundtruth_export import GROUND_TRUTH_EXPORT_SCHEMA_VERSION
from legal_rag.benchmark_synthesis.writer import read_candidates, write_candidates
from legal_rag.evaluation.metrics import load_jsonl_samples


def _candidate() -> CandidateDraft:
    """Build the smallest complete candidate accepted by the persisted schema."""
    return CandidateDraft(
        sample_id="sample-1",
        scenario_id="scenario-1",
        query_vi="Xe máy vượt đèn đỏ bị phạt bao nhiêu?",
        reference_answer_vi="Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.",
        reasoning_vi="Áp dụng mức phạt tại Điều 7.",
        required_claims=(),
        evolutions=(),
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
    )


def test_groundtruth_export_cli_writes_evaluation_ready_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The CLI exports every candidate atomically in input order."""
    candidates_path = tmp_path / "single_turn.jsonl"
    output_path = tmp_path / "ground_truth.jsonl"
    write_candidates(candidates_path, (_candidate(),))
    assert len(read_candidates(candidates_path)) == 1
    monkeypatch.setattr(
        sys,
        "argv",
        ["legal-export-groundtruth", str(candidates_path), str(output_path)],
    )

    groundtruth_export_cli.main()

    samples = load_jsonl_samples(output_path)
    assert len(samples) == 1
    assert samples[0].question == _candidate().query_vi
    assert f'"schema_version":"{GROUND_TRUTH_EXPORT_SCHEMA_VERSION}"' in output_path.read_text(encoding="utf-8")
