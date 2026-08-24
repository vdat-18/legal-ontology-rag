"""Tests for the end-to-end single-turn candidate generation command."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from legal_rag import candidate_generation_cli
from legal_rag.baml_client.types import (
    CandidateGenerationBatchInput,
    GeneratedSingleTurnCandidateBatchItem,
)
from legal_rag.baml_client.types import ClaimReference as BamlClaimReference
from legal_rag.baml_client.types import GeneratedSingleTurnCandidate as BamlGeneratedCandidate
from legal_rag.benchmark_synthesis import candidate_generation
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario
from legal_rag.benchmark_synthesis.writer import (
    read_candidates,
    write_legal_facts,
    write_scenarios,
)

_REPO_CONFIG_DIR = Path(__file__).resolve().parent.parent / "data" / "config" / "groundtruth"
_REFERENCE_ANSWER = "Người điều khiển ô tô vượt đèn đỏ bị phạt tiền."


def _fact() -> LegalFact:
    """Build one approved-ready legal fact for CLI tests."""
    return LegalFact(
        fact_id="fact-1",
        statement="Người điều khiển ô tô vượt đèn đỏ bị phạt tiền.",
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        evidence=(EvidenceReference(chunk_id="chunk-1"),),
    )


def _scenario() -> Scenario:
    """Build one scenario referencing the approved fact."""
    return Scenario(
        scenario_id="scenario-1",
        query_form="specific",
        answerability="answerable",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        reasoning_hops=1,
        fact_ids=("fact-1",),
        bridge=None,
        gold_chunk_ids=("chunk-1",),
    )


class _FakeGenerationClient:
    """Return a fixed candidate grounded in the source fact."""

    def __init__(self, settings: object, synthesis_settings: object) -> None:
        return None

    def generate(self, *args: object, **kwargs: object) -> BamlGeneratedCandidate:
        """Return one candidate grounded in the real chunk text."""
        return BamlGeneratedCandidate(
            query_vi="oto vuot den do bi phat bao nhieu",
            reference_answer_vi=_REFERENCE_ANSWER,
            reasoning_vi="Theo Điều 6.",
            required_claims=[BamlClaimReference(claim="Mức phạt tiền áp dụng", fact_id="fact-1")],
        )

    def generate_batch(self, items: list[CandidateGenerationBatchInput]) -> list[GeneratedSingleTurnCandidateBatchItem]:
        """Generate one candidate for every batch item."""
        return [
            GeneratedSingleTurnCandidateBatchItem(task_id=item.task_id, candidate=self.generate()) for item in items
        ]


def _configure_cli(
    monkeypatch: pytest.MonkeyPatch,
    facts_path: Path,
    scenarios_path: Path,
    candidates_output_path: Path,
    *,
    candidates_per_scenario: int,
) -> None:
    """Set deterministic CLI arguments, environment, and fake LLM clients."""
    from legal_rag.benchmark_synthesis import llm_execution

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(llm_execution.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(candidate_generation, "BamlCandidateGenerationClient", _FakeGenerationClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-generate-candidates",
            str(facts_path),
            str(scenarios_path),
            str(candidates_output_path),
            "--personas",
            str(_REPO_CONFIG_DIR / "personas.json"),
            "--candidates-per-scenario",
            str(candidates_per_scenario),
            "--synthesis-config",
            str(_REPO_CONFIG_DIR / "synthesis_v1.json"),
        ],
    )


def test_candidate_generation_cli_produces_candidate_from_scenario(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A generated candidate is written through end to end with no gating checks."""
    facts_path = tmp_path / "legal_facts.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    candidates_path = tmp_path / "single_turn.jsonl"

    write_legal_facts(facts_path, [_fact()])
    write_scenarios(scenarios_path, [_scenario()])
    _configure_cli(monkeypatch, facts_path, scenarios_path, candidates_path, candidates_per_scenario=1)

    candidate_generation_cli.main()

    drafts = read_candidates(candidates_path)
    assert len(drafts) == 1
    assert drafts[0].answerability == "answerable"
    assert drafts[0].query_vi == "oto vuot den do bi phat bao nhieu"


def test_candidate_generation_cli_generates_multiple_attempts_per_scenario(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """candidates_per_scenario controls how many draws are made per scenario."""
    facts_path = tmp_path / "legal_facts.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    candidates_path = tmp_path / "single_turn.jsonl"

    write_legal_facts(facts_path, [_fact()])
    write_scenarios(scenarios_path, [_scenario()])
    _configure_cli(monkeypatch, facts_path, scenarios_path, candidates_path, candidates_per_scenario=3)

    candidate_generation_cli.main()

    drafts = read_candidates(candidates_path)
    assert len(drafts) == 3
