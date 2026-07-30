"""Tests for the end-to-end single-turn candidate generation command."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from legal_rag import candidate_generation_cli
from legal_rag.baml_client.types import Answerability, EvolutionOperation
from legal_rag.baml_client.types import ClaimReference as BamlClaimReference
from legal_rag.baml_client.types import EvolvedCandidate as BamlEvolvedCandidate
from legal_rag.baml_client.types import GeneratedSingleTurnCandidate as BamlGeneratedCandidate
from legal_rag.baml_client.types import SurfaceCritique as BamlSurfaceCritique
from legal_rag.baml_client.types import VerificationResult as BamlVerificationResult
from legal_rag.benchmark_synthesis import candidate_generation, verification
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario
from legal_rag.benchmark_synthesis.writer import (
    read_candidate_progress,
    read_candidates,
    write_legal_facts,
    write_scenarios,
)
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks

_REPO_CONFIG_DIR = Path(__file__).resolve().parent.parent / "data" / "config" / "groundtruth"
_PENALTY_SPAN = "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."


def _fact() -> LegalFact:
    """Build one approved-ready legal fact for CLI tests."""
    return LegalFact(
        fact_id="fact-1",
        fact_content_sha256="content-fact-1",
        conduct="Vượt đèn đỏ",
        subject="Người điều khiển ô tô",
        conditions=(),
        monetary_penalty="4.000.000 đồng đến 6.000.000 đồng",
        license_points_deducted=None,
        additional_penalties=(),
        remedial_measures=(),
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6",
        evidence=(EvidenceReference(chunk_id="chunk-1", quoted_span=_PENALTY_SPAN),),
    )


def _chunk() -> Chunk:
    """Build one final chunk backing the fact's evidence."""
    return Chunk(
        chunk_id="chunk-1",
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
        text=_PENALTY_SPAN,
        context_text="",
        embedding_text=_PENALTY_SPAN,
        text_token_count=5,
        embedding_token_count=5,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


def _scenario() -> Scenario:
    """Build one scenario referencing the approved fact."""
    return Scenario(
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


class _FakeGenerationClient:
    """Return a fixed candidate whose evidence is a real, exact quote from the chunk."""

    def __init__(self, settings: object, synthesis_settings: object) -> None:
        return None

    def generate(self, *args: object, **kwargs: object) -> BamlGeneratedCandidate:
        """Return one candidate grounded in the real chunk text."""
        return BamlGeneratedCandidate(
            query_vi="oto vuot den do bi phat bao nhieu",
            reference_answer_vi=_PENALTY_SPAN,
            reasoning_vi="Theo Điều 6.",
            required_claims=[
                BamlClaimReference(claim="Mức phạt tiền áp dụng", fact_id="fact-1", evidence_chunk_ids=["chunk-1"]),
            ],
        )

    def critique_surface(self, *args: object, **kwargs: object) -> BamlSurfaceCritique:
        """Always accept the surface phrasing immediately."""
        return BamlSurfaceCritique(passes=True, feedback="", rewritten_query_vi=args[0])

    def evolve(
        self, query_vi: str, reference_answer_vi: str, reasoning_vi: str, operation: EvolutionOperation
    ) -> BamlEvolvedCandidate:
        """Return the query and answer unchanged, tagged by operation."""
        return BamlEvolvedCandidate(
            query_vi=query_vi, reference_answer_vi=reference_answer_vi, reasoning_vi=reasoning_vi
        )


class _FakeVerificationClient:
    """Return a fixed passing verification result, ignoring call arguments."""

    def __init__(self, settings: object, synthesis_settings: object) -> None:
        return None

    def verify(self, *args: object, **kwargs: object) -> BamlVerificationResult:
        """Return a verification result where every dimension passes."""
        return BamlVerificationResult(
            is_standalone=True,
            is_specific=True,
            is_question_grounded=True,
            is_answer_grounded=True,
            is_domain_relevant=True,
            is_temporal_correct=True,
            uses_required_bridge=True,
            necessary_reasoning_hops=1,
            observed_query_type="penalty_for_conduct",
            observed_answerability=Answerability.Answerable,
            reasoning="Đầy đủ căn cứ.",
        )


def _configure_cli(
    monkeypatch: pytest.MonkeyPatch,
    facts_path: Path,
    scenarios_path: Path,
    chunks_dir: Path,
    candidates_output_path: Path,
    *,
    candidates_per_scenario: int,
) -> None:
    """Set deterministic CLI arguments, environment, and fake LLM clients."""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(candidate_generation, "BamlCandidateGenerationClient", _FakeGenerationClient)
    monkeypatch.setattr(verification, "BamlVerificationClient", _FakeVerificationClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-generate-candidates",
            str(facts_path),
            str(scenarios_path),
            str(chunks_dir),
            str(candidates_output_path),
            "--query-taxonomy",
            str(_REPO_CONFIG_DIR / "query_taxonomy.json"),
            "--personas",
            str(_REPO_CONFIG_DIR / "personas.json"),
            "--evolution",
            str(_REPO_CONFIG_DIR / "evolution.json"),
            "--candidates-per-scenario",
            str(candidates_per_scenario),
            "--synthesis-config",
            str(_REPO_CONFIG_DIR / "synthesis_v1.json"),
        ],
    )


def test_candidate_generation_cli_produces_automatically_verified_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A well-grounded generated candidate passes both gates end to end."""
    facts_path = tmp_path / "legal_facts.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    candidates_path = tmp_path / "single_turn.jsonl"

    write_legal_facts(facts_path, [_fact()])
    write_scenarios(scenarios_path, [_scenario()])
    write_chunks(chunks_dir / "168_2024_ND-CP.chunks.jsonl", [_chunk()])
    _configure_cli(monkeypatch, facts_path, scenarios_path, chunks_dir, candidates_path, candidates_per_scenario=1)

    candidate_generation_cli.main()

    records = read_candidates(candidates_path)
    assert len(records) == 1
    assert records[0].binding.passed is True
    assert records[0].review_status == "automatically_verified"
    assert records[0].verification is not None
    assert records[0].verification.observed_answerability == "answerable"


def test_candidate_generation_cli_generates_multiple_attempts_per_scenario(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """candidates_per_scenario controls how many draws are made per scenario."""
    facts_path = tmp_path / "legal_facts.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    candidates_path = tmp_path / "single_turn.jsonl"

    write_legal_facts(facts_path, [_fact()])
    write_scenarios(scenarios_path, [_scenario()])
    write_chunks(chunks_dir / "168_2024_ND-CP.chunks.jsonl", [_chunk()])
    _configure_cli(monkeypatch, facts_path, scenarios_path, chunks_dir, candidates_path, candidates_per_scenario=3)

    candidate_generation_cli.main()

    records = read_candidates(candidates_path)
    assert len(records) == 3


def test_candidate_generation_cli_resumes_after_failed_evolution_without_repeating_prior_llm_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Generation and critique checkpoints survive an exhausted evolution task."""
    facts_path = tmp_path / "legal_facts.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    candidates_path = tmp_path / "single_turn.jsonl"
    progress_path = tmp_path / "single_turn.progress.jsonl"
    write_legal_facts(facts_path, [_fact()])
    write_scenarios(scenarios_path, [_scenario()])
    write_chunks(chunks_dir / "168_2024_ND-CP.chunks.jsonl", [_chunk()])

    class _EvolutionFailsClient(_FakeGenerationClient):
        """Complete generation and critique, then exhaust evolution retries."""

        def evolve(
            self, query_vi: str, reference_answer_vi: str, reasoning_vi: str, operation: EvolutionOperation
        ) -> BamlEvolvedCandidate:
            msg = "evolution unavailable"
            raise TimeoutError(msg)

    _configure_cli(monkeypatch, facts_path, scenarios_path, chunks_dir, candidates_path, candidates_per_scenario=1)
    monkeypatch.setattr(candidate_generation, "BamlCandidateGenerationClient", _EvolutionFailsClient)

    with pytest.raises(RuntimeError, match="candidate generation left 1 incomplete"):
        candidate_generation_cli.main()

    progress = read_candidate_progress(progress_path)
    assert len(progress) == 1
    assert progress[0].completed is False
    assert progress[0].draft.surface_refined is True
    assert progress[0].draft.evolutions == ()

    evolve_calls: list[EvolutionOperation] = []

    class _ResumeAtEvolutionClient(_FakeGenerationClient):
        """Fail if completed stages repeat and finish the remaining evolutions."""

        def generate(self, *args: object, **kwargs: object) -> BamlGeneratedCandidate:
            msg = "generation must be resumed from checkpoint"
            raise AssertionError(msg)

        def critique_surface(self, *args: object, **kwargs: object) -> BamlSurfaceCritique:
            msg = "critique must be resumed from checkpoint"
            raise AssertionError(msg)

        def evolve(
            self, query_vi: str, reference_answer_vi: str, reasoning_vi: str, operation: EvolutionOperation
        ) -> BamlEvolvedCandidate:
            evolve_calls.append(operation)
            return BamlEvolvedCandidate(
                query_vi=query_vi,
                reference_answer_vi=reference_answer_vi,
                reasoning_vi=reasoning_vi,
            )

    monkeypatch.setattr(candidate_generation, "BamlCandidateGenerationClient", _ResumeAtEvolutionClient)
    candidate_generation_cli.main()

    assert evolve_calls == [EvolutionOperation.Concretizing, EvolutionOperation.Reasoning]
    assert read_candidate_progress(progress_path)[0].completed is True
    assert len(read_candidates(candidates_path)) == 1
