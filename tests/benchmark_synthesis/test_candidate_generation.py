"""Tests for grounded single-turn candidate generation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.baml_client.types import ClaimReference as BamlClaimReference
from legal_rag.baml_client.types import FactContext
from legal_rag.baml_client.types import GeneratedSingleTurnCandidate as BamlGeneratedCandidate
from legal_rag.benchmark_synthesis import candidate_generation
from legal_rag.benchmark_synthesis.candidate_generation import (
    BamlCandidateGenerationClient,
    build_fact_context,
    generate_candidate,
)
from legal_rag.benchmark_synthesis.config import LlmStageSettings, SynthesisSettings
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario
from legal_rag.settings import Settings

_SYNTHESIS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth" / "synthesis_v1.json"


def _fact(fact_id: str = "fact-1") -> LegalFact:
    """Build one minimal legal fact for candidate-generation tests."""
    return LegalFact(
        fact_id=fact_id,
        statement="Người điều khiển ô tô vượt đèn đỏ bị phạt tiền và trừ điểm.",
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        evidence=(EvidenceReference(chunk_id="chunk-1"), EvidenceReference(chunk_id="chunk-2")),
    )


def _scenario(fact_ids: tuple[str, ...] = ("fact-1",)) -> Scenario:
    """Build one minimal scenario for candidate-generation tests."""
    return Scenario(
        scenario_id="scenario-1",
        query_form="specific",
        answerability="answerable",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        reasoning_hops=len(fact_ids),
        fact_ids=fact_ids,
        bridge=None,
        gold_chunk_ids=("chunk-1", "chunk-2"),
    )


def test_build_fact_context_exposes_only_semantic_fact_fields() -> None:
    """The LLM receives fact content and metadata, while provenance stays in code."""
    context = build_fact_context(_fact())

    assert context.fact_id == "fact-1"
    assert context.statement == _fact().statement
    assert not hasattr(context, "evidence_chunk_ids")


def test_build_fact_context_accepts_general_legal_statement() -> None:
    """Candidate generation receives the complete legal statement without specialized fields."""
    context = build_fact_context(replace(_fact(), statement="Quyết định xử phạt phải được gửi đúng hạn."))

    assert context.statement == "Quyết định xử phạt phải được gửi đúng hạn."


class _FakeGenerationClient:
    """Return a pre-programmed generated candidate, ignoring LLM specifics."""

    def __init__(self, generated: BamlGeneratedCandidate) -> None:
        self._generated = generated

    def generate(self, *args: object, **kwargs: object) -> BamlGeneratedCandidate:
        """Return the configured generated candidate."""
        return self._generated


def test_generate_candidate_attaches_all_provenance_from_the_claimed_fact() -> None:
    """Claim provenance comes from the persisted fact rather than model-generated chunk IDs."""
    fact = _fact()
    generated = BamlGeneratedCandidate(
        query_vi="oto vuot den do bi phat bao nhieu",
        reference_answer_vi="Phạt tiền 4.000.000 đồng và trừ 4 điểm.",
        reasoning_vi="Fact xác định mức phạt và số điểm bị trừ.",
        required_claims=[BamlClaimReference(claim="Mức phạt tiền áp dụng", fact_id="fact-1")],
    )

    candidate = generate_candidate(
        _scenario(),
        {"fact-1": fact},
        "Tài xế cá nhân",
        _FakeGenerationClient(generated),
        generator_model="m",
    )

    assert [entry.chunk_id for entry in candidate.required_claims[0].evidence] == ["chunk-1", "chunk-2"]


def test_generate_candidate_carries_declared_scenario_labels_unchanged() -> None:
    """Scenario labels pass through to the draft exactly as declared."""
    fact = _fact()
    generated = BamlGeneratedCandidate(query_vi="q", reference_answer_vi="a", reasoning_vi="r", required_claims=[])
    scenario = _scenario()

    candidate = generate_candidate(
        scenario,
        {"fact-1": fact},
        "Tài xế cá nhân",
        _FakeGenerationClient(generated),
        generator_model="m",
    )

    assert candidate.persona == scenario.persona
    assert candidate.answerability == scenario.answerability
    assert candidate.reasoning_hops == scenario.reasoning_hops
    assert candidate.source_fact_ids == ("fact-1",)


def test_generate_candidate_sample_id_is_deterministic() -> None:
    """Two identical generation results produce the same sample_id."""
    fact = _fact()
    generated = BamlGeneratedCandidate(query_vi="q", reference_answer_vi="a", reasoning_vi="r", required_claims=[])
    client = _FakeGenerationClient(generated)

    first = generate_candidate(_scenario(), {"fact-1": fact}, "Tài xế cá nhân", client, generator_model="m")
    second = generate_candidate(_scenario(), {"fact-1": fact}, "Tài xế cá nhân", client, generator_model="m")

    assert first.sample_id == second.sample_id


def _settings() -> Settings:
    """Build runtime settings for an offline BAML request."""
    return Settings(
        llm_base_url="http://localhost:20128/v1",
        llm_api_key="provider-test-key",
        llm_model="provider/test-model",
        llm_timeout_seconds=30,
    )


def _synthesis_settings() -> SynthesisSettings:
    """Build ground-truth synthesis settings for an offline BAML request."""
    return SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)


def test_candidate_client_uses_generation_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Candidate generation uses its versioned stage configuration once."""
    configured_tokens: list[int] = []

    def _capture_registry(**kwargs: object) -> object:
        configured_tokens.append(int(kwargs["max_output_tokens"]))
        return object()

    settings = replace(_synthesis_settings(), candidate_generation=LlmStageSettings(101, 0.0))
    monkeypatch.setattr(candidate_generation, "build_client_registry", _capture_registry)

    BamlCandidateGenerationClient(_settings(), settings)

    assert configured_tokens == [101]


def test_baml_candidate_generation_client_parses_typed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated non-stream client returns a typed candidate."""
    client = BamlCandidateGenerationClient(_settings(), _synthesis_settings())
    monkeypatch.setattr(
        candidate_generation,
        "execute_baml_request",
        lambda *args, **kwargs: (
            '[{"task_id":"single","candidate":{"query_vi":"q","reference_answer_vi":"a","reasoning_vi":"r",'
            '"required_claims":[{"claim":"c","fact_id":"fact-1"}]}}]'
        ),
    )
    fact_context = FactContext(
        fact_id="fact-1",
        statement="Người điều khiển ô tô vượt đèn đỏ bị phạt tiền.",
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
    )

    result = client.generate(
        [fact_context],
        "Tài xế cá nhân",
        "standard",
        "short",
        "specific",
        "answerable",
        "single_hop",
        None,
        1,
    )

    assert result.query_vi == "q"
    assert result.required_claims[0].fact_id == "fact-1"
