"""Tests for single-turn candidate generation and evolution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.baml_client.types import ClaimReference as BamlClaimReference
from legal_rag.baml_client.types import EvolutionOperation, FactContext
from legal_rag.baml_client.types import EvolvedCandidate as BamlEvolvedCandidate
from legal_rag.baml_client.types import GeneratedSingleTurnCandidate as BamlGeneratedCandidate
from legal_rag.benchmark_synthesis import candidate_generation
from legal_rag.benchmark_synthesis.candidate_generation import (
    BamlCandidateGenerationClient,
    build_fact_context,
    evolve_candidate,
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
        evidence=(
            EvidenceReference(chunk_id="chunk-1"),
            EvidenceReference(chunk_id="chunk-2"),
        ),
    )


def _scenario(fact_ids: tuple[str, ...] = ("fact-1",)) -> Scenario:
    """Build one minimal scenario for candidate-generation tests."""
    return Scenario(
        scenario_id="scenario-1",
        query_type="penalty_for_conduct",
        query_form="specific",
        answerability="answerable",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        reasoning_hops=len(fact_ids),
        as_of_date=None,
        fact_ids=fact_ids,
        bridge=None,
        gold_chunk_ids=("chunk-1", "chunk-2"),
    )


def test_build_fact_context_exposes_only_real_evidence_chunk_ids() -> None:
    """The BAML-facing fact context lists exactly the fact's own evidence chunk_ids."""
    context = build_fact_context(_fact())

    assert context.evidence_chunk_ids == ["chunk-1", "chunk-2"]
    assert context.fact_id == "fact-1"
    assert context.statement == _fact().statement


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

    def evolve(self, *args: object, **kwargs: object) -> BamlEvolvedCandidate:
        """Unused in generation-only tests."""
        raise NotImplementedError


def test_generate_candidate_cross_references_real_evidence_chunks() -> None:
    """Required claims retain only chunk ids present in the fact's evidence."""
    fact = _fact()
    generated = BamlGeneratedCandidate(
        query_vi="oto vuot den do bi phat bao nhieu",
        reference_answer_vi="Phạt tiền 4.000.000 đồng và trừ 4 điểm.",
        reasoning_vi="Theo Điều 6.",
        required_claims=[
            BamlClaimReference(claim="Mức phạt tiền áp dụng", fact_id="fact-1", evidence_chunk_ids=["chunk-1"]),
        ],
    )
    client = _FakeGenerationClient(generated)

    candidate = generate_candidate(
        _scenario(), {"fact-1": fact}, "Tài xế cá nhân", "Mức phạt của một hành vi", client, generator_model="m"
    )

    assert len(candidate.required_claims) == 1
    assert len(candidate.required_claims[0].evidence) == 1
    assert candidate.required_claims[0].evidence[0].chunk_id == "chunk-1"


def test_generate_candidate_drops_hallucinated_chunk_id() -> None:
    """A claim referencing a chunk_id outside the fact's evidence yields no evidence for it."""
    fact = _fact()
    generated = BamlGeneratedCandidate(
        query_vi="q",
        reference_answer_vi="a",
        reasoning_vi="r",
        required_claims=[
            BamlClaimReference(claim="Claim bịa", fact_id="fact-1", evidence_chunk_ids=["chunk-does-not-exist"]),
        ],
    )
    client = _FakeGenerationClient(generated)

    candidate = generate_candidate(
        _scenario(), {"fact-1": fact}, "Tài xế cá nhân", "Mức phạt của một hành vi", client, generator_model="m"
    )

    assert candidate.required_claims[0].evidence == ()


def test_generate_candidate_carries_declared_scenario_labels_unchanged() -> None:
    """Scenario labels pass through to the draft exactly as declared."""
    fact = _fact()
    generated = BamlGeneratedCandidate(query_vi="q", reference_answer_vi="a", reasoning_vi="r", required_claims=[])
    client = _FakeGenerationClient(generated)
    scenario = _scenario()

    candidate = generate_candidate(
        scenario, {"fact-1": fact}, "Tài xế cá nhân", "Mức phạt của một hành vi", client, generator_model="m"
    )

    assert candidate.query_type == scenario.query_type
    assert candidate.persona == scenario.persona
    assert candidate.answerability == scenario.answerability
    assert candidate.reasoning_hops == scenario.reasoning_hops
    assert candidate.source_fact_ids == ("fact-1",)


def test_generate_candidate_sample_id_is_deterministic() -> None:
    """Two identical generation results produce the same sample_id."""
    fact = _fact()
    generated = BamlGeneratedCandidate(query_vi="q", reference_answer_vi="a", reasoning_vi="r", required_claims=[])
    client = _FakeGenerationClient(generated)

    first = generate_candidate(
        _scenario(), {"fact-1": fact}, "Tài xế cá nhân", "Mức phạt của một hành vi", client, generator_model="m"
    )
    second = generate_candidate(
        _scenario(), {"fact-1": fact}, "Tài xế cá nhân", "Mức phạt của một hành vi", client, generator_model="m"
    )

    assert first.sample_id == second.sample_id


def _draft_for_evolution() -> candidate_generation.CandidateDraft:
    """Build one candidate draft for evolution tests."""
    fact = _fact()
    generated = BamlGeneratedCandidate(
        query_vi="oto vuot den do bi phat bao nhieu",
        reference_answer_vi="Phạt tiền 4.000.000 đồng.",
        reasoning_vi="r",
        required_claims=[
            BamlClaimReference(claim="Mức phạt", fact_id="fact-1", evidence_chunk_ids=["chunk-1"]),
        ],
    )
    client = _FakeGenerationClient(generated)
    return generate_candidate(
        _scenario(), {"fact-1": fact}, "Tài xế cá nhân", "Mức phạt của một hành vi", client, generator_model="m"
    )


class _FakeEvolutionClient:
    """Return a pre-programmed evolved query/answer for every operation."""

    def __init__(self, evolved_by_operation: dict[EvolutionOperation, BamlEvolvedCandidate]) -> None:
        self._evolved_by_operation = evolved_by_operation

    def generate(self, *args: object, **kwargs: object) -> BamlGeneratedCandidate:
        """Unused in evolution tests."""
        raise NotImplementedError

    def evolve(
        self,
        query_vi: str,
        reference_answer_vi: str,
        reasoning_vi: str,
        operation: EvolutionOperation,
        query_length: str,
    ) -> BamlEvolvedCandidate:
        """Return the configured evolved candidate for this operation."""
        return self._evolved_by_operation[operation]


def test_evolve_candidate_applies_operations_in_sequence_and_records_them() -> None:
    """Each configured operation is applied once, in order, and recorded in evolutions."""
    draft = _draft_for_evolution()
    client = _FakeEvolutionClient(
        {
            EvolutionOperation.Concretizing: BamlEvolvedCandidate(
                query_vi="q-concrete", reference_answer_vi="a-concrete", reasoning_vi="r-concrete"
            ),
            EvolutionOperation.Reasoning: BamlEvolvedCandidate(
                query_vi="q-reasoning", reference_answer_vi="a-reasoning", reasoning_vi="r-reasoning"
            ),
        }
    )

    evolved = evolve_candidate(
        draft, ["concretizing", "reasoning"], client, llm_max_retries=0, llm_retry_delay_seconds=0
    )

    assert evolved.evolutions == ("concretizing", "reasoning")
    assert evolved.query_vi == "q-reasoning"
    assert evolved.reference_answer_vi == "a-reasoning"
    assert evolved.reasoning_vi == "r-reasoning"
    assert evolved.sample_id != draft.sample_id


def test_evolve_candidate_never_changes_claims() -> None:
    """Evolution only touches query_vi and reference_answer_vi, never evidence."""
    draft = _draft_for_evolution()
    client = _FakeEvolutionClient(
        {
            EvolutionOperation.Hypothetical: BamlEvolvedCandidate(
                query_vi="q-hyp", reference_answer_vi="a-hyp", reasoning_vi="r-hyp"
            )
        }
    )

    evolved = evolve_candidate(draft, ["hypothetical"], client, llm_max_retries=0, llm_retry_delay_seconds=0)

    assert evolved.required_claims == draft.required_claims


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


def test_candidate_client_uses_separate_generation_and_evolution_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each candidate LLM operation uses its own versioned stage configuration."""
    configured_tokens: list[int] = []

    def _capture_registry(**kwargs: object) -> object:
        configured_tokens.append(int(kwargs["max_output_tokens"]))
        return object()

    settings = replace(
        _synthesis_settings(),
        candidate_generation=LlmStageSettings(101, 0.0),
        candidate_evolution=LlmStageSettings(103, 0.0),
    )
    monkeypatch.setattr(candidate_generation, "build_client_registry", _capture_registry)

    BamlCandidateGenerationClient(_settings(), settings)

    assert configured_tokens == [101, 103]


def test_baml_candidate_generation_client_parses_typed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated non-stream client returns a typed candidate."""
    client = BamlCandidateGenerationClient(_settings(), _synthesis_settings())
    monkeypatch.setattr(
        candidate_generation,
        "execute_baml_request",
        lambda *args, **kwargs: (
            '[{"task_id":"single","candidate":{"query_vi":"q","reference_answer_vi":"a","reasoning_vi":"r",'
            '"required_claims":[{"claim":"c","fact_id":"fact-1","evidence_chunk_ids":["chunk-1"]}]}}]'
        ),
    )
    fact_context = FactContext(
        fact_id="fact-1",
        statement="Người điều khiển ô tô vượt đèn đỏ bị phạt tiền.",
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        evidence_chunk_ids=["chunk-1"],
    )

    result = client.generate(
        [fact_context],
        "Tài xế cá nhân",
        "standard",
        "short",
        "specific",
        "Mức phạt",
        "answerable",
        "single_hop",
        None,
        1,
        None,
    )

    assert result.query_vi == "q"
    assert result.required_claims[0].fact_id == "fact-1"
