"""Tests for single-turn candidate generation, surface refinement, and evolution."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from legal_rag.baml_client.types import ClaimReference as BamlClaimReference
from legal_rag.baml_client.types import EvolutionOperation, FactContext
from legal_rag.baml_client.types import EvolvedCandidate as BamlEvolvedCandidate
from legal_rag.baml_client.types import GeneratedSingleTurnCandidate as BamlGeneratedCandidate
from legal_rag.baml_client.types import SurfaceCritique as BamlSurfaceCritique
from legal_rag.benchmark_synthesis import candidate_generation
from legal_rag.benchmark_synthesis.candidate_generation import (
    BamlCandidateGenerationClient,
    build_fact_context,
    evolve_candidate,
    generate_candidate,
    refine_surface,
)
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario
from legal_rag.settings import EvaluationSettings, Settings


def _fact(fact_id: str = "fact-1") -> LegalFact:
    """Build one minimal legal fact for candidate-generation tests."""
    return LegalFact(
        fact_id=fact_id,
        fact_content_sha256=f"content-{fact_id}",
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
            EvidenceReference(chunk_id="chunk-1", quoted_span="Phạt tiền 4.000.000 đồng.", unit_ids=(), source_refs=()),
            EvidenceReference(chunk_id="chunk-2", quoted_span="Trừ 4 điểm.", unit_ids=(), source_refs=()),
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
        bridge=(),
        gold_chunk_ids=("chunk-1", "chunk-2"),
    )


def test_build_fact_context_exposes_only_real_evidence_chunk_ids() -> None:
    """The BAML-facing fact context lists exactly the fact's own evidence chunk_ids."""
    context = build_fact_context(_fact())

    assert context.evidence_chunk_ids == ["chunk-1", "chunk-2"]
    assert context.fact_id == "fact-1"


class _FakeGenerationClient:
    """Return a pre-programmed generated candidate, ignoring LLM specifics."""

    def __init__(self, generated: BamlGeneratedCandidate) -> None:
        self._generated = generated

    def generate(self, *args: object, **kwargs: object) -> BamlGeneratedCandidate:
        """Return the configured generated candidate."""
        return self._generated

    def critique_surface(self, *args: object, **kwargs: object) -> BamlSurfaceCritique:
        """Unused in generation-only tests."""
        raise NotImplementedError

    def evolve(self, *args: object, **kwargs: object) -> BamlEvolvedCandidate:
        """Unused in generation-only tests."""
        raise NotImplementedError


def test_generate_candidate_cross_references_real_evidence_not_llm_copy() -> None:
    """quoted_span always comes from the fact's own evidence, never a value the LLM invents."""
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
    assert candidate.required_claims[0].evidence[0].quoted_span == "Phạt tiền 4.000.000 đồng."


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


def test_generate_candidate_answer_fields_are_copied_mechanically_from_fact() -> None:
    """answer_fields mirror the fact's own penalty fields exactly, not LLM output."""
    fact = _fact()
    generated = BamlGeneratedCandidate(query_vi="q", reference_answer_vi="a", reasoning_vi="r", required_claims=[])
    client = _FakeGenerationClient(generated)

    candidate = generate_candidate(
        _scenario(), {"fact-1": fact}, "Tài xế cá nhân", "Mức phạt của một hành vi", client, generator_model="m"
    )

    assert len(candidate.answer_fields) == 1
    assert candidate.answer_fields[0].fact_id == "fact-1"
    assert candidate.answer_fields[0].monetary_penalty == fact.monetary_penalty
    assert candidate.answer_fields[0].license_points_deducted == fact.license_points_deducted


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
    assert candidate.source_fact_content_sha256 == (fact.fact_content_sha256,)


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


class _FakeSurfaceClient:
    """Return pre-programmed critiques in sequence."""

    def __init__(self, critiques: Sequence[BamlSurfaceCritique]) -> None:
        self._critiques = list(critiques)
        self.calls = 0

    def generate(self, *args: object, **kwargs: object) -> BamlGeneratedCandidate:
        """Unused in surface-refinement tests."""
        raise NotImplementedError

    def critique_surface(self, *args: object, **kwargs: object) -> BamlSurfaceCritique:
        """Return the next configured critique."""
        critique = self._critiques[self.calls]
        self.calls += 1
        return critique

    def evolve(self, *args: object, **kwargs: object) -> BamlEvolvedCandidate:
        """Unused in surface-refinement tests."""
        raise NotImplementedError


def _draft_for_refinement() -> candidate_generation.CandidateDraft:
    """Build one candidate draft for surface-refinement and evolution tests."""
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


def test_refine_surface_stops_once_critique_passes() -> None:
    """The loop stops as soon as passes=True, without exhausting max_retries."""
    draft = _draft_for_refinement()
    client = _FakeSurfaceClient(
        [
            BamlSurfaceCritique(passes=False, feedback="quá dài", rewritten_query_vi="oto vuot den do"),
            BamlSurfaceCritique(passes=True, feedback="", rewritten_query_vi="oto vuot den do"),
        ]
    )

    refined = refine_surface(draft, client, max_retries=5)

    assert client.calls == 2
    assert refined.query_vi == "oto vuot den do"


def test_refine_surface_respects_max_retries_cap() -> None:
    """The loop never calls the client more than max_retries times."""
    draft = _draft_for_refinement()
    always_fails = [BamlSurfaceCritique(passes=False, feedback="x", rewritten_query_vi=f"v{i}") for i in range(10)]
    client = _FakeSurfaceClient(always_fails)

    refine_surface(draft, client, max_retries=3)

    assert client.calls == 3


def test_refine_surface_never_changes_claims_or_answer_fields() -> None:
    """Only query_vi changes; claims, evidence, and answer_fields stay frozen."""
    draft = _draft_for_refinement()
    client = _FakeSurfaceClient([BamlSurfaceCritique(passes=False, feedback="x", rewritten_query_vi="new query")])

    refined = refine_surface(draft, client, max_retries=1)

    assert refined.required_claims == draft.required_claims
    assert refined.answer_fields == draft.answer_fields
    assert refined.reference_answer_vi == draft.reference_answer_vi


class _FakeEvolutionClient:
    """Return a pre-programmed evolved query/answer for every operation."""

    def __init__(self, evolved_by_operation: dict[EvolutionOperation, BamlEvolvedCandidate]) -> None:
        self._evolved_by_operation = evolved_by_operation

    def generate(self, *args: object, **kwargs: object) -> BamlGeneratedCandidate:
        """Unused in evolution tests."""
        raise NotImplementedError

    def critique_surface(self, *args: object, **kwargs: object) -> BamlSurfaceCritique:
        """Unused in evolution tests."""
        raise NotImplementedError

    def evolve(self, query_vi: str, reference_answer_vi: str, operation: EvolutionOperation) -> BamlEvolvedCandidate:
        """Return the configured evolved candidate for this operation."""
        return self._evolved_by_operation[operation]


def test_evolve_candidate_applies_operations_in_sequence_and_records_them() -> None:
    """Each configured operation is applied once, in order, and recorded in evolutions."""
    draft = _draft_for_refinement()
    client = _FakeEvolutionClient(
        {
            EvolutionOperation.Concretizing: BamlEvolvedCandidate(
                query_vi="q-concrete", reference_answer_vi="a-concrete"
            ),
            EvolutionOperation.Reasoning: BamlEvolvedCandidate(
                query_vi="q-reasoning", reference_answer_vi="a-reasoning"
            ),
        }
    )

    evolved = evolve_candidate(draft, ["concretizing", "reasoning"], client)

    assert evolved.evolutions == ("concretizing", "reasoning")
    assert evolved.query_vi == "q-reasoning"
    assert evolved.reference_answer_vi == "a-reasoning"


def test_evolve_candidate_never_changes_claims_or_answer_fields() -> None:
    """Evolution only touches query_vi and reference_answer_vi, never evidence."""
    draft = _draft_for_refinement()
    client = _FakeEvolutionClient(
        {EvolutionOperation.Hypothetical: BamlEvolvedCandidate(query_vi="q-hyp", reference_answer_vi="a-hyp")}
    )

    evolved = evolve_candidate(draft, ["hypothetical"], client)

    assert evolved.required_claims == draft.required_claims
    assert evolved.answer_fields == draft.answer_fields


def _settings() -> Settings:
    """Build runtime settings for an offline BAML request."""
    return Settings(
        llm_base_url="http://localhost:20128/v1",
        llm_api_key="provider-test-key",
        llm_model="provider/test-model",
        llm_timeout_seconds=30,
    )


def _evaluation_settings() -> EvaluationSettings:
    """Build ground-truth synthesis settings for an offline BAML request."""
    return EvaluationSettings(
        embedding_profile="qwen3_0_6b",
        span_match_threshold=0.85,
        bundle_overlap_threshold=0.5,
        dedup_cosine_threshold=0.9,
        max_quality_retries=2,
        max_simulator_turns=8,
    )


def test_baml_candidate_generation_client_parses_typed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated non-stream client returns a typed candidate."""
    client = BamlCandidateGenerationClient(_settings(), _evaluation_settings())
    monkeypatch.setattr(
        candidate_generation,
        "execute_baml_request",
        lambda *args, **kwargs: (
            '{"query_vi":"q","reference_answer_vi":"a","reasoning_vi":"r",'
            '"required_claims":[{"claim":"c","fact_id":"fact-1","evidence_chunk_ids":["chunk-1"]}]}'
        ),
    )
    fact_context = FactContext(
        fact_id="fact-1",
        conduct="Vượt đèn đỏ",
        subject="Người điều khiển ô tô",
        conditions=[],
        monetary_penalty="4.000.000 đồng",
        license_points_deducted=None,
        additional_penalties=[],
        remedial_measures=[],
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6",
        evidence_chunk_ids=["chunk-1"],
    )

    result = client.generate([fact_context], "Tài xế cá nhân", "standard", "short", "specific", "Mức phạt", None)

    assert result.query_vi == "q"
    assert result.required_claims[0].fact_id == "fact-1"
